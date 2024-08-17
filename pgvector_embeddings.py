#!/usr/bin/env python

import torch
import re, sys, os, time, random
from transformers import BertTokenizer, BertModel
import logging
import psycopg2
from psycopg2.errors import SerializationFailure
import sqlalchemy
from sqlalchemy import create_engine, text, event, insert, Table, MetaData
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.dialects.postgresql import JSONB
import numpy as np
from sklearn.cluster import KMeans
import joblib
from flask import Flask, request, Response, g
import urllib
import json
import base64
from functools import lru_cache
import uuid
import os.path

CHARSET = "utf-8"
kmeans_model = None

batch_size = int(os.environ.get("BATCH_SIZE", "512"))
print("batch_size: {} (set via 'export BATCH_SIZE=512')".format(batch_size))

n_clusters = int(os.environ.get("N_CLUSTERS", "50"))
print("n_clusters : {} (set via 'export N_CLUSTERS=50')".format(n_clusters))

train_fraction = float(os.environ.get("TRAIN_FRACTION", "0.10"))
print("train_fraction: {} (set via 'export TRAIN_FRACTION=0.10')".format(train_fraction))

model_file = os.environ.get("MODEL_FILE", "model.pkl")
print("model_file: {} (set via 'export MODEL_FILE=./model.pkl')".format(model_file))

min_sentence_len = int(os.environ.get("MIN_SENTENCE_LEN", "8"))
print("min_sentence_len: {} (set via 'export MIN_SENTENCE_LEN=12')".format(min_sentence_len))

cache_size = int(os.environ.get("CACHE_SIZE", "1024"))
print("cache_size: {} (set via 'export CACHE_SIZE=1024')".format(cache_size))

n_threads = int(os.environ.get("N_THREADS", "1"))
print("n_threads: {} (set via 'export N_THREADS=10')".format(n_threads))

max_retries = int(os.environ.get("MAX_RETRIES", "3"))
print("max_retries: {} (set via 'export MAX_RETRIES=3')".format(max_retries))

log_level = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
  level=log_level
  , format="[%(asctime)s] %(message)s"
  , datefmt="%m/%d/%Y %I:%M:%S %p"
)
print("Log level: {} (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)".format(log_level))

db_url = os.getenv("DB_URL")
if db_url is None:
  print("DB_URL must be set")
  sys.exit(1)

if len(sys.argv) < 2:
  print("Usage: {} file [file2 ...]".format(sys.argv[0]))
  sys.exit(1)

db_url = re.sub(r"^postgres(ql)?", "cockroachdb", db_url)
engine = create_engine(db_url, pool_size=20, pool_pre_ping=True, connect_args = { "application_name": "CRDB Embeddings" })

secret = uuid.uuid4().hex
logging.warning("shared secret: {}".format(secret))

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
  cur = dbapi_connection.cursor()
  cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED;")
  cur.execute("SET plan_cache_mode = auto;")
  #cur.execute("SET default_transaction_use_follower_reads = on;")
  cur.close()

t0 = time.time()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
et = time.time() - t0
logging.info("BertTokenizer: {} s".format(et))

"""

Need to store/load via S3 or other object store

https://gist.github.com/aabadie/074587354d97d872aff6abb65510f618?permalink_comment_id=3892137
https://stackoverflow.com/questions/51921142/how-to-load-a-model-saved-in-joblib-file-from-google-cloud-storage-bucket

UPDATEable VIEWs: https://github.com/cockroachdb/cockroach/issues/20948#issuecomment-1603250501

"""

t0 = time.time()
# NOTE: I did *not* see any speedup running this on a GCP VM with nVidia T4 GPU.
# Install script for drivers on GCP VM:
#  https://github.com/GoogleCloudPlatform/compute-gpu-installation/blob/main/linux/startup_script.sh
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Model will run on {}".format(device))
# Set this up once and reuse
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True).to(device)
model.eval()
et = time.time() - t0
logging.info("BertModel + eval: {} s".format(et))

# The fist call to this takes ~ 500 ms but subsequent calls take ~ 40 ms
@lru_cache(maxsize=cache_size)
def gen_embeddings(s):
  rv = None
  marked_text = "[CLS] " + s + " [SEP]"
  tokenized_text = tokenizer.tokenize(marked_text)
  indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
  tokens_tensor = torch.tensor([indexed_tokens])
  segments_ids = [1] * len(tokenized_text)
  segments_tensors = torch.tensor([segments_ids])
  with torch.no_grad():
    if "cuda" == device:
      outputs = model(tokens_tensor.cuda(), segments_tensors.cuda())
    else:
      outputs = model(tokens_tensor, segments_tensors) # FIXME: exception here due to tensor size mismatch
    hidden_states = outputs[2]
  token_vecs = hidden_states[-2][0]
  sentence_embedding = torch.mean(token_vecs, dim=0)
  rv = sentence_embedding.tolist()
  return rv

ddl_t1 = """
CREATE TABLE text_embed
(
  uri STRING NOT NULL
  , chunk_num INT NOT NULL
  , chunk STRING NOT NULL
  , embedding VECTOR(768)
  , cluster_id INT
  , PRIMARY KEY (uri, chunk_num)
  , INDEX (cluster_id)
);
"""

ddl_t2 = """
CREATE TABLE cluster_assign
(
  uri STRING NOT NULL
  , chunk_num INT8 NOT NULL
  , cluster_id INT8 NOT NULL
  , PRIMARY KEY (uri, chunk_num)
  , INDEX (cluster_id ASC)
);
"""

ddl_t3 = """
CREATE TABLE cluster_assign_new
(
  uri STRING NOT NULL
  , chunk_num INT8 NOT NULL
  , cluster_id INT8 NOT NULL
  , PRIMARY KEY (uri, chunk_num)
  , INDEX (cluster_id ASC)
);
"""

ddl_view = """
CREATE OR REPLACE VIEW te_ca_view
AS
(
  SELECT te.uri, te.chunk_num, te.chunk, te.embedding, c.cluster_id
  FROM text_embed te, cluster_assign c
  WHERE te.uri = c.uri AND te.chunk_num = c.chunk_num
);
"""

sql_check_exists = """
SELECT COUNT(*) n FROM information_schema.tables WHERE table_catalog = 'defaultdb' AND table_name = 'text_embed';
"""

text_embed_table = None # Will be set after running setup_db()
cluster_assign_table = None

def setup_db():
  logging.info("Checking whether text_embed table exists")
  n_rows = 0
  with engine.connect() as conn:
    rs = conn.execute(text(sql_check_exists))
    for row in rs:
      n_rows = row.n
  table_exists = (n_rows == 1)
  if not table_exists:
    logging.info("Creating text_embed tables and view ...")
    with engine.connect() as conn:
      conn.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))
      conn.execute(text(ddl_t1))
      conn.execute(text(ddl_t2))
      conn.execute(text(ddl_t3))
      conn.execute(text(ddl_view))
      conn.commit()
    logging.info("OK")
  else:
    logging.info("text_embed table already exists")

def index_text(uri, text):
  te_rows = []
  ca_rows = []
  n_chunk = 0
  for s in re.split(r"\.\s+", text): # Sentence based splitting: makes sense to me.
    s = s.strip()
    if (len(s) >= min_sentence_len):
      logging.debug("URI: {}, CHUNK_NUM: {}\nCHUNK: '{}'".format(uri, n_chunk, s))
      embed = gen_embeddings(s)
      row_map = {
         "uri": uri
         , "chunk_num": n_chunk
         , "chunk": s
         , "embedding": embed
      }
      te_rows.append(row_map)
      cluster_id = int(kmeans_model.predict([embed])[0])
      row_map = {
         "uri": uri
         , "chunk_num": n_chunk
         , "cluster_id": cluster_id
      }
      ca_rows.append(row_map)
      n_chunk += 1
  with engine.begin() as conn: # Same TXN for both table INSERTs
    conn.execute(insert(text_embed_table), te_rows)
    conn.execute(insert(cluster_assign_table), ca_rows)

def index_file(in_file):
  text = ""
  with open(in_file, mode="rt") as f:
    for line in f:
      text += line
  in_file = re.sub(r"\./", '', in_file) # Trim leading '/'
  retry(index_text, (in_file, text))

# Clean any special chars out of text
def clean_text(text):
  return re.sub(r"['\",{}]", "", text)

# Decode a base64 encoded value to a UTF-8 string
def decode(b64):
  b = base64.b64decode(b64)
  return b.decode(CHARSET).strip()

app = Flask(__name__)

def gen_sql():
  rv = """
WITH q_embed AS
(
  SELECT uri, chunk, embedding
  FROM te_ca_view
  WHERE cluster_id = :cluster_id
)
SELECT uri, 1 - (embedding <=> (:q_embed)::VECTOR) sim, chunk
FROM q_embed
ORDER BY sim DESC
LIMIT :limit
"""
  return rv

def verify_secret(s):
  err = None
  if s != secret:
    err = "Provided secret '{}' != expected value '{}'".format(s, secret)
    logging.warning(err)
  return err

# FIXME: once the inserts are finished, switch the view definition to the new table
def refresh_cluster_assignments(s):
  err = verify_secret(s)
  if err is not None:
    return Response(err, status=400, mimetype="text/plain")
  select_sql = """
  SELECT uri, chunk_num, embedding
  FROM text_embed
  ORDER BY 1, 2
  """
  t0 = time.time()
  stmt = text(select_sql)
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION AS OF SYSTEM TIME '-10s';"))
    rs = conn.execute(stmt)
    ins_list = []
    if rs is not None:
      for row in rs:
        (uri, chunk_num, embed) = row
        embed = [float(x) for x in embed[1:-1].split(',')]
        cluster_id = int(kmeans_model.predict([embed])[0])
        row_map = {
          "uri": uri
          , "chunk_num": chunk_num
          , "cluster_id": cluster_id
        }
        ins_list.append(row_map)
        if len(ins_list) == batch_size:
          logging.info("Inserting batch of {} rows".format(batch_size))
          with engine.begin() as conn_ins:
            conn_ins.execute(insert(cluster_assign_table_new), ins_list)
          ins_list = []
    # Finish the INSERTs
    if len(ins_list) > 0:
      with engine.begin() as conn_ins:
        conn_ins.execute(insert(cluster_assign_table_new), ins_list)
  et = time.time() - t0
  logging.info("Cluster assign time: {} ms".format(et * 1000))
  return Response("OK", status=200, mimetype="text/plain")

@app.route("/cluster_assign/<s>")
def cluster_assign(s):
  return retry(refresh_cluster_assignments, (s,))

@app.route("/build_model/<s>")
def build_model(s):
  global kmeans_model
  err = verify_secret(s)
  if err is not None:
    return Response(err, status=400, mimetype="text/plain")
  # Grab a sample of vectors
  sql = """
  SELECT embedding
  FROM text_embed
  WHERE random() < :fraction
  """
  t0 = time.time()
  stmt = text(sql).bindparams(fraction=train_fraction)
  sampled_vecs = []
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION AS OF SYSTEM TIME '-10s';"))
    rs = conn.execute(stmt)
    if rs is not None:
      for row in rs:
        sampled_vecs.append([float(x) for x in row[0][1:-1].split(',')]) # Convert strings to float
  et = time.time() - t0
  logging.info("SQL query time: {} ms".format(et * 1000))
  # Train model over this sample
  kmeans = KMeans(
    init="random",
    n_clusters=n_clusters,
    n_init=10,
    max_iter=300,
    random_state=137
  )
  t0 = time.time()
  model = kmeans.fit(sampled_vecs)
  et = time.time() - t0
  logging.info("Model build time: {} ms".format(et * 1000))
  # Store the model to the filesystem
  joblib.dump(model, model_file)
  # Reload the in-memory copy of the model
  kmeans_model = model
  return Response("OK", status=200, mimetype="text/plain")

# Arg: search terms
# Returns: list of {"uri": uri, "sim": sim, "token": token, "chunk": chunk}
def search(terms, limit):
  q = ' '.join(terms)
  rv = []
  embed = gen_embeddings(q)
  cluster_id = int(kmeans_model.predict([embed])[0])
  logging.info("Query string: '{}'".format(q))
  logging.info("Cluster ID: {}".format(cluster_id))
  t0 = time.time()
  stmt = text(gen_sql()).bindparams(q_embed=embed, cluster_id=cluster_id, limit=limit)
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION AS OF SYSTEM TIME '-10s';"))
    rs = conn.execute(stmt)
    if rs is not None:
      for row in rs:
        (uri, sim, chunk) = row
        rv.append({"uri": uri, "sim": float(sim), "chunk": chunk})
  et = time.time() - t0
  logging.info("SQL query time: {} ms".format(et * 1000))
  return rv

# Retry wrapper for functions interacting with the DB
def retry(f, args):
  for retry in range(0, max_retries):
    if retry > 0:
      logging.warning("Retry number {}".format(retry))
    try:
      return f(*args)
    except SerializationFailure as e:
      logging.warning("Error: %s", e)
      logging.warning("EXECUTE SERIALIZATION_FAILURE BRANCH")
      sleep_s = (2**retry) * 0.1 * (random.random() + 0.5)
      logging.warning("Sleeping %s s", sleep_s)
      time.sleep(sleep_s)
    except (sqlalchemy.exc.OperationalError, psycopg2.OperationalError) as e:
      # Get a new connection and try again
      logging.warning("Error: %s", e)
      logging.warning("EXECUTE CONNECTION FAILURE BRANCH")
      sleep_s = 0.12 + random.random() * 0.25
      logging.warning("Sleeping %s s", sleep_s)
      time.sleep(sleep_s)
    except psycopg2.Error as e:
      logging.warning("Error: %s", e)
      logging.warning("EXECUTE DEFAULT BRANCH")
      raise e
  raise ValueError(f"Transaction did not succeed after {max_retries} retries")

# Verify transaction isolation level
def log_txn_isolation_level():
  txn_lvl = "Unknown"
  stmt = text("SHOW transaction_isolation;")
  with engine.connect() as conn:
    rs = conn.execute(stmt)
    cur.execute("SHOW transaction_isolation;")
    for row in rs:
      (txn_lvl) = row
  logging.info("transaction_isolation: {}".format(txn_lvl))

@app.route("/health", methods=["GET"])
def health():
  return Response("OK", status=200, mimetype="text/plain")

#
# The search/query
# EXAMPLE (with a limit of 10 results):
#   curl http://localhost:18080/search/$( echo -n "Using Lateral Joins" | base64 )
#
@app.route("/search/<q_base_64>/<int:limit>")
def do_search(q_base_64, limit):
  q = decode(q_base_64)
  q = clean_text(q)
  rv = retry(search, (q.split(), limit))
  logging.info(gen_embeddings.cache_info())
  return Response(json.dumps(rv), status=200, mimetype="application/json")

@app.route("/index", methods=["POST"])
def do_index():
  #log_txn_isolation_level()
  data = request.get_json(force=True)
  retry(index_text, (data["uri"], data["text"]))
  # Note the extra arguments here which translate the \uxxxx escape codes
  #print("Data: " + json.dumps(data, ensure_ascii=False).encode("utf8").decode())
  return Response("OK", status=200, mimetype="text/plain")

# Query mode (unlikely to get used)
if "-q" == sys.argv[1][0:2]:
  terms = sys.argv[2:]
  for row in search(terms):
    print("URI: {}\nSCORE: {}\nTOKEN: {}\nCHUNK: {}\n".format(row["uri"], row["sim"], row["token"], row["chunk"]))
# Server mode
elif "-s" == sys.argv[1][0:2]:
  if os.path.isfile(model_file): # Check to see if model already exists
    kmeans_model = joblib.load(model_file)
  else:
    logging.info("Building new K-means model")
    build_model(secret)
  logging.info("K-means model loaded")
  setup_db()
  text_embed_table = Table("text_embed", MetaData(), autoload_with=engine)
  cluster_assign_table = Table("cluster_assign", MetaData(), autoload_with=engine)
  cluster_assign_table_new = Table("cluster_assign_new", MetaData(), autoload_with=engine)
  port = int(os.getenv("FLASK_PORT", 18080))
  from waitress import serve
  serve(app, host="0.0.0.0", port=port, threads=n_threads)
# Indexing mode
else:
  print("Usage: {} [-s for server mode] [-q for query mode]\n".format(sys.argv[0]))
  sys.exit(1)

