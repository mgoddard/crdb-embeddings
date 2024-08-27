#!/usr/bin/env python3

import re, sys, os, time, random, io
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
import pickle
import requests
from fastembed import TextEmbedding
from pgvector.psycopg2 import register_vector

BLOCK_SIZE = 64 * (1 << 10) # Used when striping the model across > 1 row in blob_store
CHARSET = "utf-8"
kmeans_model = { "read": None, "write": None }

#KMEANS_DIM = 768 # Bert
KMEANS_DIM = 384 # Fastembed

kmeans_max_iter = int(os.environ.get("KMEANS_MAX_ITER", "100"))
print("kmeans_max_iter: {} (set via 'export KMEANS_MAX_ITER=25')".format(kmeans_max_iter))

kmeans_verbose = int(os.environ.get("KMEANS_VERBOSE", "0"))
print("kmeans_verbose: {} (set via 'export KMEANS_VERBOSE=1')".format(kmeans_verbose))

skip_kmeans = os.environ.get("SKIP_KMEANS", "False").lower() == "true"
print("skip_kmeans: {} (set via 'export SKIP_KMEANS=False')".format(skip_kmeans))

batch_size = int(os.environ.get("BATCH_SIZE", "512"))
print("batch_size: {} (set via 'export BATCH_SIZE=512')".format(batch_size))

n_clusters = int(os.environ.get("N_CLUSTERS", "500"))
print("n_clusters : {} (set via 'export N_CLUSTERS=50')".format(n_clusters))

train_fraction = float(os.environ.get("TRAIN_FRACTION", "0.5"))
print("train_fraction: {} (set via 'export TRAIN_FRACTION=0.10')".format(train_fraction))

model_file = os.environ.get("MODEL_FILE", "model.pkl")
print("model_file: {} (set via 'export MODEL_FILE=./model.pkl')".format(model_file))

model_url = os.environ.get("MODEL_FILE_URL", "https://storage.googleapis.com/crl-goddard-text/model.pkl")
print("model_url: {} (set via 'export MODEL_FILE_URL=https://somewhere.com/path/model.pkl')".format(model_url))

min_sentence_len = int(os.environ.get("MIN_SENTENCE_LEN", "8"))
print("min_sentence_len: {} (set via 'export MIN_SENTENCE_LEN=12')".format(min_sentence_len))

n_threads = int(os.environ.get("N_THREADS", "1"))
print("n_threads: {} (set via 'export N_THREADS=10')".format(n_threads))

max_retries = int(os.environ.get("MAX_RETRIES", "3"))
print("max_retries: {} (set via 'export MAX_RETRIES=3')".format(max_retries))

secret = os.environ.get("SECRET", uuid.uuid4().hex)
print("shared secret: {}".format(secret))

blob_store_keep_n_rows = os.environ.get("BLOB_STORE_KEEP_N_ROWS", "3")
print("blob_store_keep_n_rows: {}".format(blob_store_keep_n_rows))

log_level = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
  level=log_level
  , format="[%(asctime)s %(threadName)s] %(message)s"
  , datefmt="%m/%d/%Y %I:%M:%S %p"
)
print("Log level: {} (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)".format(log_level))

db_url = os.getenv("DB_URL")
if db_url is None:
  print("DB_URL must be set")
  sys.exit(1)

db_url = re.sub(r"^postgres(ql)?", "cockroachdb", db_url)
engine = create_engine(db_url, pool_size=20, pool_pre_ping=True, connect_args = { "application_name": "CRDB Embeddings" })

@event.listens_for(engine, "connect")
def connect(dbapi_connection, connection_record):
  register_vector(dbapi_connection)
  cur = dbapi_connection.cursor()
  cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED;")
  cur.execute("SET plan_cache_mode = auto;")
  cur.close()

t0 = time.time()
embed_model = TextEmbedding()
et = time.time() - t0
logging.info("TextEmbedding model ready: {} s".format(et))

# Used to download a model if none exists on FS or in DB
def download_file(url, local_fname):
  with requests.get(url, stream=True) as r:
    r.raise_for_status()
    with open(local_fname, "wb") as f:
      for chunk in r.iter_content(chunk_size=8192): 
        f.write(chunk)

ddl_t1 = """
CREATE TABLE text_embed
(
  uri STRING NOT NULL
  , chunk_num INT NOT NULL
  , chunk STRING NOT NULL
  , embedding VECTOR ({})
  , PRIMARY KEY (uri, chunk_num)
);
""".format(KMEANS_DIM)

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
CREATE TABLE {}
(
  uri STRING NOT NULL
  , chunk_num INT8 NOT NULL
  , cluster_id INT8 NOT NULL
  , PRIMARY KEY (uri, chunk_num)
  , INDEX (cluster_id ASC)
);
"""

ddl_t4 = """
CREATE TABLE blob_store
(
  path STRING NOT NULL
  , ts TIMESTAMP NOT NULL DEFAULT now()
  , n_row INT NOT NULL
  , blob BYTEA NOT NULL
  , PRIMARY KEY (path, ts, n_row)
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

def run_ddl(ddl):
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))
    conn.execute(text(ddl))
    conn.commit()

def prune_blob_store():
  logging.info("Pruning blob_store table ...")
  sql = """
  DELETE FROM blob_store
  WHERE (path, ts) IN
  (
    SELECT path, ts
    FROM blob_store
    GROUP BY 1, 2
    ORDER BY 2 DESC
    OFFSET {}
  );
  """
  with engine.connect() as conn:
    conn.execute(text(sql.format(blob_store_keep_n_rows)))
    conn.commit()
  logging.info("OK")

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
    run_ddl(ddl_t1)
    run_ddl(ddl_t2)
    run_ddl(ddl_t4)
    run_ddl(ddl_view)
    logging.info("OK")
  else:
    logging.info("text_embed table already exists")

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

# Return cluster ID value for embedding using "read" or "write" model
def get_cluster_id(rw, embed):
  return int(kmeans_model[rw].predict([embed])[0])

def index_text(uri, text):
  te_rows = []
  ca_rows = []
  n_chunk = 0
  s_list = []
  for s in re.split(r"\.\s+", text): # Sentence based splitting: makes sense to me.
    s = s.strip()
    if (len(s) >= min_sentence_len):
      s_list.append(s)
  t0 = time.time()
  embed_list = list(embed_model.embed(s_list))
  et = time.time() - t0
  logging.info("Time to generate embeddings(): {} ms".format(et * 1000))
  for i in range(0, len(s_list)):
    row_map = {
      "uri": uri
      , "chunk_num": n_chunk
      , "chunk": s_list[i]
      , "embedding": embed_list[i]
    }
    te_rows.append(row_map)
    if not skip_kmeans:
      cluster_id = get_cluster_id("write", embed_list[i])
      row_map = {
         "uri": uri
         , "chunk_num": n_chunk
         , "cluster_id": cluster_id
      }
      ca_rows.append(row_map)
    n_chunk += 1
  t0 = time.time()
  with engine.begin() as conn: # Same TXN for both table INSERTs
    try:
      conn.execute(insert(text_embed_table), te_rows)
    except sqlalchemy.exc.IntegrityError as e:
      return e.orig.diag.message_detail
    if not skip_kmeans:
      conn.execute(insert(cluster_assign_table), ca_rows)
    conn.commit()
  et = time.time() - t0
  logging.info("DB INSERT time: {} ms".format(et * 1000))

def index_file(in_file):
  text = ""
  with open(in_file, mode="rt") as f:
    for line in f:
      text += line
  in_file = re.sub(r"\./", '', in_file) # Trim leading '/'
  return retry(index_text, (in_file, text))

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

def refresh_cluster_assignments(s):
  err = verify_secret(s)
  if err is not None:
    return Response(err, status=400, mimetype="text/plain")
  # Temporary table to insert mappings into
  temp_table_name = "cluster_assign_temp_{}".format(uuid.uuid4().hex)
  logging.info("Inserting cluster assignments into {}".format(temp_table_name))
  run_ddl(ddl_t3.format(temp_table_name))
  cluster_assign_table_new = Table(temp_table_name, MetaData(), autoload_with=engine, extend_existing=True)
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
        cluster_id = get_cluster_id("write", embed)
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
  # Swap the tables
  t0 = time.time()
  logging.info("Swapping the tables for cluster_assign ...")
  with engine.connect() as conn:
    conn.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))
    conn.execute(text("DROP VIEW te_ca_view;"))
    conn.execute(text("DROP TABLE cluster_assign;"))
    conn.execute(text("ALTER TABLE {} RENAME TO cluster_assign;".format(temp_table_name)))
    conn.execute(text(ddl_view))
    conn.commit()
  et = time.time() - t0
  logging.info("Table swap time: {} ms".format(et * 1000))
  kmeans_model["read"] = kmeans_model["write"] # Once cluster assignments are updated
  return Response("OK", status=200, mimetype="text/plain")

@app.route("/cluster_assign/<s>")
def cluster_assign(s):
  return retry(refresh_cluster_assignments, (s,))

# Store the model to the DB
def store_model_in_db(mdl):
  rows = []
  orig_io = io.BytesIO(pickle.dumps(mdl))
  chunk = orig_io.read(BLOCK_SIZE)
  while chunk:
    row_map = {
      "path": model_file
      , "n_row": len(rows)
      , "blob": chunk
    }
    rows.append(row_map)
    chunk = orig_io.read(BLOCK_SIZE)
  with engine.begin() as conn:
    conn.execute(insert(blob_table), rows)

@app.route("/build_model/<s>")
def build_model(s):
  global kmeans_model
  err = verify_secret(s)
  if err is not None:
    return Response(err, status=400, mimetype="text/plain")
  logging.info("Getting data sample for model build ...")
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
  kmeans = KMeans(
    n_clusters=n_clusters,
    random_state=137,
    init="random",
    n_init=10,
    max_iter=kmeans_max_iter,
    verbose=kmeans_verbose
  )
  logging.info("Starting model build ...")
  t0 = time.time()
  model = kmeans.fit(sampled_vecs)
  et = time.time() - t0
  logging.info("Model build time: {} ms".format(et * 1000))
  # Store the model to the filesystem
  joblib.dump(model, model_file)
  store_model_in_db(model)
  # Reload the in-memory copy of the model
  kmeans_model["write"] = model
  prune_blob_store()
  return Response("OK", status=200, mimetype="text/plain")

# Arg: search terms
# Returns: list of {"uri": uri, "sim": sim, "token": token, "chunk": chunk}
def search(terms, limit):
  q = ' '.join(terms)
  rv = []
  embed_list = list(embed_model.embed([q]))
  #embed_list = [x.tolist() for x in embed_list] # Source of memory leak?
  embed = embed_list[0]
  cluster_id = get_cluster_id("read", embed) # This works fine with the ndarray type
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
  return Response(json.dumps(rv), status=200, mimetype="application/json")

@app.route("/index", methods=["POST"])
def do_index():
  rv = Response("OK", status=200, mimetype="text/plain")
  #log_txn_isolation_level()
  data = request.get_json(force=True)
  msg = retry(index_text, (data["uri"], data["text"]))
  if msg is not None:
    rv = Response(msg, status=400, mimetype="text/plain")
  # Note the extra arguments here which translate the \uxxxx escape codes
  #print("Data: " + json.dumps(data, ensure_ascii=False).encode("utf8").decode())
  return rv

@app.route("/health", methods=["GET"])
def health():
  return Response("OK", status=200, mimetype="text/plain")

# Fetch most recent model from the DB
def get_model_from_db():
  logging.info("Fetching model from the DB ...")
  sql = """
  WITH u AS
  (
    SELECT path, ts
    FROM blob_store
    ORDER BY ts DESC
    LIMIT 1
  )
  SELECT b.blob blob
  FROM blob_store b, u
  WHERE b.path = u.path AND b.ts = u.ts
  ORDER BY b.n_row ASC;
  """
  rv = None
  buf = io.BytesIO()
  with engine.connect() as conn:
    rs = conn.execute(text(sql))
    for row in rs:
      buf.write(row.blob)
  blob = buf.getvalue()
  if len(blob) > 0:
    rv = pickle.loads(blob)
    logging.info("OK")
  else:
    logging.info("No model in the DB")
  return rv

# main()
setup_db()
text_embed_table = Table("text_embed", MetaData(), autoload_with=engine)
cluster_assign_table = Table("cluster_assign", MetaData(), autoload_with=engine)
blob_table = Table("blob_store", MetaData(), autoload_with=engine)

# Load the K-means model
if not skip_kmeans:
  model_from_db = get_model_from_db()
  if model_from_db is None:
    if not os.path.isfile(model_file):
      logging.info("Downloading bootstrap K-means file ...")
      logging.info("\tURL: {}".format(model_url))
      logging.info("\tLocal file: {}".format(model_file))
      download_file(model_url, model_file)
      logging.info("OK")
    # Now the file is on the local FS, so load it and store it
    kmeans_model["read"] = joblib.load(model_file)
    kmeans_model["write"] = kmeans_model["read"]
    store_model_in_db(kmeans_model["read"])
  else:
    kmeans_model["write"] = model_from_db
    kmeans_model["read"] = model_from_db
  logging.info("K-means model loaded")
  logging.info("You may need to update K-means cluster assignments by making a GET request to the /cluster_assign/{} endpoint.".format(secret))
else:
  logging.info("Skipping loading K-Means model")

port = int(os.getenv("FLASK_PORT", 18080))
from waitress import serve
serve(app, host="0.0.0.0", port=port, threads=n_threads)

