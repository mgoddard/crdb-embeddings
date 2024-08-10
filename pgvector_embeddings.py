#!/usr/bin/env python

import torch
import re, sys, os, time, random
from transformers import BertTokenizer, BertModel
import base36
import logging
import psycopg2
from psycopg2.errors import SerializationFailure
import sqlalchemy
from sqlalchemy import create_engine, text, event, insert, Table, MetaData
from sqlalchemy.sql.expression import bindparam
from sqlalchemy.dialects.postgresql import JSONB
import numpy as np

# For Flask app
from flask import Flask, request, Response, g
import urllib
import json
import base64
from functools import lru_cache

CHARSET = "utf-8"

# Number of array dims to discard as these appear too frequently to be useful
N_DISCARD = 1

TOP_N = int(os.environ.get("TOP_N", "256"))
print("TOP_N: {} (set via 'export TOP_N=32')".format(TOP_N))

# Smaller value => reduced table scan with && operator
TOP_N_Q = int(os.environ.get("TOP_N_Q", "4"))
print("TOP_N_Q: {} (set via 'export TOP_N_Q=8')".format(TOP_N_Q))

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

ddl_table = """
CREATE TABLE text_embed
(
  uri STRING NOT NULL
  , chunk_num INT NOT NULL
  , chunk STRING NOT NULL
  , embedding VECTOR(768)
  , top_n INT[]
  , PRIMARY KEY (uri, chunk_num)
  , INVERTED INDEX (top_n)
);
"""

overlap_func = """
CREATE OR REPLACE FUNCTION overlap(a INT[], b INT[])
RETURNS INT
IMMUTABLE LEAKPROOF
LANGUAGE SQL
AS $$
  SELECT COUNT(*)
  FROM (
    SELECT UNNEST(a) INTERSECT SELECT UNNEST(b)
  );
$$;
"""

sql_check_exists = """
SELECT COUNT(*) n FROM information_schema.tables WHERE table_catalog = 'defaultdb' AND table_name = 'text_embed';
"""

text_embed_table = None # Will be set after running setup_db()

def setup_db():
  logging.info("Checking whether text_embed table exists")
  n_rows = 0
  with engine.connect() as conn:
    rs = conn.execute(text(sql_check_exists))
    for row in rs:
      n_rows = row.n
  table_exists = (n_rows == 1)
  if not table_exists:
    logging.info("Creating text_embed table")
    with engine.connect() as conn:
      conn.execute(text("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;"))
      conn.execute(text(ddl_table))
      logging.info("Creating overlap UDF")
      conn.execute(text(overlap_func))
      conn.commit()
  else:
    logging.info("text_embed table already exists")

def gen_top_n(embed, n):
  embed_dict = { k: v for v, k in enumerate(embed) }
  rv = list(dict(sorted(embed_dict.items(), reverse=True)[N_DISCARD:N_DISCARD + n]).values())
  return rv

def index_text(uri, text):
  rows = []
  n_chunk = 0
  for s in re.split(r"\.\s+", text): # Sentence based splitting: makes sense to me.
    s = s.strip()
    if (len(s) >= min_sentence_len):
      logging.debug("URI: {}, CHUNK_NUM: {}\nCHUNK: '{}'".format(uri, n_chunk, s))
      embed = gen_embeddings(s)
      top_n = gen_top_n(embed, TOP_N)
      row_map = {
         "uri": uri
         , "chunk_num": n_chunk
         , "chunk": s
         , "embedding": embed
         , "top_n": top_n
      }
      rows.append(row_map)
      n_chunk += 1
  with engine.begin() as conn:
    conn.execute(insert(text_embed_table), rows)

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

rerank_enum = set(["NONE", "REGEX"])

def gen_sql(rerank):
  rerank = rerank.upper()
  if rerank not in rerank_enum:
    logging.warn("rerank value '{}' not allowed".format(rerank))
    rerank = "NONE"
  rv = """
WITH q_embed AS
(
  SELECT uri, OVERLAP(:top_n, top_n) score, chunk, embedding
  FROM text_embed
  WHERE top_n @> :top_n
  ORDER BY score DESC
  LIMIT :limit * 10 /* FIXME parameterize this multiplier */
)
SELECT uri, 1 - (embedding <=> (:q_embed)::VECTOR) sim, chunk
FROM q_embed
ORDER BY sim DESC
LIMIT :limit
"""
  return rv

# Arg: search terms
# Returns: list of {"uri": uri, "sim": sim, "token": token, "chunk": chunk}
def search(terms, limit=5, rerank="none"):
  logging.info("rerank: {}".format(rerank))
  q = ' '.join(terms)
  rv = []
  embed = gen_embeddings(q)
  top_n = gen_top_n(embed, TOP_N_Q)
  logging.info("Query string: '{}'".format(q))
  logging.info("Query top_n: '{}'".format(top_n))
  t0 = time.time()
  stmt = None
  if "REGEX" == rerank.upper():
    terms_regex = '({})'.format('|'.join(list(set(terms)))) # Remove duplicate terms via the set
    logging.info("terms_regex: {}".format(terms_regex))
    stmt = text(gen_sql(rerank) + "\nWHERE chunk ~* :regex").bindparams(q_embed=embed, limit=limit, regex=terms_regex)
  else:
    stmt = text(gen_sql(rerank)).bindparams(q_embed=embed, top_n=top_n, limit=limit)
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
# rerank is one of none, regex, cosine
#
@app.route("/search/<q_base_64>/<int:limit>")
@app.route("/search/<q_base_64>/<int:limit>/<rerank>")
def do_search(q_base_64, limit, rerank="none"):
  q = decode(q_base_64)
  q = clean_text(q)
  rv = retry(search, (q.split(), limit, rerank))
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
  setup_db()
  text_embed_table = Table("text_embed", MetaData(), autoload_with=engine)
  port = int(os.getenv("FLASK_PORT", 18080))
  from waitress import serve
  serve(app, host="0.0.0.0", port=port, threads=n_threads)
# Indexing mode
else:
  print("Usage: {} [-s for server mode] [-q for query mode]\n".format(sys.argv[0]))
  sys.exit(1)

