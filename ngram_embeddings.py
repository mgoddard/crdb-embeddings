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

# Max number of dimensions to store in DB and use for queries (out of 768)
#TOP_N = 8
#TOP_N = 16
TOP_N = 32
#TOP_N = 64
#TOP_N = 128

# Discard the first N tokens as they have little differentiating value
"""
defaultdb=> select substring(token from 1 for 6), count(*) from text_embed group by 1 order by 2 desc;
 substring | count
-----------+-------
 8k al     |    30
 8k ez     |    22
 8k 69     |     2
 8k in     |     1
 8k cd     |     1
 8k 2b     |     1
 8k 50     |     1
(7 rows)
"""
N_DISCARD = 2

# Delimiter for the base36 encoded array dimension values
DELIM = '' # Set to empty string if including the sign of the value as leading +/-

# Tweak minimum similarity in the DB session.  Lower may be better given the LIMIT clause.
#   set pg_trgm.similarity_threshold = 0.25;
#   set pg_trgm.similarity_threshold = 0.1;
# The value passed in via the environment will be set in SQL session
min_sim = float(os.environ.get("MIN_SIM", "0.20"))
print("pg_trgm.similarity_threshold: {} (set via 'export MIN_SIM=0.1')".format(min_sim))

cache_size = int(os.environ.get("CACHE_SIZE", "1024"))
print("cache_size: {} (set via 'export CACHE_SIZE=1024')".format(cache_size))

n_threads = int(os.environ.get("N_THREADS", "1"))
print("n_threads: {} (set via 'export N_THREADS=10')".format(n_threads))

max_retries = int(os.environ.get("MAX_RETRIES", "3"))
print("max_retries: {} (set via 'export MAX_RETRIES=3')".format(max_retries))

token_array_len = int(os.environ.get("TOKEN_ARRAY_LEN", "8"))
print("token_array_len: {} (set via 'export TOKEN_ARRAY_LEN=8')".format(token_array_len))

log_level = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
  level=log_level
  , format="[%(asctime)s] %(message)s"
  , datefmt="%m/%d/%Y %I:%M:%S %p"
)
print("Log level: {} (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)".format(log_level))

logging.info("TOP_N: {}".format(TOP_N))

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
  cur.execute("SET pg_trgm.similarity_threshold = %s;", (min_sim,))
  cur.execute("SET SESSION CHARACTERISTICS AS TRANSACTION ISOLATION LEVEL READ COMMITTED;")
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

# Return a list of tokens, trimmed to token_array_len
def to_token_array(token, n_tokens=None):
  if n_tokens is None:
    n_tokens=int(len(token)/3)
  rv = re.split(r"([-+][a-z0-9]{2})", token)[1::2]
  rv = rv[0:n_tokens]
  return rv

# The fist call to this takes ~ 500 ms but subsequent calls take ~ 40 ms
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
      outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
  token_vecs = hidden_states[-2][0]
  sentence_embedding = torch.mean(token_vecs, dim=0)
  rv = sentence_embedding.tolist()
  logging.debug("vector dimensionality: {}".format(len(rv)))
  return rv

def gen_svec(embed_list):
  dims = {}
  for i in range(0, len(embed_list)):
    # Preserve the sign of the vector values
    v = embed_list[i]
    k = base36.dumps(i).zfill(2)
    if v < 0:
      k = '-' + k
    else:
      k = '+' + k
    dims[k] = v
  trunc = dict(sorted(dims.items(), key=lambda item: abs(item[1]), reverse=True)[N_DISCARD:TOP_N + N_DISCARD])
  vals = list(trunc.values())
  norm = np.linalg.norm(vals)
  trunc = { k: v/norm for k, v in trunc.items() } # Normalizing the remaining values in this dict
  print("trunc: ", json.dumps(trunc))
  return trunc

# From the list of embeddings, the 768 element array, return a string consisting of
# the base 36 encoded array dimension (2 chars) for the TOP_N elements having the
# largest magnitude.  These are separated by DELIM and have DELIM appended as well.
def gen_embed_token(svec):
  rv = DELIM.join(list(svec.keys()))
  rv += DELIM
  return rv

# From the given string s, return [token, svec]
@lru_cache(maxsize=cache_size)
def get_token_svec(s):
  rv = None
  t0 = time.time()
  embed = gen_embeddings(s)
  et = time.time() - t0
  logging.info("gen_embeddings: {} s".format(et))
  svec = gen_svec(embed)
  tok = gen_embed_token(svec)
  return [tok, svec]

ddl_func = """
CREATE OR REPLACE FUNCTION score_row (q JSONB, r JSONB)
RETURNS FLOAT
LANGUAGE SQL
AS $$
  SELECT COALESCE(SUM(qv * rv), 0.0) score
  FROM (
    SELECT
      (json_each_text(q::JSONB)).@1 qk
      , ((json_each_text(q::JSONB)).@2)::float qv
      , (json_each_text(r::JSONB)).@1 rk
      , ((json_each_text(r::JSONB)).@2)::float rv
  )
  WHERE qk = rk;
$$;
"""

arr_func = """
CREATE OR REPLACE FUNCTION to_token_array(token STRING, n INT)
RETURNS STRING[]
IMMUTABLE LEAKPROOF
LANGUAGE SQL
AS $$

SELECT ARRAY_REMOVE(
  REGEXP_SPLIT_TO_ARRAY(
    REGEXP_REPLACE(
      SUBSTRING(token FROM 1 FOR n*3),
      '([-+])([a-z0-9]{2})', '\\1\\2 ', 'g'
    ),
    '\\s+'
  ), '');
$$;
"""

overlap_func = """
CREATE OR REPLACE FUNCTION overlap(a STRING[], b STRING[])
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

ddl_table = """
CREATE TABLE text_embed
(
  uri STRING NOT NULL
  , chunk_num INT NOT NULL
  , token STRING NOT NULL
  , svec JSONB NOT NULL
  , chunk STRING NOT NULL
  , token_array STRING[] NOT NULL
  , PRIMARY KEY (uri, chunk_num)
  , INVERTED INDEX text_embed_token_idx (token_array)
);
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
      logging.info("Creating score_row UDF")
      conn.execute(text(ddl_func))
      logging.info("Creating to_token_array UDF")
      conn.execute(text(arr_func))
      logging.info("Creating overlap UDF")
      conn.execute(text(overlap_func))
      conn.commit()
  else:
    logging.info("text_embed table already exists")

def index_text(uri, text):
  rows = []
  n_chunk = 0
  for s in re.split(r"\.\s+", text): # Sentence based splitting: makes sense to me.
    s = s.strip()
    if (len(s) > 0):
      (token, svec) = get_token_svec(s)
      logging.debug("URI: {}, CHUNK_NUM: {}\nCHUNK: '{}'".format(uri, n_chunk, s))
      row_map = {
         "uri": uri
         , "chunk_num": n_chunk
         , "token": token
         , "svec": svec
         , "chunk": s
         , "token_array": to_token_array(token)
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

rerank_enum = set(["NONE", "REGEX", "COSINE"])
def gen_sql(rerank):
  rerank = rerank.upper()
  if rerank not in rerank_enum:
    logging.warn("rerank value '{}' not allowed".format(rerank))
    rerank = "NONE"
  rv = """
WITH q_embed AS
(
  SELECT uri, OVERLAP(TO_TOKEN_ARRAY(:q_tok, :n_tok), token_array)/10.0 sim, token, chunk, svec
  FROM text_embed
  WHERE token_array && TO_TOKEN_ARRAY(:q_tok, :n_tok)
  ORDER BY sim DESC
  LIMIT :limit
)
"""
  if "REGEX" == rerank:
    pass # Already handled in search()
  elif "COSINE" == rerank:
    rv += """
, q_cos AS
(
  SELECT  uri, score_row(:q_svec, svec) sim, token, chunk
  FROM q_embed
)
SELECT *
FROM q_cos
ORDER BY q_cos.sim DESC
    """
  if not "COSINE" == rerank:
    rv += """
  SELECT uri, sim, token, chunk from q_embed
    """
  return rv

# Arg: search terms
# Returns: list of {"uri": uri, "sim": sim, "token": token, "chunk": chunk}
def search(terms, limit=5, rerank="none"):
  logging.info("rerank: {}".format(rerank))
  q = ' '.join(terms)
  rv = []
  (tok, svec) = get_token_svec(q)
  logging.info("Query string: '{}'\nToken: '{}'".format(q, tok))
  logging.info("Query svec: {}".format(svec))
  t0 = time.time()
  stmt = None
  if "REGEX" == rerank.upper():
    terms_regex = '({})'.format('|'.join(list(set(terms)))) # Remove duplicate terms via the set
    logging.info("terms_regex: {}".format(terms_regex))
    stmt = text(gen_sql(rerank) + "\nWHERE chunk ~* :regex").bindparams(q_tok=tok, n_tok=token_array_len, limit=limit, regex=terms_regex)
  elif "COSINE" == rerank.upper():
    stmt = text(gen_sql(rerank)).bindparams(bindparam('q_svec', type_=JSONB), q_tok=tok, n_tok=token_array_len, limit=limit, q_svec=svec)
  else:
    stmt = text(gen_sql(rerank)).bindparams(q_tok=tok, n_tok=token_array_len, limit=limit)
  with engine.connect() as conn:
    rs = conn.execute(stmt)
    if rs is not None:
      for row in rs:
        (uri, sim, token, chunk) = row
        rv.append({"uri": uri, "sim": float(sim), "token": token, "chunk": chunk})
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
  logging.info(get_token_svec.cache_info())
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

