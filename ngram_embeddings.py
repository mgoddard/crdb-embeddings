#!/usr/bin/env python

import torch
import re, sys, os, time
from transformers import BertTokenizer, BertModel
import base36
import logging
import psycopg2
import psycopg2.pool
import numpy as np

# For Flask app
from flask import Flask, request, Response, g
import urllib
import json
import base64

CHARSET = "utf-8"

# Max number of dimensions to store in DB and use for queries (out of 768)
#TOP_N = 32
#TOP_N = 10
#TOP_N = 8
#TOP_N = 16
TOP_N = 64

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
DELIM = ' '
#DELIM = '~'

# Tweak minimum similarity in the DB session.  Lower may be better given the LIMIT clause.
#   set pg_trgm.similarity_threshold = 0.25;
#   set pg_trgm.similarity_threshold = 0.1;
# The value passed in via the environment will be set in SQL session
min_sim = float(os.environ.get("MIN_SIM", "0.20"))
print("pg_trgm.similarity_threshold: {} (set via 'export MIN_SIM=0.1')".format(min_sim))

n_threads = int(os.environ.get("N_THREADS", "1"))
print("n_threads: {} (set via 'export N_THREADS=10')".format(n_threads))

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

t0 = time.time()
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
et = time.time() - t0
logging.info("BertTokenizer: {} s".format(et))

t0 = time.time()
# Set this up once and reuse
model = BertModel.from_pretrained("bert-base-uncased", output_hidden_states = True)
model.eval()
et = time.time() - t0
logging.info("BertModel + eval: {} s".format(et))

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
    outputs = model(tokens_tensor, segments_tensors)
    hidden_states = outputs[2]
  token_vecs = hidden_states[-2][0]
  sentence_embedding = torch.mean(token_vecs, dim=0)
  rv = sentence_embedding.tolist()
  logging.info("gen_embeddings rv:\n", rv)
  return rv

# TODO:
# Refactor, preserving the dims dictionary to store in the DB table as JSONB

def gen_svec(embed_list):
  dims = {}
  for i in range(0, len(embed_list)):
    dims[base36.dumps(i).zfill(2)] = embed_list[i]
  trunc = dict(sorted(dims.items(), key=lambda item: abs(item[1]), reverse=True)[N_DISCARD:TOP_N + N_DISCARD])
  vals = list(trunc.values())
  norm = np.linalg.norm(vals)
  trunc = { k: v/norm for k, v in trunc.items() } # Normalize the remaining values in this dict
  return trunc

# From the list of embeddings, the 768 element array, return a string consisting of
# the base 36 encoded array dimension (2 chars) for the TOP_N elements having the
# largest magnitude.  These are separated by DELIM and have DELIM appended as well.
def gen_embed_token(svec):
  rv = DELIM.join(list(svec.keys()))
  rv += DELIM
  return rv

# From the given string s, return [token, svec]
def get_token_svec(s):
  rv = None
  t0 = time.time()
  embed = gen_embeddings(s)
  et = time.time() - t0
  logging.info("gen_embeddings: {} s".format(et))
  svec = gen_svec(embed)
  tok = gen_embed_token(svec)
  return [tok, svec]

"""
DROP TABLE IF EXISTS text_embed;
CREATE TABLE text_embed
(
  uri STRING NOT NULL
  , chunk_num INT NOT NULL
  , token STRING NOT NULL
  , chunk STRING NOT NULL
  , PRIMARY KEY (uri, chunk_num)
);
CREATE INDEX ON text_embed USING GIN (token gin_trgm_ops);
"""

ins_sql = "INSERT INTO text_embed (uri, chunk_num, token, chunk) VALUES (%s, %s, %s, %s)"
def index_text(uri, text):
  conn = get_conn()
  with conn.cursor() as cur:
    n_chunk = 0
    for s in re.split(r"\.\s+", text): # Sentence based splitting: makes sense to me.
    #for s in re.split(r"[\r\n]{2,}", text): # Paragraph based splitting: topics could vary too much?
      s = s.strip()
      if (len(s) > 0):
        token = get_token_svec(s)[0]
        logging.debug("URI: {}, CHUNK_NUM: {}\nCHUNK: '{}'\n".format(uri, n_chunk, s))
        cur.execute(ins_sql, (uri, n_chunk, token, s))
        n_chunk += 1
    conn.commit()
  put_conn(conn)

def index_file(in_file):
  text = ""
  with open(in_file, mode="rt") as f:
    for line in f:
      text += line
  in_file = re.sub(r"\./", '', in_file) # Trim leading '/'
  index_text(in_file, text)

# Extend pool class so we can SET some values in session once connected
class CrdbConnectionPool(psycopg2.pool.ThreadedConnectionPool):
  def _connect(self, key=None):
    """Create a new connection and assign it to 'key' if not None."""
    conn = psycopg2.connect(*self._args, **self._kwargs)
    with conn.cursor() as cur:
      cur.execute("SET pg_trgm.similarity_threshold = %s;", (min_sim,)) # Verified: this works
    if key is not None:
      self._used[key] = conn
      self._rused[id(conn)] = key
    else:
      self._pool.append(conn)
    return conn

pool = None
def get_conn():
  global pool
  if pool is None:
    pool = CrdbConnectionPool(2, 20, db_url)
  return pool.getconn()

def put_conn(conn):
  global pool
  pool.putconn(conn)

# Clean any special chars out of text
def clean_text(text):
  return re.sub(r"['\",{}]", "", text)

# Decode a base64 encoded value to a UTF-8 string
def decode(b64):
  b = base64.b64decode(b64)
  return b.decode(CHARSET).strip()

app = Flask(__name__)

q_sql = """
WITH q_embed AS
(
  SELECT uri, SIMILARITY(%s, token)::NUMERIC(4, 3) sim, token, chunk
  FROM text_embed@text_embed_token_idx
  WHERE %s %% token
  ORDER BY sim DESC
  LIMIT %s
)
SELECT * from q_embed
"""

# Arg: search terms
# Returns: list of {"uri": uri, "sim": sim, "token": token, "chunk": chunk}
def search(terms, limit=5, use_regex=True):
  logging.info("use_regex: {}".format(use_regex))
  q = ' '.join(terms)
  rv = []
  tok = get_token_svec(q)[0]
  logging.info("Query string: '{}'\nToken: '{}'\n".format(q, tok))
  t0 = time.time()
  conn = get_conn()
  with conn.cursor() as cur:
    if use_regex:
      # TODO: stem the terms before forming the regex
      terms_regex = '({})'.format('|'.join(list(set(terms)))) # Remove duplicate terms via the set
      logging.info("terms_regex: {}\n".format(terms_regex))
      cur.execute(q_sql + "\nWHERE chunk ~* %s", (tok, tok, limit, terms_regex,))
    else:
      cur.execute(q_sql, (tok, tok, limit,))
    rs = cur.fetchall()
    if rs is not None:
      for row in rs:
        (uri, sim, token, chunk) = row
        rv.append({"uri": uri, "sim": float(sim), "token": token, "chunk": chunk})
  et = time.time() - t0
  logging.info("SQL query time: {} ms\n".format(et * 1000))
  put_conn(conn)
  return rv

#
# The search/query
# EXAMPLE (with a limit of 10 results):
#   curl http://localhost:18080/search/$( echo -n "Using Lateral Joins" | base64 )
#
# TODO: parameterize limit as URL param
@app.route("/search/<q_base_64>/<int:limit>")
@app.route("/search/<q_base_64>/<int:limit>/<path:use_regex>")
def do_search(q_base_64, limit, use_regex=True):
  q = decode(q_base_64)
  q = clean_text(q)
  rv = search(q.split(), limit, use_regex.upper() == "TRUE")
  return Response(json.dumps(rv), status=200, mimetype="application/json")

@app.route('/index', methods=['POST'])
def do_index():
  data = request.get_json(force=True)
  index_text(data["uri"], data["text"])
  # Note the extra arguments here which translate the \uxxxx escape codes
  #print("Data: " + json.dumps(data, ensure_ascii=False).encode("utf8").decode())
  return Response("OK", status=200, mimetype="text/plain")

# Query mode
if "-q" == sys.argv[1][0:2]:
  terms = sys.argv[2:]
  for row in search(terms):
    print("URI: {}\nSCORE: {}\nTOKEN: {}\nCHUNK: {}\n".format(row["uri"], row["sim"], row["token"], row["chunk"]))
# Server mode
elif "-s" == sys.argv[1][0:2]:
  port = int(os.getenv("FLASK_PORT", 18080))
  from waitress import serve
  serve(app, host="0.0.0.0", port=port, threads=n_threads)
# Indexing mode
else:
  t0 = time.time()
  conn = get_conn()
  for in_file in sys.argv[1:]:
    print("Indexing file " + in_file + " now ...")
    index_file(in_file)
  et = time.time() - t0
  logging.info("Total time: {} s".format(et))
  put_conn(conn)

