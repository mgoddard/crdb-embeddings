#!/usr/bin/env python3

import time
import sys, os
import gzip
import html
import re
import fileinput
import logging
import mwparserfromhell # Wikipedia dump parser
import requests

log_level = os.environ.get("LOG_LEVEL", "WARN").upper()
logging.basicConfig(
  level=log_level
  , format="[%(asctime)s] %(message)s"
  , datefmt="%m/%d/%Y %I:%M:%S %p"
)
print("Log level: {} (export LOG_LEVEL=[DEBUG|INFO|WARN|ERROR] to change this)".format(log_level))

host = os.environ.get("FLASK_HOST", "localhost")
port = os.environ.get("FLASK_PORT", "18080")
url = "http://{}:{}/index".format(host, port)
logging.info("app URL: {}".format(url))

#
# curl -s -k https://storage.googleapis.com/crl-goddard-text/wikipedia_001.csv.gz | gunzip - | ./load_wiki_stdin.py
#

MAX_LEN = 2048 # Maximum length of the resulting string

max_lines = int(os.getenv("MAX_LINES", "10"))
logging.info("MAX_LINES: {}".format(max_lines))

# TODO: remove this unless retries are required from the requests client
max_retries = int(os.getenv("MAX_RETRIES", "3"))
logging.info("MAX_RETRIES: {}".format(max_retries))

def index_text(doc_uri, doc_text):
  t0 = time.time()
  req = requests.post(url, json = { "uri": doc_uri, "text": doc_text })
  et = time.time() - t0
  print("{}: {} (t = {:.3f} ms)".format(doc_uri, "SUCCESS" if req.status_code == 200 else "FAILED", et*1000))

# Skip the #REDIRECT lines
redir = "#REDIRECT"
end = " ...."
n_line = 0
for line in fileinput.input():
  line = line.strip()
  (title, text) = line.split('<')[1:]
  if text[0:len(redir)] == redir:
    continue
  parsed_wikicode = mwparserfromhell.parse(text)
  clean = parsed_wikicode.strip_code().strip()
  clean = re.sub(r"(<ref[^/]+/>|<ref[^>]*></ref>)", '', clean)
  #print("title: {}, text: {}".format(title, clean[0:MAX_LEN-len(end)] + end))
  index_text(title, clean[0:MAX_LEN-len(end)] + end)
  n_line += 1
  if n_line == max_lines:
    break

