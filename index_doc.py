#!/usr/bin/env python3

import requests
import sys, os, re
import time

host = os.environ.get("FLASK_HOST", "localhost")
port = os.environ.get("FLASK_PORT", "18080")
url = "http://{}:{}/index".format(host, port)

if len(sys.argv) < 2:
  print("Usage: {} file_to_index [file2 ...]\n".format(sys.argv[0]))
  sys.exit(1)

def read_file(in_file):
  rv = ""
  with open(in_file, mode="rb") as f:
    try:
      rv = f.read().decode(errors="replace")
    except UnicodeDecodeError as e:
      print(e)
      rv = None
  return rv

for doc_uri in sys.argv[1:]:
  t0 = time.time()
  print("Input file: {}".format(doc_uri))
  doc_text = read_file(doc_uri)
  if doc_text is None:
    print("FAILED to extract text: {}".format(doc_uri))
    continue
  doc_uri = re.sub(r"^[\./]+", '', doc_uri)
  req = requests.post(url, json = { "uri": doc_uri, "text": doc_text })
  et = time.time() - t0
  print("{} (t = {:.3f} ms)".format("SUCCESS" if req.status_code == 200 else "FAILED: " + req.content.decode("utf-8"), et*1000))

