#!/usr/bin/env python

import requests
import sys, os, re

# FIXME: get URL from environment
url = "http://localhost:1963/index"

if len(sys.argv) != 2:
  print("Usage: {} file_to_index\n".format(sys.argv[0]))
  sys.exit(1)

def read_file(in_file):
  rv = ""
  with open(in_file, mode="rt") as f:
    for line in f:
      rv += line
  return rv

doc_uri = sys.argv[1]
doc_text = read_file(doc_uri)
doc_uri = re.sub(r"^[\./]+", '', doc_uri)
#print("URI: {}\nTEXT: {}".format(doc_uri, doc_text))

req = requests.post(url, json = { "uri": doc_uri, "text": doc_text })
print("Status: {}".format("SUCCESS" if req.status_code == 200 else "FAILED"))

