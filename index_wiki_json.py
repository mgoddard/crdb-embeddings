#!/usr/bin/env python3

"""

Data set: https://www.kaggle.com/datasets/ltcmdrdata/plain-text-wikipedia-202011

Download that Zip file, unzip it, and run this script in the resulting directory;
e.g.

  ./index_wiki_json.py a*.json

Work through the files until you've loaded a sufficiently large data set.

"""

FRACTION = 0.01 # The probability that a given JSON "row" will be indexed
BASE_URL = "https://en.wikipedia.org/wiki" # Wikipedia base URL to prepend to the generated one

import sys, os, json, random, re
import urllib.parse
import requests
import time

host = os.environ.get("FLASK_HOST", "localhost")
port = os.environ.get("FLASK_PORT", "18080")
url = "http://{}:{}/index".format(host, port)
pat = re.compile(r'%[a-zA-Z0-9]{2}')

for json_file in sys.argv[1:]:
  with open(json_file) as f:
    json_array = json.load(f)
    for obj in json_array:
      if random.random() < FRACTION:
        t0 = time.time()
        title = obj["title"]
        uri = urllib.parse.quote_plus(title).replace(r'+', '_')
        mat = pat.search(uri)
        if mat:
          continue
        uri = BASE_URL + '/' + uri
        print("URI: {}".format(uri))
        txt = re.sub(r'(==+)[^=]+\1', '', obj["text"])
        http_code = 500
        n_retry = 0
        while http_code == 500 and n_retry < 2:
          req = requests.post(url, json = { "uri": uri, "text": txt })
          http_code = req.status_code
          n_retry += 1
        et = time.time() - t0
        print("URI: {} {} (t = {:.3f} ms)".format(uri, "SUCCESS" if req.status_code == 200 else "FAILED: " + req.content.decode("utf-8"), et*1000))

