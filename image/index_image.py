#!/usr/bin/env python3

import requests
import sys, os, re
import time
import base64

host = os.environ.get("FLASK_HOST", "localhost")
port = os.environ.get("FLASK_PORT", "18080")
url = "http://{}:{}/index".format(host, port)

if len(sys.argv) < 2:
  print("Usage: {} uri_to_index [...]\n".format(sys.argv[0]))
  sys.exit(1)

for img_uri in sys.argv[1:]:
  t0 = time.time()
  print("Image URI: {}".format(img_uri))
  # @app.route("/index/<urlBase64>")
  req = requests.get(url + '/' + base64.b64encode(img_uri.encode("utf-8")).decode("utf-8"))
  et = time.time() - t0
  print("{} (t = {:.3f} ms)".format("SUCCESS" if req.status_code == 200 else "FAILED: " + req.content.decode("utf-8"), et*1000))
