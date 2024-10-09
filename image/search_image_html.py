#!/usr/bin/env python3

import os, sys, re
import json
import requests
import base64

host = os.getenv("FLASK_HOST")
port = os.getenv("FLASK_PORT")
if host is None or port is None:
  print("FLASK_HOST and FLASK_PORT must be present in environment. Quitting.")
  sys.exit(1)
max_results = int(os.environ.get("MAX_RESULTS", "5"))

if len(sys.argv) != 2:
  print("Usage: {} search_image_url".format(sys.argv[0]))
  sys.exit(1)
url = sys.argv[1]

def b64(s):
  return base64.b64encode(s.encode("utf-8")).decode("utf-8")

endpoint = "http://{}:{}".format(host, port)
url_b64 = b64(url)
r = requests.get(endpoint + "/search/{}/{}".format(max_results, url_b64))
obj = r.json()

hdr = """
<!DOCTYPE html>
<html>
<head>
	<title>Search Results for {}</title>
</head>
<body>
""".format(url)

html_file = "/tmp/" + re.sub(r'=+$', '', url_b64) + ".html"

with open(html_file, "wt") as f:
  f.write(hdr)
  f.write('<table style="margin-left: auto; margin-right: auto; padding-top: 10%;">\n')
  f.write('<tr style="vertical-align: center;">\n')
  for hit in obj:
    f.write('<td><img src="{}/thumb/{}"/></td>'.format(endpoint, b64(hit["uri"])))
  f.write("</tr>\n")
  f.write('<tr style="vertical-align: center;">\n')
  for hit in obj:
    f.write('<td style="text-align: center;">score: {:.3f}</td>'.format(hit["score"]))
  f.write("</tr>\n")
  f.write("</table>\n")
  f.write("</body>\n</html>\n\n")
print("open {}".format(html_file))

