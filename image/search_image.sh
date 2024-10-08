#!/bin/bash

. ./image_env.sh

max_results="${MAX_RESULTS:-4}"

os=$( uname -o )
base64="base64"
if [[ $os == *"Linux"* ]]
then
  base64="$base64 -w 0"
fi

if [ $# -ne 1 ]
then
  echo "Usage: $0 image_URL"
  exit 1
fi

# @app.route("/search/<int:nItems>/<urlBase64>")
curl -s http://$FLASK_HOST:$FLASK_PORT/search/$max_results/$( echo -n "$1" | $base64 ) | jq
