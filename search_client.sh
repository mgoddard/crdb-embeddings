#!/bin/bash

. ./env.sh

max_results="${MAX_RESULTS:-4}"
rerank="${RERANK:-regex}"

if [ $# -lt 1 ]
then
  echo "Usage: $0 word [word2 ... wordN]"
  exit 1
fi

echo "rerank: $rerank"

curl -s http://$FLASK_HOST:$FLASK_PORT/search/$( echo -n "$@" | base64 )/$max_results/$rerank | jq

