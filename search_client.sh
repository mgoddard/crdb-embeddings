#!/bin/bash

. ./env.sh

max_results=3
use_regex=true

if [ $# -lt 1 ]
then
  echo "Usage: $0 word [word2 ... wordN]"
  exit 1
fi

curl -s http://$FLASK_HOST:$FLASK_PORT/search/$( echo -n "$@" | base64 )/$max_results/$use_regex | jq

