#!/bin/bash

# TODO: Set the host and port of where the app is running
host=localhost
port=1963
max_results=4
use_regex=True

if [ $# -lt 1 ]
then
  echo "Usage: $0 word [word2 ... wordN]"
  exit 1
fi

time curl http://$host:$port/search/$( echo -n "$@" | base64 )/$max_results/$use_regex

