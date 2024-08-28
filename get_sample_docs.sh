#!/bin/bash

. ./env.sh

if [ $# -lt 1 ]
then
  echo "Usage: $0 n_docs"
  exit 1
fi

n_docs=$1

curl -s http://$FLASK_HOST:$FLASK_PORT/sample/$n_docs | jq

