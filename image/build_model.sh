#!/bin/bash

. ./image_env.sh

# Arg: secret
if [ $# -ne 1 ]
then
  echo "Usage: $0 secret"
  exit 1
fi

time curl -s http://$FLASK_HOST:$FLASK_PORT/build_model/$1

