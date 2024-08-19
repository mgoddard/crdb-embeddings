#!/bin/bash

if [ $# -ne 3 ]
then
  echo "Usage: $0 pod_name source_file_path dest_file_path"
  exit 1
fi

#kubectl cp crdb-embeddings-764547c5b5-h52w8:/tmp/model.pkl model.pkl
kubectl cp $1:$2 $3

