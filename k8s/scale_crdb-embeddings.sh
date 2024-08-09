#!/bin/bash

if [ "$#" -ne 1 ]
then
  echo "Usage: $0 new_replica_count"
  exit 1
fi

n_rep=$1

kubectl scale --replicas=$n_rep deployment crdb-embeddings

