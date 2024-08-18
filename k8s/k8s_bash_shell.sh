#!/bin/bash

if [ "$#" -ne 1 ]; then
  me=$( basename $0 )
  echo "Usage: $me container_name"
  exit 1
fi

kubectl exec --stdin --tty $1 -- /bin/bash

