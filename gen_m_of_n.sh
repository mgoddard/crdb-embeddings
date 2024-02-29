#!/bin/bash

if [ "$#" -lt 2 ]
then
  echo "Usage: $0 my_id total_procs"
  exit 1
fi

me=$1
n=$2

for i in "${@:3}"
do
  s=$( cksum $i | awk '{print $1}' )
  if [ $((s % n)) -eq $me ]
  then
    echo "$i"
  fi
done

