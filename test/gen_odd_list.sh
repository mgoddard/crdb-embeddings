#!/bin/bash

for i in ./data/*.txt
do
  s=$( cksum $i | awk '{print $1}' )
  if [ $((s % 2)) -eq 1 ]
  then
    echo "$i"
  fi
done

