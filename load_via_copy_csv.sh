#!/bin/bash

for i in {1..11}
  do url=$( printf "https://storage.googleapis.com/crl-goddard-text/text_embed_fastembed.%03d.csv.gz" $i )
  curl $url | gunzip - | psql "postgresql://user:passwd@hostname:26257/defaultdb?sslmode=require&sslrootcert=./ca.crt" -c "COPY text_embed FROM STDIN (DELIMITER E'\t');"
done

