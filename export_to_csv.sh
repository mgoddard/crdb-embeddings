#!/bin/bash

for n in {0..2}
do
  time psql $DB_URL -F $'\t' -tAc "select uri, chunk_num, regexp_replace(chunk, E'[\n\r\t]+', ' ', 'g' ), embedding from text_embed where mod(abs(fnv64(uri || chunk_num::string)), 3) = ${n};" | gzip - > /tmp/text_embeddings.${n}.csv.gz
done

