#!/bin/bash

n_files=12
f_base_name="text_embed_fastembed"
out_dir="/tmp"

for n in $( seq 0 $(( n_files - 1 )) )
do
  fname=$( printf "${f_base_name}.%03d.csv.gz" $n )
  time psql $DB_URL -F $'\t' -tAc "SELECT uri, chunk_num, REGEXP_REPLACE(chunk, E'[\n\r\t]+', ' ', 'g' ), embedding
  FROM text_embed
  WHERE MOD(ABS(FNV64(uri || chunk_num::string)), $n_files) = ${n};" | gzip - > /$out_dir/$fname
done

