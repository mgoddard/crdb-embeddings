#!/bin/bash

time psql $DB_URL -F $'\t' -tAc "select uri, chunk_num, regexp_replace(chunk, E'[\n\r\t]+', ' ', 'g' ), embedding from text_embed where uri in (with u as (select uri from text_embed where random() < 0.0004) select distinct uri from u) order by 1, 2;" | gzip - > /tmp/text_embed.tsv.gz

