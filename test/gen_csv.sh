#!/bin/bash

# This is how to create the .csv files to compare results with single vs. multiple threads
psql $DB_URL -tAc "select uri, chunk_num, token from text_embed order by 1, 2" > threads_2_b.csv

