#!/bin/bash

# 2 parallel procs
#./index_doc.py $( ./gen_m_of_n.sh 0 2 ./data/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 1 2 ./data/*.txt ) &

# 4 parallel procs
./index_doc.py $( ./gen_m_of_n.sh 0 4 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 1 4 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 2 4 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 3 4 ./data/*.txt ) &

wait

