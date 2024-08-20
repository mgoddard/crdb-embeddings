#!/bin/bash

# 2 parallel procs
#./index_doc.py $( ./gen_m_of_n.sh 0 2 ./data/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 1 2 ./data/*.txt ) &

# 4 parallel procs
#./index_doc.py $( ./gen_m_of_n.sh 0 4 ./data/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 1 4 ./data/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 2 4 ./data/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 3 4 ./data/*.txt ) &

# 8 parallel procs
./index_doc.py $( ./gen_m_of_n.sh 0 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 1 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 2 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 3 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 4 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 5 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 6 8 ./data/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 7 8 ./data/*.txt ) &

wait

