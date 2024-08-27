#!/bin/bash

. ./env.sh

if [ $# -ne 1 ]
then
  echo "Usage: $0 path_to_file"
  exit 1
fi

data_dir=$1
echo "Data directory: $data_dir"

# 2 parallel procs
#./index_doc.py $( ./gen_m_of_n.sh 0 2 ./$data_dir/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 1 2 ./$data_dir/*.txt ) &

# 4 parallel procs
#./index_doc.py $( ./gen_m_of_n.sh 0 4 ./$data_dir/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 1 4 ./$data_dir/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 2 4 ./$data_dir/*.txt ) &
#./index_doc.py $( ./gen_m_of_n.sh 3 4 ./$data_dir/*.txt ) &

# 9 parallel procs
./index_doc.py $( ./gen_m_of_n.sh 0 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 1 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 2 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 3 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 4 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 5 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 6 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 7 9 ./$data_dir/*.txt ) &
./index_doc.py $( ./gen_m_of_n.sh 8 9 ./$data_dir/*.txt ) &

wait

