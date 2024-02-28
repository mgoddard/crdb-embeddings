#!/bin/bash

. ./docker_include.sh

img="$docker_id/$img_name"

#export DB_URL="postgres://tourist:tourist@host.docker.internal:26257/defaultdb"
export DB_URL="postgres://test_role:123abc@host.docker.internal:26257/defaultdb?sslmode=require&sslrootcert=/Users/mgoddard/certs/ca.crt"
export MIN_SIM=0.2
export N_THREADS=1
export LOG_LEVEL=WARN
export FLASK_PORT=18080
export CACHE_SIZE=1024

docker pull $img:$tag
docker run -e DB_URL -e MIN_SIM -e N_THREADS -e LOG_LEVEL -e FLASK_PORT -e CACHE_SIZE --publish 1999:18080 $img

