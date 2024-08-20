# For server
#export FLASK_PORT=1972
#export FLASK_HOST=localhost
export FLASK_PORT=80
export FLASK_HOST=34.150.212.10
export DB_URL="postgres://test_role:123abc@127.0.0.1:26257/defaultdb?sslmode=require&sslrootcert=$HOME/certs/ca.crt"

export LOG_LEVEL=INFO
export N_THREADS=10
export CACHE_SIZE=1024
export MIN_SENTENCE_LEN=8
export N_CLUSTERS=100
export TRAIN_FRACTION=0.75
export MODEL_FILE=/tmp/model.pkl
export BATCH_SIZE=512
export KMEANS_VERBOSE=1
export KMEANS_MAX_ITER=25

# For client
export MAX_RESULTS=5

