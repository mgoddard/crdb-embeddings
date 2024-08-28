# For server
#export FLASK_PORT=1972
export FLASK_PORT=1999
export FLASK_HOST=localhost
#export FLASK_PORT=80
#export FLASK_HOST=vector.la-cucaracha.net
export DB_URL="postgres://test_role:123abc@127.0.0.1:26257/defaultdb?sslmode=require&sslrootcert=$HOME/certs/ca.crt"

export LOG_LEVEL=INFO
export N_THREADS=10
export MIN_SENTENCE_LEN=8
export N_CLUSTERS=1000
export TRAIN_FRACTION=0.5
export MODEL_FILE=/tmp/model.pkl
export MODEL_FILE_URL="https://storage.googleapis.com/crl-goddard-text/model_Fastembed_1k.pkl"
export BATCH_SIZE=768
export KMEANS_VERBOSE=1
export KMEANS_MAX_ITER=250
export SKIP_KMEANS=False
export SECRET="TextWithNoSpecialChars"
export BLOB_STORE_KEEP_N_ROWS=3
export TOKENIZERS_PARALLELISM=false

# For client
export MAX_RESULTS=5

