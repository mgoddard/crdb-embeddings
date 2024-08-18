#!/bin/bash

# Startup command for Docker image: install Python dependencies and then start the app
pip3 install --no-cache-dir -r ./requirements.txt && rm -rf ~/.cache/pip

# Grab model via curl and copy to $MODEL_FILE
curl -o $MODEL_FILE $MODEL_FILE_URL

# Start the app
exec python ./pgvector_embeddings.py -s

