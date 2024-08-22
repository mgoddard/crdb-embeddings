#!/bin/bash

# Startup command for Docker image: install Python dependencies and then start the app
pip3 install --no-cache-dir -r ./requirements.txt && rm -rf ~/.cache/pip

# Start the app
exec python ./pgvector_embeddings.py

