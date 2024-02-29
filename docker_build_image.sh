#!/bin/bash

. ./docker_include.sh

# Ref: https://everythingdevops.dev/building-x86-images-on-an-apple-m1-chip/
docker buildx build --platform=linux/amd64 -t $docker_id/$img_name .
#docker build -t $docker_id/$img_name .

