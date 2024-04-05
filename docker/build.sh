#!/bin/bash

echo "Building container"
docker build . \
    -f Dockerfile \
    -t mmseg:latest \
    --progress plain 

