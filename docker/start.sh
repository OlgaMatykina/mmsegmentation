#!/bin/bash

docker run --rm -it -d --gpus all --shm-size=8g -v /home/matykina_ov/mmsegmentation:/mmsegmentation -v /datasets/nkb_robosegment_small:/data --name mmseg mmseg:latest "/bin/bash"
