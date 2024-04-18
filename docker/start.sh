#!/bin/bash

docker run --rm -it -d --gpus '"device=1"' --shm-size=8g -v /home/matykina_ov/mmsegmentation:/mmsegmentation -v /datasets/nkb_robosegment_small:/data --name mmseg_matykina mmseg:latest "/bin/bash"
