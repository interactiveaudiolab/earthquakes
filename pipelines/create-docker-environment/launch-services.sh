#!/usr/bin/env bash
# [wf] launch tensorboard, jupyter
TENSORBOARD_HOST_PORT=6006
JUPYTER_HOST_PORT=8888

./stop-services.sh

docker run --rm --workdir=/pipeline -v `pwd`/..:/pipeline \
    -p 127.0.0.1:$TENSORBOARD_HOST_PORT:6006 -d \
    --entrypoint tensorboard \
    --name tensorboard \
  local/earthquake --logdir .

docker run --rm --workdir=/pipeline -v `pwd`/../..:/pipeline \
    -p 127.0.0.1:$JUPYTER_HOST_PORT:8888 -d \
    --entrypoint jupyter \
    --name jupyter \
    --ipc=host \
  local/earthquake lab --ip=0.0.0.0