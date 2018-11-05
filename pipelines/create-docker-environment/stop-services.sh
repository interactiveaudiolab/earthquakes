#!/usr/bin/env bash

docker stop -t 0 tensorboard || true
docker stop -t 0 jupyter  || true