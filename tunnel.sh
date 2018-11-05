#!/usr/bin/env bash
ssh -f -N -L 6006:localhost:6006 cortex
ssh -f -N -L 8889:0.0.0.0:8888 cortex
