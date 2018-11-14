#!/usr/bin/env bash
# [wf] execute test stage

num_run=$(ls runs/ | wc -l |  tr -d ' ')
num_run=$((num_run-1))
run_id="run$num_run"

python /exp/code/test.py \
    --output_directory "runs/$run_id"