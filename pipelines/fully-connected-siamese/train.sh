#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
num_run=$(ls runs/ | wc -l |  tr -d ' ')
run_id="run$num_run"

mkdir -p runs/$run_id
cp train.sh runs/${run_id}/${run_id}_train.sh

python /exp/code/train.py \
    --output_directory "runs/$run_id" \
    --dataset_directory "/exp/data/prepared/" \
    --batch_size 10 \
    --num_workers 10 \
    --num_epochs 100 \
    --transforms "demean:bandpass" \
    --length 100000 \
    --sample_strategy "random" \
    --learning_rate 1e-3 \
    --split "SAC_20100227_Chile_prem"