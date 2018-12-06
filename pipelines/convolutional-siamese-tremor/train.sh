#!/usr/bin/env bash
# [wf] execute train stage

mkdir -p runs
num_run=$(ls runs/ | wc -l |  tr -d ' ')
run_id="run$num_run"

mkdir -p runs/$run_id
cp train.sh runs/${run_id}/${run_id}_train.sh

python /exp/code/train.py \
    --model_type "conv" \
    --output_directory "runs/$run_id" \
    --dataset_directory "/exp/data/prepared/tremor/" \
    --batch_size 64 \
    --num_workers 10 \
    --num_epochs 100 \
    --transforms "bandpass:whiten" \
    --augmentations "amplitude:noise" \
    --length 16384 \
    --sample_strategy "sequential" \
    --learning_rate 2e-5 \
    --weight_decay 1e-1 \
    --split $split \
    --embedding_size 10 \
    --loss_function "dpcl"

python /exp/code/test.py \
    --output_directory "runs/$run_id"