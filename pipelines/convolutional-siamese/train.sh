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
    --dataset_directory "/exp/data/trigger/prepared/" \
    --batch_size 40 \
    --num_workers 10 \
    --num_epochs 1 \
    --transforms "bandpass:whiten" \
    --augmentations "noise:amplitude" \
    --length 50000 \
    --sample_strategy "sequential" \
    --learning_rate 1e-3 \
    --weight_decay 1e-3 \
    --split $split \
    --embedding_size 10 \
    --loss_function "dpcl"