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
    --batch_size 10 \
    --num_workers 10 \
    --num_epochs 100 \
    --transforms "bandpass:whiten" \
    --length 10000 \
    --sample_strategy "random" \
    --learning_rate 1e-4 \
    --weight_decay 1e-2 \
    --split "SAC_20100227_Chile_prem" \
    --embedding_size 10 \
    --loss_function "dpcl"