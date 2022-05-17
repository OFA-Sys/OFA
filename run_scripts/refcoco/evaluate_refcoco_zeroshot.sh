#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=2,3,4,5
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,4,2,3

data=../../dataset/refcoco_data/refcoco_val.tsv
path=../../checkpoints/ofa_large.pt
result_path=../../results/refcoco
split='refcoco_val'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --selected-cols=${selected_cols} \
    --task=refcoco \
    --patch-image-size=480 \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --min-len=4 \
    --max-len-a=0 \
    --max-len-b=4 \
    --no-repeat-ngram-size=3 \
    --zero-shot \
    --fp16 \
    --num-workers=0