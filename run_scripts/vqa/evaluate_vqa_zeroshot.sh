#!/usr/bin/env bash

# This script evaluates pretrained OFA-Large checkpoint on zero-shot open-domain VQA task.

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split=$1

data=../../dataset/vqa_data/vqa_${split}.tsv
path=../../checkpoints/ofa_large.pt
result_path=../../results/vqa_${split}_zeroshot
selected_cols=0,5,2,3,4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --selected-cols=${selected_cols} \
    --bpe-dir=${bpe_dir} \
    --patch-image-size=480 \
    --prompt-type='none' \
    --batch-size=8 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --zero-shot \
    --beam=20 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0