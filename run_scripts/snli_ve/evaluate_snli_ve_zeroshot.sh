#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
selected_cols=0,2,3,4,5

data=../../dataset/snli_ve_data/snli_ve_dev.tsv
path=../../checkpoints/ofa_base.pt
result_path=../../results/snli_ve
split='snli_ve_zeroshot'
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --bpe-dir=${bpe_dir} \
    --selected-cols=${selected_cols} \
    --task=snli_ve \
    --patch-image-size=384 \
    --max-src-length=80 \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --zero-shot \
    --prompt-type='prev_output' \
    --fp16 \
    --num-workers=0