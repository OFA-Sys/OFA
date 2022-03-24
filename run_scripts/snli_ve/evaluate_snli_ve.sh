#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=7081

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# dev or test
split=dev

data=../../dataset/snli_ve_data/snli_ve_${split}.tsv
path=dummy_checkpoints/6_2e-6/checkpoint_best.pt

result_path=../../results/snli_ve
selected_cols=0,2,3,4,5

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=snli_ve \
    --batch-size=6 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
