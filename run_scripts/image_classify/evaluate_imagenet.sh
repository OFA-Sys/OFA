#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8087

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/imagenet_1k_data/imagenet_1k_val.tsv
ans2label_file=../../dataset/imagenet_1k_data/class2label_new.pkl
path=../../checkpoints/imagenet_1k_large_best.pt
result_path=../../results/imagenet_1k_val
selected_cols=0,2

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=image_classify \
    --batch-size=8 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=val \
    --results-path=${result_path} \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"