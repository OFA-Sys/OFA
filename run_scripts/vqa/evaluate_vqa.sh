#!/usr/bin/env bash

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split=$1

data=../../dataset/vqa_data/vqa_${split}.tsv
ans2label_file=../../dataset/vqa_data/trainval_ans2label.pkl
path=../../checkpoints/vqa_large_best_clean.pt
result_path=../../results/vqa
selected_cols=0,5,2,3,4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=8 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --ema-eval \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"