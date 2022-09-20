#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6081
export CUDA_VISIBLE_DEVICES=7
export GPUS_PER_NODE=1

user_dir=../../ofa_module
bpe_dir=../../utils/BERT_CN_dict
selected_cols=0,3,1,2   # make sure that you index "id, image, text, box"

# Choose your dataset split and the corresponding checkpoint
data=../../dataset/refcoco_cn_data/refcoco_val.tsv
path=../../checkpoints/refcoco_cn_large.pt
result_path=../../results/refcoco
split='refcoco_val'

python3 ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=refcoco \
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
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"
