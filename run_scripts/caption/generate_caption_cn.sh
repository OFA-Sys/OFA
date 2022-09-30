#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1081
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export GPUS_PER_NODE=8

user_dir=../../ofa_module
bpe_dir=../../utils/BERT_CN_dict

data=../../dataset/muge_caption_data/caption_test.tsv
path=../../checkpoints/caption_cn_large.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'

# We do not provide scripts for the evaluation. For more information, please refer to https://tianchi.aliyun.com/muge
python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=8 \
    --max-len-b=128 \
    --no-repeat-ngram-size=3 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"
