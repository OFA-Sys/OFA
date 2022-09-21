#!/usr/bin/env bash

# This script evaluates **unconstrainedly** finetuned OFA-Large checkpoint (with --unconstrained-training set to True during finetuning)
# which does not use a fixed candidate answer set (trainval_ans2label.pkl).
# For more details about the unconstrained finetuning, refer to Line 62-68 in train_vqa_distributed.sh

# Usage: bash evaluate_vqa_unconstrained.sh ${split} ${ckpt_path}

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=8082

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split=$1

data=../../dataset/vqa_data/vqa_${split}.tsv
path=$2 # please speficy your path of unconstrainedly finetuned checkpoint
result_path=../../results/vqa_${split}_unconstrained
selected_cols=0,5,2,3,4

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --ema-eval \
    --beam-search-vqa-eval \
    --beam=5 \
    --unnormalized \
    --temperature=1.0 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"