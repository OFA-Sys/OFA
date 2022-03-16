#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=2081
export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/gigaword_data/gigaword_test.tsv
path=../../checkpoints/gigaword_large_best.pt
result_path=../../results/gigaword
selected_cols=0,1
split='test'

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=gigaword \
    --batch-size=32 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=6 \
    --lenpen=0.7 \
    --max-len-b=32 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\"}"

python3 eval_rouge.py ${result_path}/test_predict.json
