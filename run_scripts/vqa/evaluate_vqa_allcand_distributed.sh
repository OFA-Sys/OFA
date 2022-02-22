#!/usr/bin/env bash

# Guide:
# This script supports distributed inference on multi-gpu workers (as well as single-worker inference). 
# Please set the options below according to the comments. 
# For multi-gpu workers inference, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.

# Number of GPUs per GPU worker
GPUS_PER_NODE=8 
# Number of GPU workers, for single-worker inference, please set to 1
WORKER_CNT=4 
# The ip address of the rank-0 worker, for single-worker inference, please set to localhost
export MASTER_ADDR=XX.XX.XX.XX
# The port for communication
export MASTER_PORT=8216
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker inference, please set to 0
export RANK=0 

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# val or test
split=$1

data=../../dataset/vqa_data/vqa_${split}.tsv
ans2label_file=../../dataset/vqa_data/trainval_ans2label.pkl
path=../../checkpoints/vqa_large_best.pt
result_path=../../results/vqa_${split}_allcand
selected_cols=0,5,2,3,4

python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=vqa_gen \
    --batch-size=4 \
    --valid-batch-size=20 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --fp16 \
    --ema-eval \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"ans2label_file\":\"${ans2label_file}\"}"