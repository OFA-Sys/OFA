#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1083
export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS_PER_NODE=4

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/caption_data/caption_test.tsv
path=../../checkpoints/caption_huge_best.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'

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
    --beam=10 \
    --max-len-b=16 \
    --temperature=0.98 \
    --lenpen=0.98 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python coco_eval.py ../../results/caption/test_predict.json ../../dataset/caption_data/test_caption_coco_format.json
