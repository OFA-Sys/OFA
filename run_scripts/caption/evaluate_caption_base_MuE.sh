#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1091

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=/data/tsk/caption_data/caption_test.tsv
path=/home/sht22008/tsk/projects/OFA/run_scripts/caption/stage1_checkpoints/{5,}_{0.06,}_{6000,}/checkpoint.best_cider_0.3240.pt
result_path=../../results/caption
selected_cols=1,4,2
split='test'

CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=caption \
    --batch-size=1 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=16 \
    --no-repeat-ngram-size=3 \
    --fp16 \
    --num-workers=0 \
    --img_thres=0.99\
    --txt_thres=0.7\
    --decoder_thres=0.9\
    --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"eval_cider\":False,\"selected_cols\":\"${selected_cols}\"}"

python coco_eval.py ../../results/caption/test_predict.json ../../dataset/caption_data/test_caption_coco_format.json
