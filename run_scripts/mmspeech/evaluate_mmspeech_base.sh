#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1091

user_dir=../../ofa_module
bpe_dir=../../utils/BPE
data_dir=../../dataset/aishell1

bpe=bert
lang=zh
text2phone_dict_path=../../utils/phone/zh/text2phone_dict.txt
phone_dict_path=../../utils/phone/zh/phone_dict.txt
config_yaml_path=${data_dir}/fbank_config.yaml

data=${data_dir}/aishell_test.tsv
valid_data=${data_dir}/aishell_test.tsv
path=../../checkpoints/mmspeech_base_best.pt
result_path=../../results/asr
speech_text_selected_cols=0,1,2
split='test'

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
    ${data} \
    --valid-data=${valid_data} \
    --speech-text-selected-cols=${speech_text_selected_cols} \
    --path=${path} \
    --user-dir=${user_dir} \
    --task=unify_speech_text_task \
    --batch-size=16 \
    --log-format=simple --log-interval=10 \
    --seed=7 \
    --gen-subset=${split} \
    --results-path=${result_path} \
    --beam=5 \
    --max-len-b=256 \
    --no-repeat-ngram-size=5 \
    --fp16 \
    --num-workers=0 \
    --model-overrides="{\"valid_data\":\"${valid_data}\",\"config_yaml_path\":\"${config_yaml_path}\",\"train_stage\":4,\"eval_wer\":True}" \
    --constraint-range="4,21134"