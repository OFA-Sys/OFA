#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=1052
export GPUS_PER_NODE=1

bpe_dir=../../utils/BERT_CN_dict
user_dir=../../ofa_module
data_dir=../../dataset

bpe=bert
lang=zh

text2phone_dict_path=../../utils/phone/zh/text2phone_dict.txt
phone_dict_path=../../utils/phone/zh/phone_dict.txt
config_yaml_path=${data_dir}/aishell1/fbank_config.yaml

data=${data_dir}/aishell1/aishell_train.tsv
text_data=${data_dir}/aishell2/aishell_train.tsv
audio_data=${data_dir}/aishell2/aishell_train.tsv
speech_text_data=${data_dir}/aishell1/aishell_train.tsv
valid_data=${data_dir}/aishell1/aishell_dev.tsv

restore_file=../../checkpoints/mmspeech_base_stage2/checkpoint_best.pt

text_selected_cols=0,2
audio_selected_cols=0,1,2
speech_text_selected_cols=0,1,2

task=unify_speech_text_task
arch=ofa_speech_base
criterion=speech_pretrain_loss
label_smoothing=0.1
lr=5e-04
total_num_updates=300000
warmup_updates=15000
batch_size=32
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=256
max_tgt_length=128
num_bins=1000
patch_image_size=384
sample_patch_num=196
max_image_size=512
audio_code_dict_size=30000

save_path=../checkpoints/mmspeech_base_stage3/

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --master_port=${MASTER_PORT} ../../train.py \
  $data \
  --text-data=${text_data} \
  --audio-data=${audio_data} \
  --speech-text-data=${speech_text_data} \
  --valid-data=${valid_data} \
  --text-selected-cols=${text_selected_cols} \
  --audio-selected-cols=${audio_selected_cols} \
  --speech-text-selected-cols=${speech_text_selected_cols} \
  --bpe-dir=${bpe_dir} \
  --bpe=${bpe} \
  --lang=${lang} \
  --phone-dict-path=${phone_dict_path} \
  --text2phone-path=${text2phone_dict_path} \
  --config-yaml-path=${config_yaml_path} \
  --user-dir=${user_dir} \
  --eval-wer \
  --save-dir=${save_path} \
  --task=${task} \
  --arch=${arch} \
  --criterion=${criterion} \
  --label-smoothing=${label_smoothing} \
  --batch-size=${batch_size} \
  --update-freq=${update_freq} \
  --decoder-normalize-before \
  --share-decoder-input-output-embed \
  --share-all-embeddings \
  --layer-norm-first \
  --layernorm-embedding \
  --patch-layernorm-embedding \
  --code-layernorm-embedding \
  --resnet-drop-path-rate=${resnet_drop_path_rate} \
  --encoder-drop-path-rate=${encoder_drop_path_rate} \
  --decoder-drop-path-rate=${decoder_drop_path_rate} \
  --dropout=${dropout} \
  --attention-dropout=${attention_dropout} \
  --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=5.0 \
  --lr-scheduler=polynomial_decay --lr=${lr} \
  --total-num-update=${total_num_updates} --warmup-updates=${warmup_updates} \
  --log-format=simple --log-interval=10 \
  --fixed-validation-seed=7 \
  --keep-last-epochs=15 \
  --save-interval=1 \
  --save-interval-updates=6000 \
  --disable-validation \
  --max-src-length=${max_src_length} \
  --max-tgt-length=${max_tgt_length} \
  --add-type-embedding \
  --scale-attn \
  --scale-fc \
  --scale-heads \
  --num-bins=${num_bins} \
  --audio-code-dict-size=${audio_code_dict_size} \
  --code-dict-size=0 \
  --num-bins=0 \
  --patch-image-size=${patch_image_size} \
  --sample-patch-num=${sample_patch_num} \
  --max-image-size=${max_image_size} \
  --fp16 \
  --fp16-scale-window=128 \
  --num-workers=0 \
  --train-stage=3 \
  --sentence-avg \
  --restore-file=${restore_file} \
  --reset-optimizer --reset-dataloader --reset-meters \
  --reset-lr-scheduler
