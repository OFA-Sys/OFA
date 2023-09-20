#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6062
# export NCCL_DEBUG=INFO
# export NCCL_SOCKET_IFNAME=eno1

log_dir=./scene_graph_logs
save_dir=/data/hulab/zcai75/checkpoints/OFA/scene_graph_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module
data_dir=/data/hulab/zcai75/OFA_data/vg
data=${data_dir}/vg_train_full.tsv,${data_dir}/vg_val_toy.tsv
# restore_file=/data/hulab/zcai75/checkpoints/OFA/scene_graph_checkpoints/_2_3e-5_512_/checkpoint1.pt
# restore_file=/data/hulab/zcai75/checkpoints/OFA/scene_graph_checkpoints/prompt_tuning_2_3e-5_512_prompt_tuning_sgdet/checkpoint1.pt 
# restore_file=/data/hulab/zcai75/checkpoints/OFA/scene_graph_checkpoints/adapter_10_3e-5_512_sgdet/checkpoint10.pt 
# restore_file=/data/hulab/zcai75/checkpoints/OFA/scene_graph_checkpoints/adapter_all_cand_20_3e-5_512_sgdet/checkpoint19.pt
restore_file=/data/hulab/zcai75/checkpoints/OFA/scene_graph_checkpoints/adapter_all_cand_20_3e-5_512_sgcls/checkpoint_last.pt
# restore_file=/data/hulab/zcai75/checkpoints/OFA/ofa_base.pt

tag=adapter_all_cand
task=scene_graph
arch=ofa_base
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=20
warmup_ratio=0.06
batch_size=8
update_freq=4
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.1
decoder_drop_path_rate=0.1
dropout=0.1
attention_dropout=0.0
max_src_length=200
max_tgt_length=200
# num_bins=480
num_bins=1000
patch_image_size=512

prompt_type_method=prefix
encoder_prompt_length=100
decoder_prompt_length=100

sg_mode=sgcls

log_file=${log_dir}/${tag}_${max_epoch}"_"${lr}"_"${patch_image_size}_${sg_mode}".log"
save_path=${save_dir}/${tag}_${max_epoch}"_"${lr}"_"${patch_image_size}_${sg_mode}
tensorboard_logdir=./tensorboard/${tag}_${max_epoch}"_"${lr}"_"${patch_image_size}
mkdir -p $save_path

# if [ -f $log_file ]; then
#     echo "Log file ${log_file} already exists, aborting..."
#     exit 1
# fi

#     --reset-optimizer --reset-dataloader --reset-meters \

    # --encoder-prompt \
    # --decoder-prompt \
    # --encoder-prompt-type=${prompt_type_method} \
    # --decoder-prompt-type=${prompt_type_method} \
    # --encoder-prompt-length=${encoder_prompt_length} \
    # --decoder-prompt-length=${decoder_prompt_length} \

CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --master_port=${MASTER_PORT} ../../train.py \
    $data \
    --sg-mode=${sg_mode} \
    --bpe-dir=${bpe_dir} \
    --user-dir=${user_dir} \
    --restore-file=${restore_file} \
    --save-dir=${save_path} \
    --task=${task} \
    --arch=${arch} \
    --criterion=${criterion} \
    --label-smoothing=${label_smoothing} \
    --batch-size=${batch_size} \
    --update-freq=${update_freq} \
    --encoder-normalize-before \
    --decoder-normalize-before \
    --share-decoder-input-output-embed \
    --share-all-embeddings \
    --layernorm-embedding \
    --patch-layernorm-embedding \
    --code-layernorm-embedding \
    --resnet-drop-path-rate=${resnet_drop_path_rate} \
    --encoder-drop-path-rate=${encoder_drop_path_rate} \
    --decoder-drop-path-rate=${decoder_drop_path_rate} \
    --dropout=${dropout} \
    --attention-dropout=${attention_dropout} \
    --weight-decay=0.01 --optimizer=adam --adam-betas="(0.9,0.999)" --adam-eps=1e-08 --clip-norm=1.0 \
    --lr-scheduler=polynomial_decay --lr=${lr} \
    --max-epoch=${max_epoch} --warmup-ratio=${warmup_ratio} \
    --log-format=simple --log-interval=10 \
    --tensorboard-logdir=${tensorboard_logdir} \
    --wandb-project=OFA-VG \
    --fixed-validation-seed=7 \
    --keep-best-checkpoints=1 \
    --save-interval=1 --validate-interval=3 --validate-interval-updates=1 \
    --all-gather-list-size=2097152 \
    --eval-args='{"beam":5,"max_len_a":0,"max_len_b":999,"no_repeat_ngram_size":5}' \
    --best-checkpoint-metric=loss --maximize-best-checkpoint-metric \
    --max-src-length=${max_src_length} \
    --max-tgt-length=${max_tgt_length} \
    --find-unused-parameters \
    --add-type-embedding \
    --scale-attn \
    --scale-fc \
    --freeze-encoder \
    --scale-heads \
    --disable-entangle \
    --num-bins=${num_bins} \
    --patch-image-size=${patch_image_size} \
    --fp16 \
    --fp16-scale-window=512 \
    --num-workers=0 > ${log_file} 2>&1
