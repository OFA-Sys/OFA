#!/usr/bin/env

# Guide:
# This script supports distributed training on multi-gpu workers (as well as single-worker training). 
# Please set the options below according to the comments. 
# For multi-gpu workers training, these options should be manually set for each worker. 
# After setting the options, please run the script on each worker.
# To use the shuffled data (if exists), please uncomment the Line 23.

# Number of GPUs per GPU worker
GPUS_PER_NODE=8 
# Number of GPU workers, for single-worker training, please set to 1
WORKER_CNT=4 
# The ip address of the rank-0 worker, for single-worker training, please set to localhost
export MASTER_ADDR=XX.XX.XX.XX
# The port for communication
export MASTER_PORT=8215
# The rank of this worker, should be in {0, ..., WORKER_CNT-1}, for single-worker training, please set to 0
export RANK=0
data_dir=../../dataset/coco_image_gen_data
data=${data_dir}/coco_vqgan_train.tsv,${data_dir}/coco_vqgan_dev.tsv
# Note: If you have shuffled the data in advance, please uncomment the line below.
# data=${data_dir}/coco_vqgan_train_1.tsv,${data_dir}/coco_vqgan_train_2.tsv,${data_dir}/coco_vqgan_train_3.tsv,${data_dir}/coco_vqgan_train_4.tsv,${data_dir}/coco_vqgan_train_5.tsv,${data_dir}/coco_vqgan_train_6.tsv,${data_dir}/coco_vqgan_train_7.tsv,${data_dir}/coco_vqgan_train_8.tsv,${data_dir}/coco_vqgan_train_9.tsv,${data_dir}/coco_vqgan_train_10.tsv,${data_dir}/coco_vqgan_dev.tsv
restore_file=../../checkpoints/40000_2000_1e-3/checkpoint_last.pt
selected_cols=0,2,1

log_dir=./image_gen_stage2_logs
save_dir=./image_gen_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

task=image_gen
arch=ofa_large
criterion=clip_scst_reward_criterion
batch_size=1
update_freq=1
encoder_drop_path_rate=0.0
decoder_drop_path_rate=0.0
dropout=0.0
attention_dropout=0.0
max_src_length=64
max_tgt_length=1024
num_bins=1000
code_image_size=256
constraint_range=50265,58457

VQGAN_MODEL_PATH=../../checkpoints/vqgan/last.ckpt
VQGAN_CONFIG_PATH=../../checkpoints/vqgan/model.yaml
CLIP_MODEL_PATH=../../checkpoints/clip/ViT-B-16.pt
GEN_IMAGES_PATH=../../results/image_gen_stage2

for total_num_updates in 5000; do
  echo "total_num_updates "${total_num_updates}
  for warmup_updates in 0; do
    echo "warmup_updates "${warmup_updates}  
    for lr in 1e-6; do
      echo "lr "${lr}

        log_file=${log_dir}/${total_num_updates}"_"${warmup_updates}"_"${lr}"_rank"${RANK}".log"
        save_path=${save_dir}/${total_num_updates}"_"${warmup_updates}"_"${lr}
        mkdir -p $save_path

        python3 -m torch.distributed.launch --nproc_per_node=${GPUS_PER_NODE} --nnodes=${WORKER_CNT} --node_rank=${RANK} --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} ../../train.py \
            ${data} \
            --selected-cols=${selected_cols} \
            --bpe-dir=${bpe_dir} \
            --user-dir=${user_dir} \
            --restore-file=${restore_file} \
            --reset-optimizer --reset-dataloader --reset-meters \
            --save-dir=${save_path} \
            --task=${task} \
            --arch=${arch} \
            --criterion=${criterion} \
            --batch-size=${batch_size} \
            --batch-size-valid=1 \
            --update-freq=${update_freq} \
            --encoder-normalize-before \
            --decoder-normalize-before \
            --share-decoder-input-output-embed \
            --share-all-embeddings \
            --layernorm-embedding \
            --patch-layernorm-embedding \
            --code-layernorm-embedding \
            --encoder-drop-path-rate=${encoder_drop_path_rate} \
            --decoder-drop-path-rate=${decoder_drop_path_rate} \
            --dropout=${dropout} \
            --attention-dropout=${attention_dropout} \
            --weight-decay=0.01 \
            --optimizer=adam \
            --adam-betas="(0.9,0.999)" \
            --adam-eps=1e-08 \
            --clip-norm=1.0 \
            --lr-scheduler=polynomial_decay \
            --lr=${lr} \
            --total-num-update=${total_num_updates} \
            --warmup-updates=${warmup_updates} \
            --log-format=simple \
            --log-interval=10 \
            --fixed-validation-seed=7 \
            --keep-last-epochs=15 \
            --save-interval=1 --validate-interval=1 \
            --save-interval-updates=200 --validate-interval-updates=200 \
            --freeze-resnet \
            --max-update=${total_num_updates} \
            --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
            --eval-args='{"beam":24,"min_len":1024,"max_len_a":0,"max_len_b":1024,"sampling_topk":256,"temperature":1.0}' \
            --scst \
            --scst-args='{"beam":5,"min_len":1024,"max_len_a":0,"max_len_b":1024,"sampling_topk":256,"temperature":1.0}' \
            --max-src-length=${max_src_length} \
            --max-tgt-length=${max_tgt_length} \
            --find-unused-parameters \
            --add-type-embedding \
            --scale-attn \
            --scale-fc \
            --scale-heads \
            --disable-entangle \
            --num-bins=${num_bins} \
            --code-image-size=${code_image_size} \
            --constraint-range=${constraint_range} \
            --vqgan-model-path=${VQGAN_MODEL_PATH} \
            --vqgan-config-path=${VQGAN_CONFIG_PATH} \
            --clip-model-path=${CLIP_MODEL_PATH} \
            --gen-images-path=${GEN_IMAGES_PATH} \
            --fp16 \
            --fp16-scale-window=256 \
            --num-workers=0 > ${log_file} 2>&1
    done
  done
done
