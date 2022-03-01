#!/usr/bin/env

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=6052

log_dir=./refcocog_logs
save_dir=./refcocog_checkpoints
mkdir -p $log_dir $save_dir

bpe_dir=../../utils/BPE
user_dir=../../ofa_module

data_dir=../../dataset/refcocog_data
data=${data_dir}/refcocog_train.tsv,${data_dir}/refcocog_val.tsv
restore_file=../../checkpoints/ofa_large.pt
selected_cols=0,4,2,3

task=refcoco
arch=ofa_large
criterion=adjust_label_smoothed_cross_entropy
label_smoothing=0.1
lr=3e-5
max_epoch=5
warmup_ratio=0.06
batch_size=4
update_freq=8
resnet_drop_path_rate=0.0
encoder_drop_path_rate=0.2
decoder_drop_path_rate=0.2
dropout=0.1
attention_dropout=0.0
max_src_length=80
max_tgt_length=20
num_bins=1000
patch_image_size=512

for max_epoch in {10,}; do
  echo "max_epoch "${max_epoch}
  for lr in {3e-5,}; do
    echo "lr "${lr}
    for patch_image_size in {512,}; do
      echo "patch_image_size "${patch_image_size}

      log_file=${log_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}".log"
      save_path=${save_dir}/${max_epoch}"_"${lr}"_"${patch_image_size}
      mkdir -p $save_path

      CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../train.py \
          $data \
          --selected-cols=${selected_cols} \
          --bpe-dir=${bpe_dir} \
          --user-dir=${user_dir} \
          --restore-file=${restore_file} \
          --reset-optimizer --reset-dataloader --reset-meters \
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
          --fixed-validation-seed=7 \
          --no-epoch-checkpoints --keep-best-checkpoints=1 \
          --save-interval=1 --validate-interval=1 \
          --save-interval-updates=500 --validate-interval-updates=500 \
          --eval-acc \
          --eval-args='{"beam":5,"min_len":4,"max_len_a":0,"max_len_b":4}' \
          --best-checkpoint-metric=score --maximize-best-checkpoint-metric \
          --max-src-length=${max_src_length} \
          --max-tgt-length=${max_tgt_length} \
          --find-unused-parameters \
          --add-type-embedding \
          --scale-attn \
          --scale-fc \
          --scale-heads \
          --disable-entangle \
          --num-bins=${num_bins} \
          --patch-image-size=${patch_image_size} \
          --fp16 \
          --fp16-scale-window=512 \
          --num-workers=0 > ${log_file} 2>&1
    done
  done
done