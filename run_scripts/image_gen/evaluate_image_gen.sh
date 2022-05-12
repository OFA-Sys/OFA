#!/usr/bin/env bash

# The port for communication. Note that if you want to run multiple tasks on the same machine,
# you need to specify different port numbers.
export MASTER_PORT=4081

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

# It may take a long time for the full evaluation. You can sample a small split from the full_test split.
# But please remember that you need at least thousands of images to compute FID and IS, otherwise the resulting scores
# might also no longer correlate with visual quality.

data=../../dataset/coco_image_gen_data/coco_vqgan_full_test.tsv
path=../../checkpoints/image_gen_large_best.pt
selected_cols=0,2,1
split='test'
VQGAN_MODEL_PATH=../../checkpoints/vqgan/last.ckpt
VQGAN_CONFIG_PATH=../../checkpoints/vqgan/model.yaml
CLIP_MODEL_PATH=../../checkpoints/clip/ViT-B-16.pt
GEN_IMAGE_PATH=../../results/image_gen

CUDA_VISIBLE_DEVICES=4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=4 --master_port=${MASTER_PORT} ../../evaluate.py \
  ${data} \
  --path=${path} \
  --user-dir=${user_dir} \
  --task=image_gen \
  --batch-size=1 \
  --log-format=simple --log-interval=1 \
  --seed=42 \
  --gen-subset=${split} \
  --beam=24 \
  --min-len=1024 \
  --max-len-a=0 \
  --max-len-b=1024 \
  --sampling-topk=256 \
  --temperature=1.0 \
  --code-image-size=256 \
  --constraint-range=50265,58457 \
  --fp16 \
  --num-workers=0 \
  --model-overrides="{\"data\":\"${data}\",\"bpe_dir\":\"${bpe_dir}\",\"selected_cols\":\"${selected_cols}\",\"clip_model_path\":\"${CLIP_MODEL_PATH}\",\"vqgan_model_path\":\"${VQGAN_MODEL_PATH}\",\"vqgan_config_path\":\"${VQGAN_CONFIG_PATH}\",\"gen_images_path\":\"${GEN_IMAGE_PATH}\"}"

# install requiremnts
pip install scipy
# compute IS
python inception_score.py --gpu 4 --batch-size 128 --path1 ${GEN_IMAGE_PATH}/top1
# compute FID, download statistics for test dataset first.
wget https://ofa-beijing.oss-cn-beijing.aliyuncs.com/datasets/coco_image_gen_data/gt_stat.npz
python fid_score.py --gpu 4 --batch-size 128 --path1 ${GEN_IMAGE_PATH}/top1 --path2 ./gt_stat.npz