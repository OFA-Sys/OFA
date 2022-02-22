#!/usr/bin/env bash

user_dir=../../ofa_module
bpe_dir=../../utils/BPE

data=../../dataset/coco_text2image/coco_vqgan_dev.tsv
path=../../checkpoints/image_gen.pt
selected_cols=0,2,1
split='test'
VQGAN_MODEL_PATH=../../checkpoints/vqgan/last.ckpt
VQGAN_CONFIG_PATH=../../checkpoints/vqgan/model.yaml
CLIP_MODEL_PATH=../../checkpoints/clip/ViT-B-16.pt
GEN_IMAGE_PATH=../../results/image_gen

CUDA_VISIBLE_DEVICES=4,5,6,7 python ../../evaluate.py \
  ${data} \
  --path=${path} \
  --user-dir=${user_dir} \
  --task=image_gen \
  --batch-size=1 \
  --log-format=simple --log-interval=10 \
  --seed=7 \
  --gen-subset=${split} \
  --beam=16 \
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

cd ./eval_utils
# compute IS
python inception_score.py --gpu 4 --batch-size 128 --path1 ../${GEN_IMAGE_PATH}/top1
# compute FID
wget https://jirenmr.oss-cn-zhangjiakou.aliyuncs.com/ofa/gt_stat.npz
python fid_score.py --gpu 4 --batch-size 128 --path1 ../${GEN_IMAGE_PATH}/top1 --path2 ./gt_stat.npz