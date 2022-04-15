#!/usr/bin/env

# for text-image paired data, each line of the given input file should contain these information (separated by tabs):
# input format
#   uniq-id, image-id, image base64 string and text
# input example
#   162365  12455 /9j/4AAQSkZJ....UCP/2Q==  two people in an ocean playing with a yellow frisbee.
#
# output format
#   uniq-id, image-id, text and code
# output example
#   162364 12455 two people in an ocean playing with a yellow frisbee.  6288 4495 4139...4691 4844 6464

CUDA_VISIBLE_DEVICES=0 python generate_code.py \
  --file ./custom_data.txt \
  --outputs ./custom_data_code.txt \
  --selected_cols 0,1,2,3 \
  --code_image_size 256 \
  --vq_model vqgan \
  --vqgan_model_path ../../checkpoints/vqgan/last.ckpt \
  --vqgan_config_path ../../checkpoints/vqgan/model.yaml

# for image-only data each line of the given input file should contain these information (separated by tabs):
# input format
#   image-id and image base64 string
# input example:
#   12455 /9j/4AAQSkZJ....UCP/2Q==
#
# output format
#   image-id and code
#   12455 6288 4495 4139...4691 4844 6464

CUDA_VISIBLE_DEVICES=0 python generate_code.py \
  --file ./custom_data.txt \
  --outputs ./custom_data_code.txt \
  --selected_cols 0,1 \
  --code_image_size 256 \
  --vq_model vqgan \
  --vqgan_model_path ../../checkpoints/vqgan/last.ckpt \
  --vqgan_config_path ../../checkpoints/vqgan/model.yaml
