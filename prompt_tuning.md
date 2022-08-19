<!---
Copyright 2022 The OFA-Sys Team. 
All rights reserved.
This source code is licensed under the Apache 2.0 license found in the LICENSE file in the root directory.
-->

## Prompt Tuning for Generative Multimodal Pretrained Models

### Overview
This is the code for **"Prompt Tuning for Generative Multimodal Pretrained Models"**, [Check our paper on ArXiv](https://arxiv.org/abs/2208.02532). This paper explores prompt tuning for generative multimodal pretrained models, instead of the constrastive learning models. We specifically focuses on the unified sequence-to-sequence learning framework and implement on our OFA models. 
<br>

### Requirements
* python 3.7.4
* pytorch 1.8.1
* torchvision 0.9.1
* JAVA 1.8 (for COCO evaluation)
<br></br>

### Installation
```bash
pip install -r requirements.txt
```
<br>

### Datasets and Checkpoints
See [datasets.md](datasets.md) and [checkpoints.md](checkpoints.md).
<br>

### Training
We provide a demo script (`run_scripts/refcoco/train_refcoco_prefix.sh`) that has all the required parts for training.

```sh
sh ./run_scripts/refcoco/train_refcoco_prefix.sh
```
A few options of note:
*   `--encoder-prompt` :: whether to insert prompts to the encoder
*   `--decoder-prompt` :: whether to insert prompts to the decoder
*   `--encoder-prompt-length` :: encoder prompt length
*   `--decoder-prompt-length` :: decoder prompt length
*   `--bitfit` :: whether to use bitfit
*   `--adapter` :: whether to use adapter
*   `--adapter-dim` :: adapter projection dim

We recommend that your workspace directory should be organized like this: 
```
OFA/
├── checkpoints/
│   ├── ofa_base.pt
│   ├── ofa_large.pt
│   └── ...
├── criterions/
├── data/
├── dataset/
│   ├── caption_data/
│   ├── refcoco_data/
│   └── ...
├── fairseq/
├── models/
├── run_scripts/
├── tasks/
├── train.py
├── trainer.py
└── utils/
```
<br>
