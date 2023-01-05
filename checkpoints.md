# Checkpoints

We provide links for you to download our checkpoints, including pretrained and finetuned models on different tasks. If you would like to use OFA with Transformers, please download checkpoints at [https://huggingface.co/OFA-Sys](https://huggingface.co/OFA-Sys), and check the code in the branch `feature/add_transformers`. 

## Pretraining
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_huge.pt"> Pre-trained checkpoint (OFA-Huge) </a> (~930M parameters)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_large.pt"> Pre-trained checkpoint (OFA-Large) </a> (~470M parameters)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_base.pt"> Pre-trained checkpoint (OFA-Base) </a> (~180M parameters)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_medium.pt"> Pre-trained checkpoint (OFA-Medium) </a> (~93M parameters)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_tiny.pt"> Pre-trained checkpoint (OFA-Tiny) </a> (~33M parameters)

## Finetuning (OFA-Huge)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_huge_best.pt"> Finetuned checkpoint for Caption on COCO </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vqa_huge_best.pt"> Finetuned checkpoint for VQAv2 </a>

## Finetuning (OFA-Large)

* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_large_best_clean.pt"> Finetuned checkpoint for Caption on COCO </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_stage1_best.pt"> Finetuned checkpoint for Caption on COCO During Stage1 Finetuning </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcoco_large_best.pt"> Finetuned checkpoint for RefCOCO </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocoplus_large_best.pt"> Finetuned checkpoint for RefCOCO+ </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocog_large_best.pt"> Finetuned checkpoint for RefCOCOg </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vqa_large_best.pt"> Finetuned checkpoint for VQAv2 </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/snli_ve_large_best.pt"> Finetuned checkpoint for SNLI-VE </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/image_gen_large_best.zip"> Finetuned checkpoint for Text-to-Image Generation on COCO && CLIP checkpoint && VQGAN checkpoint </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/imagenet_1k_large_best.pt"> Finetuned checkpoint for ImageNet-1K </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/gigaword_large_best.pt"> Finetuned checkpoint for Gigaword </a>

## Finetuning (OFA-Base)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_base_best.pt"> Finetuned base checkpoint for Caption on COCO </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcoco_base_best.pt"> Finetuned base checkpoint for RefCOCO </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocoplus_base_best.pt"> Finetuned base checkpoint for RefCOCO+ </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocog_base_best.pt"> Finetuned base checkpoint for RefCOCOg </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/vqa_base_best.pt"> Finetuned base checkpoint for VQAv2 </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/snli_ve_base_best.pt"> Finetuned base checkpoint for SNLI-VE </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/image_gen_base_best.pt"> Finetuned base checkpoint for Text-to-Image Generation on COCO  </a>

## Pretrained Language Models
To follow our multimodal pretraining, we suggest using pretrained language models for the initialization. Note that for the base-size and large-size models, we directly use BART-base and BART-large, and for the other sizes, we pretrained the tiny-size, medium size, and huge-size OFA-based language models. 
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_tiny_plaintext.pt"> Tiny-size encoder-decoder language model (OFA) </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_medium_plaintext.pt"> Medium-size encoder-decoder language model (OFA) </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/bart_base.pt"> Base-size encoder-decoder language model (BART) </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/bart_large.pt"> Large-size encoder-decoder language model (BART) </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_huge_plaintext.pt"> Huge-size encoder-decoder language model (OFA) </a>
