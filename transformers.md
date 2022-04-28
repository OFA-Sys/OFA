# Use in huggingface transformers (Beta)

We now support inference of OFA on the huggingface transformers. In the near future, we will support training as well and merge OFA to the official transformers. 

Model checkpoints are stored in our [huggingface models](https://huggingface.co/OFA-Sys). Specifically, 4 versions of the pretrained OFA models, namely OFA-tiny, OFA-medium, OFA-base, and OFA-large, have been already uploaded. For more information about the models, please refer to the Model Card on our [README](https://github.com/OFA-Sys/OFA). 
Note that each directory includes 4 files, namely `config.json` which consists of model configuration, `vocab.json` and `merge.txt` for our OFA tokenizer, and lastly `pytorch_model.bin` which consists of model weights. There is no need to worry about the mismatch between Fairseq and transformers, since we have addressed the issue yet. 

To use it in transformers, please refer to https://github.com/OFA-Sys/OFA/tree/feature/add_transformers and download the directory of transformers. After installation, download our models and save them to `ckpt_dir`, and prepare an image for the testing example below. Also, ensure that you have pillow and torchvision in your environment. 

```
>>> from PIL import Image
>>> from torchvision import transforms
>>> from transformers import OFATokenizer, OFAForConditionalGeneration

>>> mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
>>> resolution = 256
>>> patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])

>>> model = OFAForConditionalGeneration.from_pretrained(ckpt_dir)
>>> tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

>>> txt = " what is the description of the image?"
>>> inputs = tokenizer([txt], max_length=1024, return_tensors="pt")["input_ids"]
>>> img = Image.open(path_to_image)
>>> patch_img = patch_resize_transform(img).unsqueeze(0)

>>> gen = model.generate(inputs, patch_img=patch_img, num_beams=4)
>>> print(tokenizer.batch_decode(gen, skip_special_tokens=True))
```
