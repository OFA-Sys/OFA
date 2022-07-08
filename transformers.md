# Use in huggingface transformers (Beta)

We now support inference of OFA on the huggingface transformers. In the near future, we will support training as well and merge OFA to the official transformers. 

Model checkpoints are stored in our [huggingface models](https://huggingface.co/OFA-Sys). Specifically, 5 versions of the pretrained OFA models, namely OFA-tiny, OFA-medium, OFA-base, OFA-large, and OFA-huge have been already uploaded. For more information about the models, please refer to the Model Card on our [README](https://github.com/OFA-Sys/OFA). 
Note that each directory includes 4 files, namely `config.json` which consists of model configuration, `vocab.json` and `merge.txt` for our OFA tokenizer, and lastly `pytorch_model.bin` which consists of model weights. There is no need to worry about the mismatch between Fairseq and transformers, since we have addressed the issue yet. 

To use it in transformers, please refer to https://github.com/OFA-Sys/OFA/tree/feature/add_transformers. Install the transformers and download the models (take OFA-tiny as an example) as shown below.

```
git clone --single-branch --branch feature/add_transformers https://github.com/OFA-Sys/OFA.git
pip install OFA/transformers/
git clone https://huggingface.co/OFA-Sys/OFA-tiny 
```

After, refer the path to OFA-tiny to `ckpt_dir`, and prepare an image for the testing example below. Also, ensure that you have pillow and torchvision in your environment. 

```
>>> from PIL import Image
>>> from torchvision import transforms
>>> from transformers import OFATokenizer, OFAModel
>>> from generate import sequence_generator

>>> mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
>>> resolution = 256
>>> patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((resolution, resolution), interpolation=Image.BICUBIC),
        transforms.ToTensor(), 
        transforms.Normalize(mean=mean, std=std)
    ])

>>> model = OFAModel.from_pretrained(ckpt_dir)
>>> tokenizer = OFATokenizer.from_pretrained(ckpt_dir)

>>> txt = " what does the image describe?"
>>> inputs = tokenizer([txt], return_tensors="pt").input_ids
>>> img = Image.open(path_to_image)
>>> patch_img = patch_resize_transform(img).unsqueeze(0)


>>> # using the generator of fairseq version
>>> generator = sequence_generator.SequenceGenerator(tokenizer=tokenizer,beam_size=5,
                                                                      max_len_b=16,
                                                                      min_len=0,
                                                                      no_repeat_ngram_size=3) # using the generator of fairseq version
>>> data = {}
>>> data["net_input"] = {"input_ids": inputs, 'patch_images': patch_img, 'patch_masks':torch.tensor([True])}
>>> gen_output = generator.generate([model], data)
>>> gen = [gen_output[i][0]["tokens"] for i in range(len(gen_output))]

>>> # using the generator of huggingface version
>>> gen = model.generate(inputs, patch_images=patch_img, num_beams=5, no_repeat_ngram_size=3) 

>>> print(tokenizer.batch_decode(gen, skip_special_tokens=True))
```
