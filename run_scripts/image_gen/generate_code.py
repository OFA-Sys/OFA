import sys

sys.path.append('../../')
import argparse
import base64
from io import BytesIO
from data.file_dataset import FileDataset
from PIL import Image, ImageFile
from torchvision import transforms
from omegaconf import OmegaConf
from models.taming.models.vqgan import GumbelVQ
import os

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None


class VQGANDataset(Dataset):
    def __init__(self, file, selected_cols):
        self.reader = FileDataset(
            file,
            selected_col_ids=selected_cols,
        )
        self.code_resize_transform = transforms.Compose([
            lambda image: image.convert("RGB"),
            transforms.Resize(args.code_image_size, interpolation=Image.LANCZOS),
            transforms.ToTensor(),
            preprocess_vqgan
        ])

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, item):
        column_l = self.reader[item]
        if len(column_l) == 4:
            pair_id, image_id, image, text = column_l
        elif len(column_l) == 2:
            image_id, image = column_l
        else:
            raise NotImplementedError

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image)))
        code_image = self.code_resize_transform(image)
        if len(column_l) == 4:
            return {"code_image": code_image, "pair_id": pair_id, "image_id": image_id, "text": text}
        elif len(column_l) == 2:
            return {"code_image": code_image, "image_id": image_id}


def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")
    return x


def map_pixels(x, eps=0.1):
    return (1 - 2 * eps) * x + eps


def preprocess_vqgan(x):
    x = 2. * x - 1.
    return x


def image_to_base64(img, format):
    output_buffer = BytesIO()
    img.save(output_buffer, format=format)
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    base64_str = str(base64_str, encoding='utf-8')
    return base64_str


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, default="")
    parser.add_argument("--outputs", type=str, default="")
    parser.add_argument("--selected_cols", type=str, required=True)
    parser.add_argument("--code_image_size", type=int, required=True)
    parser.add_argument("--vq_model", type=str, required=True)
    parser.add_argument("--vqgan_model_path", type=str, default=None)
    parser.add_argument("--vqgan_config_path", type=str, default=None)
    parser.add_argument("--log_interval", default=100, type=int, help="log interval")
    parser.add_argument("--worker_cnt", type=int, default=1)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    vqgan_config = OmegaConf.load(args.vqgan_config_path)
    vqgan = GumbelVQ(**vqgan_config.model.params)
    sd = torch.load(args.vqgan_model_path, map_location="cpu")["state_dict"]
    missing, unexpected = vqgan.load_state_dict(sd, strict=False)
    for k, v in vqgan.named_parameters():
        v.requires_grad = False
    image_tokenizer = vqgan.cuda().eval()

    writer = open(args.outputs, 'w')

    print("begin process")

    data_cnt = 0

    dataset = VQGANDataset(args.file, args.selected_cols)
    dataloader = DataLoader(dataset, batch_size=args.batch_size)

    for data in dataloader:
        batch_size = data["code_image"].size()[0]
        with torch.no_grad():
            z, _, [_, _, image_codes] = image_tokenizer.encode(data["code_image"].cuda())
            image_codes = image_codes.view(batch_size, -1).detach()

        for i, image_code in enumerate(image_codes):
            code = ' '.join([str(num) for num in image_code.tolist()])

            if len(data.keys()) == 4:
                writer.write('\t'.join([data['pair_id'][i], data['image_id'][i], data['text'][i], code])+'\n')
            elif len(data.keys()) == 2:
                writer.write('\t'.join([data['image_id'][i], code])+'\n')
            else:
                raise NotImplementedError
    writer.close()

    print("finish")
