from io import BytesIO

import logging
import warnings
import string

import numpy as np
import torch
import base64
from torchvision import transforms
import h5py

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def coord2bin(coord_list, box_size, w, h, max_img_size, num_bins):
	# coord / box_size(1024) * max_img_size / w_or_h
	bin_list = []
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[0] / box_size * max_img_size / w * (num_bins - 1)))))]
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[1] / box_size * max_img_size / h * (num_bins - 1)))))]
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[2] / box_size * max_img_size / w * (num_bins - 1)))))]
	bin_list += ["<bin_{}>".format(int(max(0, round(coord_list[3] / box_size * max_img_size / h * (num_bins - 1)))))]
	assert '<bin_-1>' not in bin_list, 'coord2bin error!'
	return ' '.join(bin_list)

def collate(samples, pad_idx, eos_idx):
	if len(samples) == 0:
		return {}

	def merge(key):
		return data_utils.collate_tokens(
			[s[key] for s in samples],
			pad_idx,
			eos_idx=eos_idx,
		)

	id = np.array([s["id"] for s in samples])
	src_tokens = merge("source")
	src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

	patch_images = torch.stack([sample['patch_image'] for sample in samples], dim=0)
	patch_masks = torch.cat([sample['patch_mask'] for sample in samples])

	prev_output_tokens = None
	target = None
	if samples[0].get("target", None) is not None:
		target = merge("target")
		tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
		ntokens = tgt_lengths.sum().item()

		if samples[0].get("prev_output_tokens", None) is not None:
			prev_output_tokens = merge("prev_output_tokens")
	else:
		ntokens = src_lengths.sum().item()

	gt_relations = [sample['gt_relations'] for sample in samples]
	gt_boxes = [sample['gt_boxes'] for sample in samples]
	gt_classes = [sample['gt_classes'] for sample in samples]
	img_size = [sample['img_size'] for sample in samples]

	batch = {
		"id": id,
		"nsentences": len(samples),
		"ntokens": ntokens,
		"net_input": {
			"src_tokens": src_tokens,
			"src_lengths": src_lengths,
			"patch_images": patch_images,
			"patch_masks": patch_masks,
			"prev_output_tokens": prev_output_tokens
		},
		"target": target,
		"gt_relations": gt_relations,
		"gt_boxes": gt_boxes,
		"gt_classes": gt_classes,
		"img_size": img_size
	}

	return batch



class SGDataset(OFADataset):
	def __init__(
		self,
		split,
		dataset,
		bpe,
		src_dict,
		tgt_dict=None,
		max_src_length=128,
		max_tgt_length=30,
		patch_image_size=224,
		num_bins=480,
		imagenet_default_mean_and_std=False,
		mode='sgg'
	):
		super().__init__(split, dataset, bpe, src_dict, tgt_dict)
		self.max_src_length = max_src_length
		self.max_tgt_length = max_tgt_length
		self.patch_image_size = patch_image_size
		self.num_bins = num_bins

		if imagenet_default_mean_and_std:
			mean = IMAGENET_DEFAULT_MEAN
			std = IMAGENET_DEFAULT_STD
		else:
			mean = [0.5, 0.5, 0.5]
			std = [0.5, 0.5, 0.5]

		self.patch_resize_transform = transforms.Compose([
			lambda image: image.convert("RGB"),
			transforms.Resize((patch_image_size, patch_image_size), interpolation=Image.BICUBIC),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])

		# if type(bpe).__name__ == 'GPT2BPE':
		# 	self.prompt = " What are the relations in the image?"
		# elif type(bpe).__name__ == 'BertBPE':
		# 	self.prompt = "图片描述了什么内容?"
		self.prompt = ''

		self.mode = mode

	def __getitem__(self, index):
		uniq_id, pred_ids, box_ids, img_rels, boxes, pred_label, box_label, img_str = self.dataset[index]

		image = Image.open(BytesIO(base64.urlsafe_b64decode(img_str)))
		img_size = image.size
		patch_image = self.patch_resize_transform(image)
		patch_mask = torch.tensor([True])

		img_rels = np.array([list(map(int, rels.split())) for rels in img_rels.split(',')])
		max_img_size = max(image.width, image.height)

		box_ids = list(map(int, box_ids.split(',')))
		boxes = [list(map(int, box.split())) for box in boxes.split(',')]
		boxes_bin = [coord2bin(box, 1024, image.width, image.height, max_img_size, self.num_bins) for box in boxes]

		box_label = box_label.split(',')
		pred_label = pred_label.split(',')
		pred_ids = list(map(int, pred_ids.split(',')))

		assert all(id > 0 and id < 51 for id in pred_ids)
		assert all(id > 0 and id < 151 for id in box_ids)

		if self.mode == 'sgdet':
			self.prompt = " What are the relations in the image?"
		elif self.mode == 'sgcls':
			self.prompt = " Describe the relations between these objects " + ', '.join(boxes_bin)
		elif self.mode == 'predcls':
			self.prompt = " Describe the relations between these objects " + ', '.join(map(' '.join, zip(box_label, boxes_bin)))

		dic = {}
		for i, rel in enumerate(img_rels):
			pred = self.bpe.encode(' {}'.format(pred_label[i]))
			if rel[0] not in dic:
				dic[rel[0]] = {rel[1]: pred}
			else:
				dic[rel[0]][rel[1]] = pred

		caption = ""
		for r1 in dic:
			l1 = self.bpe.encode(' {}'.format(box_label[r1]))
			caption += "<sub> {} {} ".format(l1, boxes_bin[r1])
			for r2 in dic[r1]:
				l2 = self.bpe.encode(' {}'.format(box_label[r2]))
				caption += "<pred> {} <obj> {} {} ".format(dic[r1][r2], l2, boxes_bin[r2])

		caption_token_list = caption.strip().split()
		tgt_caption = ' '.join(caption_token_list[:self.max_tgt_length])

		src_item = self.encode_text(self.prompt, use_bpe=False)
		tgt_item = self.encode_text(tgt_caption, use_bpe=False)

		src_item = torch.cat([self.bos_item, src_item, self.eos_item])
		target_item = torch.cat([tgt_item, self.eos_item])
		prev_output_item = torch.cat([self.bos_item, tgt_item])

		gt_relations = np.array(
			[[rel[0], rel[1], pred_ids[i]] for i, rel in enumerate(img_rels)]
		)

		example = {
			"id": uniq_id,
			"source": src_item,
			"patch_image": patch_image,
			"patch_mask": patch_mask,
			"target": target_item,
			"prev_output_tokens": prev_output_item,
			"gt_relations": gt_relations, 
			"gt_boxes": boxes,
			"gt_classes": box_ids,
			"img_size": img_size
		}
		return example

	def collater(self, samples, pad_to_length=None):
		"""Merge a list of samples to form a mini-batch.
		Args:
			samples (List[dict]): samples to collate
		Returns:
			dict: a mini-batch containing the data of the task
		"""
		return collate(samples, pad_idx=self.pad, eos_idx=self.eos)