# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from io import BytesIO

import math
import logging
import random
import warnings

import numpy as np
import torch
import base64
from torchvision import transforms

from PIL import Image, ImageFile

from data import data_utils
from data.ofa_dataset import OFADataset
from utils.vision_helper import RandomAugment
import utils.transforms as T

ImageFile.LOAD_TRUNCATED_IMAGES = True
ImageFile.MAX_IMAGE_PIXELS = None
Image.MAX_IMAGE_PIXELS = None

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def get_whole_word_mask(bpe, dictionary):
    if bpe is not None:

        def is_beginning_of_word(i):
            if i < dictionary.nspecial:
                # special elements are always considered beginnings
                return True
            tok = dictionary[i]
            if tok.startswith("madeupword"):
                return True
            try:
                return bpe.is_beginning_of_word(tok)
            except ValueError:
                return True

        mask_whole_words = torch.ByteTensor(
            list(map(is_beginning_of_word, range(len(dictionary))))
        )
        return mask_whole_words
    return None


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

    code_masks = None
    if samples[0].get("code_mask", None) is not None:
        code_masks = torch.cat([sample['code_mask'] for sample in samples])

    conf = torch.cat([s['conf'] for s in samples], dim=0)

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

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "patch_images": patch_images,
            "patch_masks": patch_masks,
            "code_masks": code_masks,
            "prev_output_tokens": prev_output_tokens
        },
        "target": target,
        "conf": conf
    }

    return batch


class UnifyDataset(OFADataset):
    def __init__(
        self,
        split,
        dataset,
        bpe,
        src_dict,
        tgt_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        seed=7,
        code_dict_size=8192,
        num_bins=1000,
        patch_image_size=384,
        code_image_size=128,
        pure_text_dataset=None,
        pure_image_dataset=None,
        detection_dataset=None,
        all_object_list=None,
        all_caption_list=None,
        type2ans_dict=None,
        ans2type_dict=None,
        max_image_size=512,
        mask_ratio=0.3,
        random_ratio=0.0,
        keep_ratio=0.0,
        mask_length="span-poisson",
        poisson_lambda=3.0,
        replace_length=1
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.seed = seed
        self.code_dict_size = code_dict_size
        self.num_bins = num_bins
        self.patch_image_size = patch_image_size
        self.code_image_size = code_image_size

        self.pure_text_dataset = pure_text_dataset
        self.pure_image_dataset = pure_image_dataset
        self.detection_dataset = detection_dataset
        self.epoch = 0

        self.all_object_list = all_object_list
        self.all_caption_list = all_caption_list
        self.type2ans_dict = type2ans_dict
        self.ans2type_dict = ans2type_dict

        self.mask_ratio = mask_ratio
        self.random_ratio = random_ratio
        self.keep_ratio = keep_ratio
        self.mask_length = mask_length
        self.poisson_lambda = poisson_lambda
        self.replace_length = replace_length
        if self.replace_length not in [-1, 0, 1]:
            raise ValueError(f"invalid arg: replace_length={self.replace_length}")
        if self.mask_length not in ["subword", "word", "span-poisson"]:
            raise ValueError(f"invalid arg: mask-length={self.mask_length}")
        if self.mask_length == "subword" and self.replace_length not in [0, 1]:
            raise ValueError(f"if using subwords, use replace-length=1 or 0")

        self.mask_idx = src_dict.index("<mask>")
        self.mask_whole_word = (
            get_whole_word_mask(self.bpe, self.src_dict)
            if self.mask_length != "subword"
            else None
        )
        self.mask_span_distribution = None
        if self.mask_length == "span-poisson":
            _lambda = self.poisson_lambda
            lambda_to_the_k = 1
            e_to_the_minus_lambda = math.exp(-_lambda)
            k_factorial = 1
            ps = []
            for k in range(0, 128):
                ps.append(e_to_the_minus_lambda * lambda_to_the_k / k_factorial)
                lambda_to_the_k *= _lambda
                k_factorial *= k + 1
                if ps[-1] < 0.0000001:
                    break
            ps = torch.FloatTensor(ps)
            self.mask_span_distribution = torch.distributions.Categorical(ps)

        self.pos_tgt_item = self.encode_text(" yes")
        self.neg_tgt_item = self.encode_text(" no")

        self.mask_left = self.mask_top = int(0.5 * self.code_image_size)
        self.mask_right = self.mask_bottom = int(1.5 * self.code_image_size)
        self.mask_ids = [
            i*self.code_image_size*2+j
            for i in range(self.code_image_size*2) for j in range(self.code_image_size*2)
            if not (self.mask_left <= i < self.mask_right and self.mask_top <= j < self.mask_bottom)
        ]

        scales = np.arange(patch_image_size, 481).tolist()

        # for image-text pair
        self.patch_resize_transform = transforms.Compose([
            T.RandomResize(scales, max_size=672),
            transforms.CenterCrop(patch_image_size),
            RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for pure image
        self.patch_crop_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        # for detection
        self.detection_transform = T.Compose([
            T.RandomHorizontalFlip(),
            T.LargeScaleJitter(output_size=self.code_image_size*2, aug_scale_min=1.0, aug_scale_max=1.5),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])
        # for visual grounding
        self.visual_grounding_transform = T.Compose([
            T.RandomResize(scales, max_size=672),
            T.ObjectCenterCrop((patch_image_size, patch_image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_image_size=max_image_size)
        ])

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def get_negative_caption(self, caption, gt_objects):
        prob = random.random()
        if gt_objects is not None and gt_objects != '' and prob > 0.6:
            gt_object = random.choice(gt_objects.strip().split('&&'))
            negative_object = random.choice(self.all_object_list[:-1])
            negative_object = self.all_object_list[-1] if negative_object == gt_object else negative_object
            negative_caption = caption.replace(gt_object, negative_object)
        else:
            negative_caption = random.choice(self.all_caption_list)
        return negative_caption

    def get_negative_answer(self, answer, conf):
        prob = random.random()
        if conf > (prob + 0.1) and answer in self.ans2type_dict:
            negative_answer_type = self.ans2type_dict[answer]
            if negative_answer_type == 'how many' and answer.isdigit() and prob > 0.5:
                negative_answer = int(answer) + random.choice([-1, 1]) if answer != 0 else 1
            else:
                negative_answer_list = self.type2ans_dict[negative_answer_type]
                negative_answer = random.choice(negative_answer_list[:-1])
                negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
            return negative_answer

        negative_answer_list = self.type2ans_dict['other']
        negative_answer = random.choice(negative_answer_list[:-1])
        negative_answer = negative_answer_list[-1] if negative_answer == answer else negative_answer
        return negative_answer

    def process_image_text_pair(self, index):
        uniq_id, image, caption, question, refs, gt_objects, dataset_name, type = self.dataset[index]

        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        patch_image = self.patch_resize_transform(image) if type != 'visual_grounding' else None
        patch_mask = torch.tensor([True])
        conf = torch.tensor([1.0])
        if type == 'caption':
            tgt_caption = self.pre_caption(caption, self.max_tgt_length)
            pos_src_caption = self.pre_caption(caption, self.max_src_length)
            neg_src_caption = self.pre_caption(self.get_negative_caption(caption, gt_objects), self.max_src_length)
            src_item = self.encode_text(" what does the image describe?")
            tgt_item = self.encode_text(" {}".format(tgt_caption))
            pos_src_item = self.encode_text(' does the image describe " {} "?'.format(pos_src_caption))
            neg_src_item = self.encode_text(' does the image describe " {} "?'.format(neg_src_caption))
        elif type == 'qa':
            question = self.pre_question(question, self.max_src_length)
            ref_dict = {item.split('|!+')[1]: float(item.split('|!+')[0]) for item in refs.split('&&')}
            answer = max(ref_dict, key=ref_dict.get)
            conf = ref_dict[answer]
            src_item = self.encode_text(" {}".format(question))
            tgt_item = self.encode_text(" {}".format(answer))
            conf = torch.tensor([conf])
            pos_src_item = self.encode_text(' what is the answer to question " {} ". is " {} "?'.format(question, answer))
            neg_src_item = self.encode_text(
                ' what is the answer to question " {} ". is " {} "?'.format(question, self.get_negative_answer(answer, conf))
            )
        elif type == 'visual_grounding':
            conf = torch.tensor([1.0])
            w, h = image.size
            boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
            x0, y0, x1, y1 = refs.strip().split(',')
            boxes_target["boxes"] = torch.tensor([[float(x0), float(y0), float(x1), float(y1)]])
            boxes_target["labels"] = np.array([0])
            boxes_target["area"] = torch.tensor([(float(x1) - float(x0)) * (float(y1) - float(y0))])
            patch_image, boxes_target = self.visual_grounding_transform(image, boxes_target)
            quant_x0 = "<bin_{}>".format(int((boxes_target["boxes"][0][0] * (self.num_bins - 1)).round()))
            quant_y0 = "<bin_{}>".format(int((boxes_target["boxes"][0][1] * (self.num_bins - 1)).round()))
            quant_x1 = "<bin_{}>".format(int((boxes_target["boxes"][0][2] * (self.num_bins - 1)).round()))
            quant_y1 = "<bin_{}>".format(int((boxes_target["boxes"][0][3] * (self.num_bins - 1)).round()))
            region_coord = "{} {} {} {}".format(quant_x0, quant_y0, quant_x1, quant_y1)
            src_caption = self.pre_caption(caption, self.max_src_length)
            src_item = self.encode_text(' which region does the text " {} " describe?'.format(src_caption))
            tgt_item = self.encode_text(region_coord, use_bpe=False)
        else:
            logger.info('type {} is not implemented'.format(type))
            raise NotImplementedError

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])
        pos_src_item = torch.cat([self.bos_item, pos_src_item, self.eos_item]) if type != 'visual_grounding' else None
        neg_src_item = torch.cat([self.bos_item, neg_src_item, self.eos_item]) if type != 'visual_grounding' else None

        if type == 'caption' and dataset_name == 'cc12m':
            target_item[:2] = self.src_dict.pad()
            target_item[-1] = self.eos_item

        example = {
            "id": uniq_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }

        examples = [example]
        prob = random.random()
        if type == 'visual_grounding':
            region_example = example.copy()
            region_prefix_item = self.encode_text('  what does the region describe? region:')
            region_coord_item = self.encode_text('{}'.format(region_coord), use_bpe=False)
            region_src_item = torch.cat([region_prefix_item, region_coord_item])
            region_tgt_item = self.encode_text(' {}'.format(self.pre_caption(caption, self.max_tgt_length)))
            region_example["source"] = torch.cat([self.bos_item, region_src_item, self.eos_item])
            region_example["target"] = torch.cat([region_tgt_item, self.eos_item])
            region_example["prev_output_tokens"] = torch.cat([self.bos_item, region_tgt_item])
            region_example["conf"] = torch.tensor([1.0])
            examples.append(region_example)
        elif prob >= 0.5 and self.split == 'train':
            pos_example = example.copy()
            pos_example["source"] = pos_src_item
            pos_example["target"] = torch.cat([self.pos_tgt_item, self.eos_item])
            pos_example["prev_output_tokens"] = torch.cat([self.bos_item, self.pos_tgt_item])
            examples.append(pos_example)
        elif self.split == 'train':
            neg_example = example.copy()
            neg_example["source"] = neg_src_item
            neg_example["target"] = torch.cat([self.neg_tgt_item, self.eos_item])
            neg_example["prev_output_tokens"] = torch.cat([self.bos_item, self.neg_tgt_item])
            examples.append(neg_example)
        return examples

    def process_pure_text(self, index):
        patch_image = torch.zeros((3, self.code_image_size*2, self.code_image_size*2))
        patch_mask = torch.tensor([False])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        examples = []
        for _ in range(2):
            uniq_id, text = self.pure_text_dataset[index]
            text = text.strip().lower()
            text_item = self.encode_text(" {}".format(text), length=512)
            text_item = text_item[-256:]
            text_item = torch.cat([self.bos_item, text_item, self.eos_item])
            mask_text_item = self.add_whole_word_mask(text_item.clone(), self.mask_ratio)
            prefix_item = self.encode_text(' what is the complete text of " "?')
            src_item = torch.cat([prefix_item[:-2], mask_text_item[1:-1], prefix_item[-2:]])
            tgt_item = text_item[1:-1]
            src_item = torch.cat([self.bos_item, src_item, self.eos_item])
            target_item = torch.cat([tgt_item, self.eos_item])
            prev_output_item = torch.cat([self.bos_item, tgt_item])
            example = {
                "id": uniq_id,
                "source": src_item,
                "patch_image": patch_image,
                "patch_mask": patch_mask,
                "code_mask": code_mask,
                "target": target_item,
                "prev_output_tokens": prev_output_item,
                "conf": conf,
            }
            examples.append(example)

        return examples

    def process_pure_image(self, index):
        image_id, image, code = self.pure_image_dataset[index]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")
        patch_image = self.patch_crop_transform(image)
        patch_image[:, self.mask_top:self.mask_bottom, self.mask_left:self.mask_right] = 0
        patch_mask = torch.tensor([True])
        src_item = self.encode_text(" what is the image in the middle part?")
        image_code = torch.LongTensor([int(num) for num in code.strip().split()])
        tgt_item = image_code + len(self.src_dict) - self.code_dict_size - self.num_bins
        code_mask = torch.tensor([True])
        conf = torch.tensor([2.0])

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def process_detection(self, index):
        image_id, image, label = self.detection_dataset[index]
        image = Image.open(BytesIO(base64.urlsafe_b64decode(image))).convert("RGB")

        w, h = image.size
        boxes_target = {"boxes": [], "labels": [], "area": [], "size": torch.tensor([h, w])}
        label_list = label.strip().split('&&')
        for label in label_list:
            x0, y0, x1, y1, cat_id, cat = label.strip().split(',', 5)
            boxes_target["boxes"].append([float(x0), float(y0), float(x1), float(y1)])
            boxes_target["labels"].append(cat)
            boxes_target["area"].append((float(x1) - float(x0)) * (float(y1) - float(y0)))
        boxes_target["boxes"] = torch.tensor(boxes_target["boxes"])
        boxes_target["labels"] = np.array(boxes_target["labels"])
        boxes_target["area"] = torch.tensor(boxes_target["area"])

        patch_image, boxes_target = self.detection_transform(image, boxes_target)
        patch_mask = torch.tensor([True])
        code_mask = torch.tensor([False])
        conf = torch.tensor([2.0])

        quant_boxes = []
        for i, box in enumerate(boxes_target["boxes"]):
            quant_boxes.extend(["<bin_{}>".format(int((pos * (self.num_bins - 1)).round())) for pos in box[:4]])
            quant_boxes.append(self.bpe.encode(' {}'.format(boxes_target["labels"][i])))
        src_item = self.encode_text(' what are the objects in the image?')
        tgt_item = self.encode_text(' '.join(quant_boxes), use_bpe=False)

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        example = {
            "id": image_id,
            "source": src_item,
            "patch_image": patch_image,
            "patch_mask": patch_mask,
            "code_mask": code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def __getitem__(self, index):
        with data_utils.numpy_seed(self.seed, self.epoch):
            pair_samples = self.process_image_text_pair(index)
            extra_samples = []
            if self.split == 'train' and self.dataset.data_cnt % 8 == 0:
                extra_samples += self.process_pure_text(0) if self.pure_text_dataset else []
                extra_samples += self.process_pure_image(0) if self.pure_image_dataset else []
                extra_samples += self.process_detection(0) if self.detection_dataset else []
        return pair_samples, extra_samples

    def word_starts(self, source):
        if self.mask_whole_word is not None:
            is_word_start = self.mask_whole_word.gather(0, source)
        else:
            is_word_start = torch.ones(source.size())
        is_word_start[0] = 0
        is_word_start[-1] = 0
        return is_word_start

    def add_whole_word_mask(self, source, p):
        is_word_start = self.word_starts(source)
        num_to_mask = int(math.ceil(is_word_start.float().sum() * p))
        num_inserts = 0
        if num_to_mask == 0:
            return source

        if self.mask_span_distribution is not None:
            lengths = self.mask_span_distribution.sample(sample_shape=(num_to_mask,))

            # Make sure we have enough to mask
            cum_length = torch.cumsum(lengths, 0)
            while cum_length[-1] < num_to_mask:
                lengths = torch.cat(
                    [
                        lengths,
                        self.mask_span_distribution.sample(sample_shape=(num_to_mask,)),
                    ],
                    dim=0,
                )
                cum_length = torch.cumsum(lengths, 0)

            # Trim to masking budget
            i = 0
            while cum_length[i] < num_to_mask:
                i += 1
            lengths[i] = num_to_mask - (0 if i == 0 else cum_length[i - 1])
            num_to_mask = i + 1
            lengths = lengths[:num_to_mask]

            # Handle 0-length mask (inserts) separately
            lengths = lengths[lengths > 0]
            num_inserts = num_to_mask - lengths.size(0)
            num_to_mask -= num_inserts
            if num_to_mask == 0:
                return self.add_insertion_noise(source, num_inserts / source.size(0))

            assert (lengths > 0).all()
        else:
            lengths = torch.ones((num_to_mask,)).long()
        assert is_word_start[-1] == 0
        word_starts = is_word_start.nonzero(as_tuple=False)
        indices = word_starts[
            torch.randperm(word_starts.size(0))[:num_to_mask]
        ].squeeze(1)
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < self.random_ratio

        source_length = source.size(0)
        assert source_length - 1 not in indices
        to_keep = torch.ones(source_length, dtype=torch.bool)
        is_word_start[
            -1
        ] = 255  # acts as a long length, so spans don't go over the end of doc
        if self.replace_length == 0:
            to_keep[indices] = 0
        else:
            # keep index, but replace it with [MASK]
            source[indices] = self.mask_idx
            source[indices[mask_random]] = torch.randint(
                4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
            )

        if self.mask_span_distribution is not None:
            assert len(lengths.size()) == 1
            assert lengths.size() == indices.size()
            lengths -= 1
            while indices.size(0) > 0:
                assert lengths.size() == indices.size()
                lengths -= is_word_start[indices + 1].long()
                uncompleted = lengths >= 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                lengths = lengths[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                    )
        else:
            # A bit faster when all lengths are 1
            while indices.size(0) > 0:
                uncompleted = is_word_start[indices + 1] == 0
                indices = indices[uncompleted] + 1
                mask_random = mask_random[uncompleted]
                if self.replace_length != -1:
                    # delete token
                    to_keep[indices] = 0
                else:
                    # keep index, but replace it with [MASK]
                    source[indices] = self.mask_idx
                    source[indices[mask_random]] = torch.randint(
                        4, len(self.tgt_dict) - self.code_dict_size - self.num_bins, size=(mask_random.sum(),)
                    )

                assert source_length - 1 not in indices

        source = source[to_keep]

        if num_inserts > 0:
            source = self.add_insertion_noise(source, num_inserts / source.size(0))

        return source

    def add_insertion_noise(self, tokens, p):
        if p == 0.0:
            return tokens

        num_tokens = len(tokens)
        n = int(math.ceil(num_tokens * p))

        noise_indices = torch.randperm(num_tokens + n - 2)[:n] + 1
        noise_mask = torch.zeros(size=(num_tokens + n,), dtype=torch.bool)
        noise_mask[noise_indices] = 1
        result = torch.LongTensor(n + len(tokens)).fill_(-1)

        num_random = int(math.ceil(n * self.random_ratio))
        result[noise_indices[num_random:]] = self.mask_idx
        result[noise_indices[:num_random]] = torch.randint(
            low=4, high=len(self.tgt_dict)-self.code_dict_size-self.num_bins, size=(num_random,)
        )

        result[~noise_mask] = tokens

        assert (result >= 0).all()
        return result

    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing image-text pairs
        samples_v2 = []   # containing detection data, text data and image data
        for sample_tuple in samples:
            samples_v1 += sample_tuple[0]
            samples_v2 += sample_tuple[1]
        if samples_v2 != []:
            res_v1 = collate(samples_v1, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
            res_v2 = collate(samples_v2, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
            return res_v1, res_v2
        else:
            res_v1 = collate(samples_v1, pad_idx=self.src_dict.pad(), eos_idx=self.eos)
            return res_v1
