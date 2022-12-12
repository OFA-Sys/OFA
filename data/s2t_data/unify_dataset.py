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

from fairseq.data.audio.speech_to_text_dataset import S2TDataConfig
from fairseq.data.audio.feature_transforms import *
from fairseq.data.audio.audio_utils import (
    convert_waveform, _get_kaldi_fbank, _get_torchaudio_fbank
)
from pathlib import Path
import soundfile as sf
import librosa
import torchaudio
from typing import List

from pypinyin import pinyin, Style
from utils.text2phone import Text2Phone
from g2p_en import G2p
g2p = G2p()

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=False,
    left_pad_target=False,
    feature_only = True,
    mask = False,
    mask_prob = 0.0
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad,
            move_eos_to_beginning,
        )

    def _collate_frames(
        frames: List[torch.Tensor]
    ):
        """
        Convert a list of 2D frames into a padded 3D tensor
        Args:
            frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
                length of i-th frame and f_dim is static dimension of features
        Returns:
            3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
        """
        max_len = max(frame.size(0) for frame in frames)
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1)))
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v
        return out

    def _collate_constraint_masks(
        frames: List[torch.Tensor]
    ):
        """
        Convert a list of 2D frames into a padded 3D tensor
        Args:
            frames (list): list of 2D frames of size L[i]*f_dim. Where L[i] is
                length of i-th frame and f_dim is static dimension of features
        Returns:
            3D tensor of size len(frames)*len_max*f_dim where len_max is max of L[i]
        """
        max_len = max(frame.size(0) for frame in frames)
        out = frames[0].new_zeros((len(frames), max_len, frames[0].size(1))).bool()
        for i, v in enumerate(frames):
            out[i, : v.size(0)] = v
        return out

    id = np.array([s["id"] for s in samples])
    src_tokens = merge("source", left_pad=left_pad_source)
    src_lengths = torch.LongTensor([s["source"].ne(pad_idx).long().sum() for s in samples])

    fbank = None
    fbank_length = None
    fbank_masks = None
    if samples[0].get("fbank", None) is not None:
        fbank = _collate_frames([s["fbank"] for s in samples])
        fbank_length = torch.tensor([s["fbank"].size(0) for s in samples], dtype=torch.long)
        fbank_masks = torch.tensor([s["fbank_mask"] for s in samples])

    audio_code_masks = None
    if samples[0].get("audio_code_mask", None) is not None:
        audio_code_masks = torch.cat([sample['audio_code_mask'] for sample in samples])

    phone_items = None
    phone_lengths = None
    if samples[0].get("phone_item", None) is not None:
        phone_items = merge("phone_item", left_pad=left_pad_source)
        phone_lengths = torch.LongTensor([len(s["phone_item"]) for s in samples])
    phone_masks = None
    if samples[0].get("phone_mask", None) is not None:
        phone_masks = torch.cat([sample['phone_mask'] for sample in samples])

    prev_output_tokens = None
    target = None
    if samples[0].get("target", None) is not None:
        target = merge("target", left_pad=left_pad_target)
        tgt_lengths = torch.LongTensor([s["target"].ne(pad_idx).long().sum() for s in samples])
        ntokens = tgt_lengths.sum().item()

        if samples[0].get("prev_output_tokens", None) is not None:
            prev_output_tokens = merge("prev_output_tokens", left_pad=left_pad_target)
    else:
        ntokens = src_lengths.sum().item()

    constraint_masks = None
    if samples[0].get("constraint_masks", None) is not None:
        constraint_masks = _collate_constraint_masks([s["constraint_masks"] for s in samples])

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "net_input": {
            "src_tokens": src_tokens,
            "src_lengths": src_lengths,
            "fbank": fbank,
            "fbank_length": fbank_length,
            "fbank_masks": fbank_masks,
            "phone_items": phone_items,
            "phone_lengths": phone_lengths,
            "phone_masks": phone_masks,
            "audio_code_masks": audio_code_masks,
            "prev_output_tokens": prev_output_tokens,
            "encoder_features_only": feature_only,
            "mask": mask,
            "mask_prob": mask_prob
        },
        "target": target,
        "ctc_outputs": phone_items,
        "ctc_output_lengths": phone_lengths,
        "constraint_masks": constraint_masks
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
        phone_dict=None,
        max_src_length=128,
        max_tgt_length=30,
        seed=7,
        code_dict_size=8192,
        audio_code_dict_size=30000,
        num_bins=1000,
        pure_text_dataset=None,
        pure_audio_dataset=None,
        speech_text_dataset=None,
        config_yaml_path=None,
        lang="zh",
        text2phone_path=None,
        train_stage=2,
        n_frames_per_step=1,
        sample_rate=16000,
    ):
        super().__init__(split, dataset, bpe, src_dict, tgt_dict)
        self.phone_dict = phone_dict
        self.max_src_length = max_src_length
        self.max_tgt_length = max_tgt_length
        self.seed = seed
        self.code_dict_size = code_dict_size
        self.audio_code_dict_size = audio_code_dict_size
        self.num_bins = num_bins

        self.pure_text_dataset = pure_text_dataset
        self.pure_audio_dataset = pure_audio_dataset
        self.speech_text_dataset = speech_text_dataset
        self.epoch = 0
        self.remove_pure_audio = self.pure_audio_dataset is None
        self.remove_pure_text = self.pure_text_dataset is None
    
        # config_yaml_path = Path(cfg.user_dir) / cfg.config_yaml)
        self.data_cfg = S2TDataConfig(Path(config_yaml_path))
        self.lang = lang
        self.train_stage= train_stage
        self.feature_transforms = CompositeAudioFeatureTransform.from_config_dict(
            self.data_cfg.get_feature_transforms(split, split.startswith("train"))
        )
        self.n_frames_per_step = n_frames_per_step
        self.sample_rate = sample_rate
        self.blank_id = self.phone_dict.index("<blank>")
        self.phone_mask_idx = self.phone_dict.index("<mask>")
        self.text2phone_tokenizer = None
        if text2phone_path is not None:
            self.blank_id = self.phone_dict.index("<unk>")
            self.text2phone_tokenizer = Text2Phone(text2phone_path)

    def set_epoch(self, epoch, **unused):
        self.epoch = epoch

    def process_pure_text(self, index):
        if self.train_stage == 1:
            speech_id, text = self.dataset[index]
        else:
            speech_id, text = self.pure_text_dataset[index]
        conf = torch.tensor([1.0])

        # fake input
        fbank = torch.zeros((8, self.data_cfg.input_feat_per_channel))
        fbank_mask = torch.tensor([False])
        #
        audio_code_mask = torch.tensor([False])

        if self.lang == "en":
            text = self.pre_caption(text, self.max_tgt_length)
        elif self.lang == "zh":
            text = self.pre_chinese(text, self.max_tgt_length)
        else:
            raise ValueError("lang must be en or zh")

        phone = self.to_phone(text, self.lang)
        phone_item = [int(x) for x in phone]
        phone_item = torch.tensor(phone_item)
        phone_item = self.add_noise_to_phone(phone_item, 0.3)

        phone_mask = torch.tensor([True])

        target = text

        src_item = self.encode_text(" what does the phone say?")
        tgt_item = self.encode_text(" {}".format(target))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        constraint_masks = torch.stack([torch.arange(len(self.tgt_dict)) < len(
            self.tgt_dict) - self.audio_code_dict_size - self.code_dict_size - self.num_bins for _ in
                                        range(len(target_item))])

        example = {
            "id": speech_id,
            "source": src_item,
            "fbank": fbank,
            "fbank_mask": fbank_mask,
            "phone_item": phone_item,
            "phone_mask": phone_mask,
            "audio_code_mask": audio_code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
            "constraint_masks": constraint_masks
        }
        return [example]

    def process_pure_audio(self, index):
        if self.train_stage == 2:
            speech_id, wav_data, code = self.dataset[index]
        else:
            speech_id, wav_data, code = self.pure_audio_dataset[index]

        # fake input
        phone_item = [6, 6, 6]
        phone_item = torch.tensor(phone_item)
        phone_mask = torch.tensor([False])

        # speed
        if self.split == "train":
            speed = random.choice([0.9, 1.0, 1.1])
        else:
            speed = 1.0
        wav, sr = sf.read(wav_data)
        # spec_augmentation
        fbank = self.prepare_fbank(torch.tensor([wav], dtype=torch.float32), sr, speed)

        fbank_mask = torch.tensor([True])
        audio_code_mask = torch.tensor([True])

        if code is not None and len(code) > 0:
            text = torch.LongTensor([int(num) for num in code.strip().split(",")])
            tgt_item = text + len(self.tgt_dict) - self.audio_code_dict_size - self.code_dict_size - self.num_bins
        else:
            # fake
            text = torch.LongTensor([1, 2, 3])
            tgt_item = text

        conf = torch.tensor([1.0])

        # useless
        src_item = self.encode_text(' what does the audio say?')

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        constraint_masks = torch.stack([torch.arange(len(self.tgt_dict)) >= len(
            self.tgt_dict) - self.audio_code_dict_size - self.code_dict_size - self.num_bins for _ in
                                        range(len(target_item))])
        constraint_masks[:, :3] = True

        example = {
            "id": speech_id,
            "source": src_item,
            "fbank": fbank,
            "fbank_mask": fbank_mask,
            "phone_item": phone_item,
            "phone_mask": phone_mask,
            "audio_code_mask": audio_code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
        }
        return [example]

    def process_speech_text_pair(self, index, dataset=None):
        if dataset is not None:
            speech_id, wav_data, text = dataset[index]
        elif self.train_stage == 2:
            speech_id, wav_data, text = self.speech_text_dataset[index]
        else:
            speech_id, wav_data, text = self.dataset[index]
       
        conf = torch.tensor([1.0])
        audio_code_mask = torch.tensor([False])

        # speed
        if self.split == "train":
            speed = random.choice([0.9, 1.0, 1.1])
        else:
            speed = 1.0
        # wav, sr = sf.read(wav_data)
        wav, sr = librosa.load(wav_data, self.sample_rate)
        # spec_augmentation
        fbank = self.prepare_fbank(torch.tensor([wav], dtype=torch.float32), sr, speed, speech_id)

        fbank_mask = torch.tensor([True])

        if self.lang == "en":
            text = self.pre_caption(text, self.max_tgt_length)
        elif self.lang == "zh":
            text = self.pre_chinese(text, self.max_tgt_length)
        else:
            raise ValueError("lang must be en or zh")
        target = text

        phone_item = self.to_phone(text, self.lang)-3
        phone_mask = torch.tensor([False])

        src_item = self.encode_text(" what does the audio say?")
        tgt_item = self.encode_text(" {}".format(target))

        src_item = torch.cat([self.bos_item, src_item, self.eos_item])
        target_item = torch.cat([tgt_item, self.eos_item])
        prev_output_item = torch.cat([self.bos_item, tgt_item])

        constraint_masks = torch.stack([torch.arange(len(self.tgt_dict)) < len(
            self.tgt_dict) - self.audio_code_dict_size - self.code_dict_size - self.num_bins for _ in
                                        range(len(target_item))])

        example = {
            "id": speech_id,
            "source": src_item,
            "fbank": fbank,
            "fbank_mask": fbank_mask,
            "phone_item": phone_item,
            "phone_mask": phone_mask,
            "audio_code_mask": audio_code_mask,
            "target": target_item,
            "prev_output_tokens": prev_output_item,
            "conf": conf,
            "constraint_masks": constraint_masks
        }
        return [example]

    def __getitem__(self, index):

        with data_utils.numpy_seed(self.seed, self.epoch):
            if self.train_stage == 1:
                extra_samples = []
                if self.dataset is not None:
                    extra_samples += self.process_pure_text(index) if not self.remove_pure_text else []
                return extra_samples, [], []
            elif self.train_stage == 2:
                pair_examples = []
                audio_examples = []
                extra_samples = []
                if self.split == 'train':
                    if self.dataset is not None:
                        audio_examples += self.process_pure_audio(index) if not self.remove_pure_audio else []
                    if self.speech_text_dataset is not None and self.dataset.data_cnt % 4 == 0:
                        pair_examples += self.process_speech_text_pair(index)
                    if self.pure_text_dataset is not None and self.dataset.data_cnt % 2 == 0:
                        extra_samples += self.process_pure_text(index) if not self.remove_pure_text else []
                else:
                    if self.dataset is not None:
                        pair_examples += self.process_speech_text_pair(index, self.dataset)
                return pair_examples, extra_samples, audio_examples
            else:
                pair_examples = []
                extra_samples = []
                if self.split == 'train':
                    if self.dataset is not None:
                        pair_examples += self.process_speech_text_pair(index)
                    if self.pure_text_dataset is not None and self.dataset.data_cnt % 2 == 0:
                        extra_samples += self.process_pure_text(index) if not self.remove_pure_text else []
                else:
                    if self.dataset is not None:
                        pair_examples += self.process_speech_text_pair(index, self.dataset)
                return pair_examples, extra_samples, []

    def to_phone(self, text, lang):

        if lang == "en":
            phone_result = None
            try:
                phone_result = " ".join(p for p in g2p(text))
            except Exception as e:
                print(e, text)
            return self.encode_phone(phone_result)

        elif lang == "zh":
            if self.text2phone_tokenizer is not None:
                final_phone = self.text2phone_tokenizer.trans(text)
                return self.encode_phone(final_phone)
            else:
                shengmu = pinyin(text, style=Style.INITIALS, strict=False)
                yunmu = pinyin(text, style=Style.FINALS_TONE3, strict=False)
                assert len(shengmu) == len(yunmu)
                final_phone = []
                for s, y in zip(shengmu, yunmu):
                    if s[0] == y[0] or s[0] == "":
                        final_phone.append(y[0])
                    else:
                        final_phone.append(s[0] + " " + y[0])
                return self.encode_phone(" ".join(final_phone))

    def encode_phone(self, phone_item):
        tokens = self.phone_dict.encode_line(
            line=phone_item, add_if_not_exist=False, append_eos=False).long()
        return tokens

    def add_noise_to_phone(self, phone, p, random_p=0.1):
        num_to_mask = int(math.ceil(phone.size(0) * p))
        indices = torch.randperm(phone.size(0))[:num_to_mask]
        mask_random = torch.FloatTensor(num_to_mask).uniform_() < random_p
        phone[indices] = self.phone_mask_idx
        if mask_random.sum() > 0:
            phone[indices[mask_random]] = torch.randint(
                4, self.phone_mask_idx, size=(mask_random.sum(),)
            )
        return phone

    def prepare_fbank(self, waveform, sample_rate, speed, speech_id=None):
        waveform, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speed)], ['rate', str(sample_rate)]])
        _waveform, _ = convert_waveform(waveform, sample_rate, to_mono=True, normalize_volume=True)
        # Kaldi compliance: 16-bit signed integers
        _waveform = _waveform * (2 ** 15)
        _waveform = _waveform.numpy()
        fbank = _get_kaldi_fbank(_waveform, sample_rate, 80)
        if fbank is None:
            fbank = _get_torchaudio_fbank(_waveform, sample_rate, 80)
        if fbank is None:
            raise ImportError(
                "Please install pyKaldi or torchaudio to enable fbank feature extraction"
            )
        if self.feature_transforms is not None:
            fbank = self.feature_transforms(fbank)
        fbank = torch.from_numpy(fbank).float()
        fbank = self.pack_frames(fbank)
        return fbank

    def pack_frames(self, feature: torch.Tensor):
        if self.n_frames_per_step == 1:
            return feature
        n_packed_frames = feature.shape[0] // self.n_frames_per_step
        feature = feature[: self.n_frames_per_step * n_packed_frames]
        return feature.reshape(n_packed_frames, -1)

    def collater(self, samples, pad_to_length=None):
        """Merge samples of different tasks to form two mini-batches.
        Args:
            samples (List[Tuple]): samples to collate
        Returns:
            Tuple[dict]: two mini-batch containing the data of different tasks
        """

        samples_v1 = []   # containing phone-text pairs at stage-1, containing speech-text pairs at stage-2
        samples_v2 = []   # containing phone-text pairs
        samples_v3 = []   # containing pure_audio_pairs
        for sample_tuple in samples:
            samples_v1 += sample_tuple[0]
            samples_v2 += sample_tuple[1]
            if len(sample_tuple) > 2:
                samples_v3 += sample_tuple[2]

        if samples_v1 == []:
            if self.train_stage == 1:
                samples_v1 += self.process_pure_text(0)
            else:
                samples_v1 += self.process_speech_text_pair(0)

        mask = False
        mask_prob = None
        if self.split == "train" and self.train_stage != 1:
            mask = True
            mask_prob = 0.3

        res_v1 = collate(
            samples_v1,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            feature_only=True,
            mask=mask,
            mask_prob=mask_prob
        )

        if self.split == 'train' and self.train_stage != 1:
            if samples_v2 == []:
                if self.pure_text_dataset is not None:
                    samples_v2 += self.process_pure_text(0) if not self.remove_pure_text else []
            res_v2 = collate(
                samples_v2,
                pad_idx=self.src_dict.pad(),
                eos_idx=self.eos
            )
            if samples_v3 == []:
                if self.pure_audio_dataset is not None:
                    samples_v3 += self.process_pure_audio(0) if not self.remove_pure_audio else []
                else:
                    return res_v1, res_v2
            res_v3 = collate(
                samples_v3,
                pad_idx=self.src_dict.pad(),
                eos_idx=self.eos,
                feature_only=False,
                mask=True
            )
            return res_v1, res_v2, res_v3
        else:
            return res_v1
