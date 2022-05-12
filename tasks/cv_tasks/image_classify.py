# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
import pickle
from typing import Optional
from data.file_dataset import FileDataset

import torch
from fairseq import metrics
from fairseq.tasks import register_task

from data.cv_data.image_classify_dataset import ImageClassifyDataset
from data import data_utils
from tasks.ofa_task import OFAConfig, OFATask
from utils.trie import Trie

logger = logging.getLogger(__name__)

@dataclass
class ImageClassifyConfig(OFAConfig):
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )
    ans2label_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to load ans2label file"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )


@register_task("image_classify", dataclass=ImageClassifyConfig)
class ImageClassifyTask(OFATask):
    def __init__(self, cfg: ImageClassifyConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.ans2label_dict = None
        if self.cfg.ans2label_file is not None:
            self.ans2label_dict = pickle.load(open(self.cfg.ans2label_file, "rb"))
        else:
            self.ans2label_dict = json.loads(self.cfg.ans2label_dict)
        
        self.uses_ema = self.cfg.uses_ema

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            table_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            table_path = paths[-1]
        dataset = FileDataset(table_path, self.cfg.selected_cols)

        self.datasets[split] = ImageClassifyDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            constraint_trie=self.constraint_trie,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        tgt_list = []
        prev_output_list = []
        self.index2ans = {}
        self.ans2index = {}
        self.constraint_trie = Trie(self.tgt_dict.eos())
        for i, answer in enumerate(self.ans2label_dict.keys()):
            answer_item = self.tgt_dict.encode_line(
                line=self.bpe.encode(' ' + answer),
                add_if_not_exist=False,
                append_eos=False
            ).long()
            tgt_list += [torch.cat([answer_item, torch.LongTensor([self.tgt_dict.eos()])])]
            prev_output_list += [torch.cat([torch.LongTensor([self.tgt_dict.bos()]), answer_item])]
            self.index2ans[i] = answer
            self.ans2index[answer] = i
            self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])

        constraint_mask_list = []
        for prev_output_item in prev_output_list:
            constraint_mask = torch.zeros((len(prev_output_item), len(self.tgt_dict))).bool()
            for i in range(len(prev_output_item)):
                constraint_prefix_token = prev_output_item[:i+1].tolist()
                constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                constraint_mask[i][constraint_nodes] = True
            constraint_mask_list.append(constraint_mask)

        eos = self.src_dict.eos()
        pad = self.src_dict.pad()
        self.valid_tgt_list = []
        self.valid_prev_output_list = []
        self.valid_constraint_masks_list = []
        for i in range(0, len(tgt_list), self.cfg.valid_batch_size):
            tgt_item = tgt_list[i:i+self.cfg.valid_batch_size]
            prev_output_item = prev_output_list[i:i+self.cfg.valid_batch_size]
            constrain_mask = constraint_mask_list[i:i+self.cfg.valid_batch_size]
            self.valid_tgt_list.append(
                data_utils.collate_tokens(tgt_item, pad_idx=pad, eos_idx=eos, left_pad=False)
            )
            self.valid_prev_output_list.append(
                data_utils.collate_tokens(prev_output_item, pad_idx=pad, eos_idx=eos, left_pad=False)
            )
            self.valid_constraint_masks_list.append(
                data_utils.collate_tokens(constrain_mask, pad_idx=pad, left_pad=False)
            )

        return model

    def build_generator(
        self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        seq_generator = super().build_generator(models, args, seq_gen_cls, extra_gen_cls_kwargs, prefix_allowed_tokens_fn)
        seq_generator.constraint_trie = self.constraint_trie

        return seq_generator

    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)

        if self.uses_ema:
            assert 'ema_model' in extra_kwargs and extra_kwargs['ema_model'] is not None
        if self.uses_ema:
            eval_model = extra_kwargs['ema_model']
        else:
            eval_model = model

        eval_model.eval()
        with torch.no_grad():
            batch_size = sample["net_input"]["src_tokens"].size(0)
            encoder_out = eval_model.encoder(
                sample["net_input"]["src_tokens"],
                src_lengths=sample["net_input"]["src_lengths"],
                patch_images=sample["net_input"]["patch_images"],
                patch_masks=sample["net_input"]["patch_masks"]
            )
            device = sample["net_input"]["src_tokens"].device
            valid_result = []
            for valid_tgt, valid_prev_output, valid_constraint_masks in zip(self.valid_tgt_list,
                                                                            self.valid_prev_output_list,
                                                                            self.valid_constraint_masks_list):
                valid_tgt_size = valid_tgt.size(0)
                valid_tgt = valid_tgt.repeat(batch_size, 1).to(device)
                valid_prev_output = valid_prev_output.repeat(batch_size, 1).to(device)
                valid_constraint_masks = valid_constraint_masks.repeat(batch_size, 1, 1).to(device)
                new_encoder_out = {}
                new_encoder_out["encoder_out"] = [
                    encoder_out["encoder_out"][0].repeat_interleave(valid_tgt_size, dim=1)
                ]
                new_encoder_out["encoder_padding_mask"] = [
                    encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_tgt_size, dim=0)
                ]
                new_encoder_out["position_embeddings"] = [
                    encoder_out["position_embeddings"][0].repeat_interleave(valid_tgt_size, dim=0)
                ]

                decoder_out = eval_model.decoder(valid_prev_output, encoder_out=new_encoder_out)
                decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                lprobs = eval_model.get_normalized_probs(decoder_out, log_probs=True)
                scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                scores = scores.masked_fill(valid_tgt.eq(self.tgt_dict.pad()), 0)
                scores = scores.sum(1)
                scores = scores.view(-1, valid_tgt_size)
                valid_result.append(scores)

        valid_result = torch.cat(valid_result, dim=-1)
        predicts = valid_result.argmax(1).tolist()
        hyps = [self.index2ans[predict_index] for predict_index in predicts]
        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        logging_output["_score_sum"] = sum(scores)
        logging_output["_score_cnt"] = len(scores)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_score(meters):
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 3)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)
