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
from argparse import Namespace
from data.file_dataset import FileDataset

import torch
from fairseq import metrics
from fairseq.tasks import register_task

from models import search
from data.mm_data.vqa_gen_dataset import VqaGenDataset
from data import data_utils
from tasks.ofa_task import OFAConfig, OFATask
from utils.trie import Trie

logger = logging.getLogger(__name__)


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.bos, generator.eos}


def decode_fn(x, tgt_dict, bpe, generator, tokenizer=None):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    if tokenizer is not None:
        x = tokenizer.decode(x)
    return x


@dataclass
class VqaGenConfig(OFAConfig):
    max_object_length: int = field(
        default=30, metadata={"help": "the maximum object sequence length"}
    )    
    ans2label_dict: Optional[str] = field(
        default='{"no": 0, "yes":1}',
        metadata={"help": 'answer to label dict'},
    )
    ans2label_file: Optional[str] = field(
        default=None,
        metadata={"help": "path to load ans2label file"},
    )
    unconstrained_training: bool = field(
        default=False,
        metadata={"help": "do not use Trie to constrain loss into the closed candidate answer set, default to False. \
            If set to True, then open-ended training is facilitated, ans2label_file and ans2label_dict will not be used \
                and inference type must be beamsearch"},
    )
    add_object: bool = field(
        default=False,
        metadata={"help": "add object to encoder"},
    )
    valid_batch_size: int = field(
        default=20,
        metadata={"help": "valid batch size per step"},
    )
    prompt_type: Optional[str] = field(
        default=None,
        metadata={"help": "prompt_type"},
    )
    uses_ema: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to use ema"},
    )
    val_inference_type: Optional[str] = field(
        default='allcand',
        metadata={"help": "inference type in validation (allcand or beamsearch), default to allcand"},
    )    
    eval_args: Optional[str] = field(
        default='{"beam":5,"unnormalized":true,"temperature":1.0}',
        metadata={
            "help": 'generation args as JSON string for inference, only activated when --val-inference-type=beamsearch'
        },
    )    


@register_task("vqa_gen", dataclass=VqaGenConfig)
class VqaGenTask(OFATask):
    def __init__(self, cfg: VqaGenConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        if not self.cfg.unconstrained_training:
            self.ans2label_dict = None
            if self.cfg.ans2label_file is not None:
                self.ans2label_dict = pickle.load(open(self.cfg.ans2label_file, "rb"))
            else:
                self.ans2label_dict = json.loads(self.cfg.ans2label_dict)

        self.uses_ema = self.cfg.uses_ema

        assert self.cfg.val_inference_type in ["allcand", "beamsearch"], \
            "Unknown inference type encountered: {}, should be allcand or beamsearch.".format(self.cfg.val_inference_type)
        assert not (self.cfg.unconstrained_training and self.cfg.val_inference_type != "beamsearch"), \
            "For open-ended training, there is no fixed candidate answer set, then inference type must be beamsearch"

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            table_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            table_path = paths[-1]
        dataset = FileDataset(table_path, self.cfg.selected_cols)

        self.datasets[split] = VqaGenDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_object_length=self.cfg.max_object_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            add_object=self.cfg.add_object,
            constraint_trie=self.constraint_trie,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            prompt_type=self.cfg.prompt_type
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        # for open-ended training without fixed candidate answer set
        if self.cfg.unconstrained_training:
            self.constraint_trie = None
        # (default) for trie-based constraint training with fixed candidate answer set
        # (provided by ans2label_file or ans2label_dict)
        else:
            answer_item_list = []
            self.index2ans = {}
            self.constraint_trie = Trie(self.tgt_dict.eos())
            for i, answer in enumerate(self.ans2label_dict.keys()):
                answer_item = self.tgt_dict.encode_line(
                    line=self.bpe.encode(' ' + answer),
                    add_if_not_exist=False,
                    append_eos=False
                ).long()
                answer_item_list.append(answer_item)
                self.index2ans[i] = answer
                self.constraint_trie.insert([self.tgt_dict.bos()] + answer_item.tolist() + [self.tgt_dict.eos()])

            constraint_mask_list = []
            for answer_item in answer_item_list:
                constraint_mask = torch.zeros((len(answer_item)+1, len(self.tgt_dict))).bool()
                for i in range(len(answer_item)+1):
                    constraint_prefix_token = [self.src_dict.bos()] + answer_item[:i].tolist()
                    constraint_nodes = self.constraint_trie.get_next_layer(constraint_prefix_token)
                    constraint_mask[i][constraint_nodes] = True
                constraint_mask_list.append(constraint_mask)

        if self.cfg.val_inference_type == "allcand":
            assert not self.cfg.unconstrained_training
            self.valid_answers_list = []
            self.valid_constraint_masks_list = []
            for i in range(0, len(answer_item_list), self.cfg.valid_batch_size):
                self.valid_answers_list += [answer_item_list[i:i+self.cfg.valid_batch_size]]
                self.valid_constraint_masks_list += [constraint_mask_list[i:i+self.cfg.valid_batch_size]]
        elif self.cfg.val_inference_type == "beamsearch":
            gen_args = json.loads(self.cfg.eval_args)
            self.generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        else:
            raise NotImplementedError("Error: Unknown inference type encountered.")

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
            if self.cfg.val_inference_type == "allcand":
                encoder_out = eval_model.encoder(
                    sample["net_input"]["src_tokens"],
                    src_lengths=sample["net_input"]["src_lengths"],
                    patch_images=sample["net_input"]["patch_images"],
                    patch_masks=sample["net_input"]["patch_masks"]
                )
                device = sample["net_input"]["src_tokens"].device
                eos_item = torch.tensor([self.src_dict.eos()])
                pad = self.src_dict.pad()
                valid_result = []
                for valid_answers, valid_constraint_masks in zip(self.valid_answers_list, self.valid_constraint_masks_list):
                    valid_size = len(valid_answers)
                    valid_tgt_items = [
                        torch.cat([torch.tensor(decoder_prompt[1:]), valid_answer, eos_item])
                        for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
                    ]
                    valid_prev_items = [
                        torch.cat([torch.tensor(decoder_prompt), valid_answer])
                        for decoder_prompt in sample["decoder_prompts"] for valid_answer in valid_answers
                    ]
                    valid_constraint_mask_items = [
                        torch.cat([torch.zeros(len(decoder_prompt)-1, valid_constraint_mask.size(1)).bool(), valid_constraint_mask], dim=0)
                        for decoder_prompt in sample["decoder_prompts"] for valid_constraint_mask in valid_constraint_masks
                    ]
                    valid_tgt = data_utils.collate_tokens(valid_tgt_items, pad_idx=pad, left_pad=False).to(device)
                    valid_prev_output = data_utils.collate_tokens(valid_prev_items, pad_idx=pad, left_pad=False).to(device)
                    valid_constraint_masks = data_utils.collate_tokens(valid_constraint_mask_items, pad_idx=pad, left_pad=False).to(device)

                    new_encoder_out = {}
                    new_encoder_out["encoder_out"] = [
                        encoder_out["encoder_out"][0].repeat_interleave(valid_size, dim=1)
                    ]
                    new_encoder_out["encoder_padding_mask"] = [
                        encoder_out["encoder_padding_mask"][0].repeat_interleave(valid_size, dim=0)
                    ]
                    new_encoder_out["position_embeddings"] = [
                        encoder_out["position_embeddings"][0].repeat_interleave(valid_size, dim=0)
                    ]

                    decoder_out = eval_model.decoder(valid_prev_output, encoder_out=new_encoder_out)
                    decoder_out[0].masked_fill_(~valid_constraint_masks, -math.inf)
                    lprobs = eval_model.get_normalized_probs(decoder_out, log_probs=True)
                    scores = lprobs.gather(dim=-1, index=valid_tgt.unsqueeze(-1)).squeeze(-1)
                    scores = scores.masked_fill(valid_tgt.eq(self.tgt_dict.pad()), 0)
                    scores = scores.masked_fill((~valid_constraint_masks).all(2), 0)
                    scores = scores.sum(1)
                    scores = scores.view(-1, valid_size)
                    valid_result.append(scores)

                valid_result = torch.cat(valid_result, dim=-1)
                predicts = valid_result.argmax(1).tolist()
                hyps = [self.index2ans[predict_index] for predict_index in predicts]                    
            
            elif self.cfg.val_inference_type == "beamsearch":
                raw_hyps = self.inference_step(self.generator, [eval_model], sample, prefix_tokens=sample['prefix_tokens'])
                hyps = []
                for i, sample_id in enumerate(sample["id"].tolist()):
                    prefix_len = sample['prefix_tokens'][i].ne(1).sum().item()
                    detok_hypo_str = decode_fn(raw_hyps[i][0]["tokens"][prefix_len:], self.tgt_dict, self.bpe, self.generator)
                    hyps.append(detok_hypo_str.strip())

            else:
                raise NotImplementedError("Error: Unknown inference type encountered.")

        scores = [ref_dict.get(hyp, 0) for ref_dict, hyp in zip(sample['ref_dict'], hyps)]
        logging_output["_vqa_score_sum"] = sum(scores)
        logging_output["_vqa_cnt"] = len(scores)

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
            score = meters["_vqa_score_sum"].sum / meters["_vqa_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        if sum_logs("_vqa_cnt") > 0:
            metrics.log_scalar("_vqa_score_sum", sum_logs("_vqa_score_sum"))
            metrics.log_scalar("_vqa_cnt", sum_logs("_vqa_cnt"))
            metrics.log_derived("vqa_score", compute_score)
