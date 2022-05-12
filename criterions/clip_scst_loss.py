# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
from torchvision import transforms

import torch
import numpy as np
from fairseq import metrics
from fairseq.data import data_utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from fairseq import utils
from omegaconf import II

from models import clip


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


def scst_loss(lprobs, target, reward, ignore_index=None, reduce=True):
    loss = -lprobs.gather(dim=-1, index=target.unsqueeze(-1)).squeeze() * reward.unsqueeze(-1)
    if ignore_index is not None:
        pad_mask = target.eq(ignore_index)
        loss.masked_fill_(pad_mask, 0.0)
        ntokens = (~pad_mask).sum()
    else:
        loss = loss.squeeze(-1)
        ntokens = target.numel()
    if reduce:
        loss = loss.sum()
    return loss, ntokens


@dataclass
class ClipScstRewardCriterionConfig(FairseqDataclass):
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )


@register_criterion(
    "clip_scst_reward_criterion", dataclass=ClipScstRewardCriterionConfig
)
class ClipScstRewardCriterion(FairseqCriterion):
    CLIP_REWARD_WEIGHT = 2.5

    def __init__(
        self,
        task,
        sentence_avg,
        ignore_prefix_size=0,
        constraint_range=None
    ):
        super().__init__(task)
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, score, ntokens, nsentences = self.compute_loss(model, sample, reduce=reduce)

        sample_size = (
            nsentences if self.sentence_avg else ntokens
        )
        logging_output = {
            "loss": loss.data,
            "score": score,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        return loss, sample_size, logging_output

    def _calculate_clip_scores(self, gen_res, gt_text, device):
        '''
        gen_res: generated images, list of Image
        gt_text: input captions.
        device: device for clip model
        '''
        batch_size = len(gt_text)
        gen_res_size = len(gen_res)
        img_per_seq = gen_res_size // batch_size

        hyp_images = torch.stack(
            [self.task.clip_preprocess(gen_image) for gen_image in gen_res], dim=0
        ).to(device)

        clip_input = clip.tokenize([text for text in gt_text]).to(device)
        with torch.no_grad():
            image_features = self.task.clip_model.encode_image(hyp_images)
            text_features = self.task.clip_model.encode_text(clip_input)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            image_features = image_features.view(batch_size, img_per_seq, -1)
            text_features = text_features.view(batch_size, 1, -1)
            ti_similarity = image_features @ text_features.transpose(1, 2)
            ti_similarity = ti_similarity.view(-1)

        scores = self.CLIP_REWARD_WEIGHT * ti_similarity
        return scores

    def get_generator_out(self, model, sample):
        model.eval()
        with torch.no_grad():
            self.task.scst_generator.model.eval()
            gen_out = self.task.scst_generator.generate([model], sample)

        gen_target = []
        gen_res = []
        gt_text = []
        for i in range(len(gen_out)):
            with torch.no_grad():
                tokens = torch.stack([item['tokens'][:-1] for item in gen_out[i]], dim=0)
                tokens += -len(self.task.src_dict) + self.task.cfg.code_dict_size + self.task.cfg.num_bins
                images = self.task.image_tokenizer.decode_code(
                    tokens.view(-1, self.task.cfg.code_image_size // 8, self.task.cfg.code_image_size // 8)
                )
                images = [custom_to_pil(image) for image in images]

            gen_target += [item['tokens'] for item in gen_out[i]]
            gen_res += images
            gt_text.append(
                self.task.bpe.decode(
                    self.task.tgt_dict.string(
                        utils.strip_pad(sample['net_input']['src_tokens'][i], self.padding_idx).cpu().int()
                    )
                )[38:] # remove task instruction.
            )

        return gen_target, gen_res, gt_text

    def get_reward_and_scores(self, gen_res, gt_text, device):
        batch_size = len(gt_text)
        gen_res_size = len(gen_res)
        img_per_sample = gen_res_size // batch_size

        scores = self._calculate_clip_scores(gen_res, gt_text, device)
        sc_ = scores.reshape(batch_size, img_per_sample)
        baseline = (sc_.sum(1, keepdim=True) - sc_) / (sc_.shape[1] - 1)
        # sample - baseline
        reward = scores.reshape(batch_size, img_per_sample)
        reward = reward - baseline
        reward = reward.view(-1)

        return reward, scores

    def get_net_output(self, model, sample, gen_target):
        def merge(sample_list, eos=self.task.tgt_dict.eos(), move_eos_to_beginning=False):
            return data_utils.collate_tokens(
                sample_list,
                pad_idx=self.padding_idx,
                eos_idx=eos,
                left_pad=False,
                move_eos_to_beginning=move_eos_to_beginning,
            )

        batch_size = len(sample["target"])
        gen_target_size = len(gen_target)
        img_per_sample = gen_target_size // batch_size

        model.train()
        sample_src_tokens = torch.repeat_interleave(
            sample['net_input']['src_tokens'], img_per_sample, dim=0
        )
        sample_src_lengths = torch.repeat_interleave(
            sample['net_input']['src_lengths'], img_per_sample, dim=0
        )
        sample_code_masks = torch.repeat_interleave(
            sample['net_input']['code_masks'], img_per_sample, dim=0
        )
        gen_prev_output_tokens = torch.as_tensor(
            merge(gen_target, eos=self.task.tgt_dict.bos(), move_eos_to_beginning=True),
            device=sample["target"].device, dtype=torch.int64
        )
        gen_target_tokens = torch.as_tensor(
            merge(gen_target), device=sample["target"].device, dtype=torch.int64
        )
        net_output = model(
            src_tokens=sample_src_tokens, src_lengths=sample_src_lengths,
            code_masks=sample_code_masks, prev_output_tokens=gen_prev_output_tokens
        )

        return net_output, gen_target_tokens

    def get_lprobs_and_target(self, model, net_output, gen_target):
        if self.constraint_start is not None and self.constraint_end is not None:
            net_output[0][:, :, 4:self.constraint_start] = -math.inf
            net_output[0][:, :, self.constraint_end:] = -math.inf
        lprobs = model.get_normalized_probs(net_output, log_probs=True)
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size :, :].contiguous()
                gen_target = gen_target[:, self.ignore_prefix_size :].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size :, :, :].contiguous()
                gen_target = gen_target[self.ignore_prefix_size :, :].contiguous()
        return lprobs, gen_target

    def compute_loss(self, model, sample, reduce=True):
        gen_target, gen_res, gt_text = self.get_generator_out(model, sample)
        reward, scores = self.get_reward_and_scores(gen_res, gt_text, device=sample["target"].device)
        net_output, gen_target_tokens = self.get_net_output(model, sample, gen_target)
        gen_lprobs, gen_target_tokens = self.get_lprobs_and_target(model, net_output, gen_target_tokens)
        loss, ntokens = scst_loss(gen_lprobs, gen_target_tokens, reward, ignore_index=self.padding_idx, reduce=reduce)
        nsentences = gen_target_tokens.size(0)

        return loss, scores.sum(), ntokens, nsentences

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        score_sum = sum(log.get("score", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "score", score_sum / nsentences, nsentences, round=3
        )

        metrics.log_scalar(
            "ntokens", ntokens, 1, round=3
        )
        metrics.log_scalar(
            "nsentences", nsentences, 1, round=3
        )
        metrics.log_scalar(
            "sample_size", sample_size, 1, round=3
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
