# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import math
import string
from dataclasses import dataclass, field
from collections import OrderedDict
from typing import Optional

import torch
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass
from omegaconf import II

from data import data_utils
from utils.cider.pyciderevalcap.ciderD.ciderD import CiderD


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
class ScstRewardMOECriterionConfig(FairseqDataclass):
    scst_cider_cached_tokens: str = field(
        default="coco-train-words.p",
        metadata={"help": "path to cached cPickle file used to calculate CIDEr scores"},
    )
    ignore_prefix_size: int = field(
        default=0,
        metadata={"help": "Ignore first N tokens"},
    )
    sentence_avg: bool = II("optimization.sentence_avg")
    constraint_range: Optional[str] = field(
        default=None,
        metadata={"help": "constraint range"}
    )
    moe_gate_loss_wt: float = field(
        default=1.0,
        metadata={
            "help": "Weight associated with MoE gate loss"
                    "in the weighted sum of gate loss and cross entropy loss"
        }
    )
    moe_gate_loss_combine_method: str = field(
        default="average",
        metadata={
            "help": "Method of combining the gate loss from each MoE layers"
                    "('sum', 'average')"
        }
    )
    moe_gate_loss_transform: str = field(
        default="none",
        metadata={
            "help": "Transformation to apply to the gate loss ('none', 'neg_log')"
        }
    )


@register_criterion(
    "scst_reward_moe_criterion", dataclass=ScstRewardMOECriterionConfig
)
class ScstRewardMOECriterion(FairseqCriterion):
    CIDER_REWARD_WEIGHT = 1

    moe_logging_keys = [
        "overflow_expert1",        # average % of overflowed tokens from 1st expert
        "overflow_expert2",        # average % of overflowed tokens from 2nd expert
        "entropy_gating",          # average entropy of the gating distribution
        "expert1_balance_top",     # average cumulative % of tokens processed by the most used 20% 1st experts
        "expert1_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 1st experts
        "unused_expert1_count",    # average number of 1st experts which process no tokens
        "expert2_balance_top",     # average cumulative % of tokens processed by the most used 20% 2nd experts
        "expert2_balance_bottom",  # average cumulative % of tokens processed by the least used 20% 2nd experts
        "unused_expert2_count",    # average number of 2nd experts which process no tokens
        "all_to_all_cpu_time_ms",  # CPU time spent in all to all calls in milliseconds
        "all_to_all_cuda_time_ms", # CUDA ttime spent in all to all calls in milliseconds
    ]

    def __init__(
        self,
        task,
        moe_gate_loss_wt, 
        moe_gate_loss_combine_method, 
        moe_gate_loss_transform, 
        scst_cider_cached_tokens,
        sentence_avg,
        ignore_prefix_size=0,
        constraint_range=None
    ):
        super().__init__(task)
        self.scst_cider_scorer = CiderD(df=scst_cider_cached_tokens)
        self.sentence_avg = sentence_avg
        self.ignore_prefix_size = ignore_prefix_size
        self.transtab = str.maketrans({key: None for key in string.punctuation})

        self.constraint_start = None
        self.constraint_end = None
        if constraint_range is not None:
            constraint_start, constraint_end = constraint_range.split(',')
            self.constraint_start = int(constraint_start)
            self.constraint_end = int(constraint_end)

        self.gate_loss_weight = moe_gate_loss_wt
        self.gate_loss_combine_method = moe_gate_loss_combine_method
        self.gate_loss_transform = moe_gate_loss_transform


    def forward(self, model, sample, update_num=0, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        loss, moe_loss, score, ntokens, nsentences = self.compute_loss(model, sample, reduce=reduce)

        sample_size = (
            nsentences if self.sentence_avg else ntokens
        )
        logging_output = {
            "loss": loss.data,
            "moe_loss": moe_loss.data,
            "score": score,
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }
        logging_output.update(self.get_moe_metadata(model))
        return loss, sample_size, logging_output

    def _calculate_eval_scores(self, gen_res, gt_idx, gt_res):
        '''
        gen_res: generated captions, list of str
        gt_idx: list of int, of the same length as gen_res
        gt_res: ground truth captions, list of list of str.
            gen_res[i] corresponds to gt_res[gt_idx[i]]
            Each image can have multiple ground truth captions
        '''
        gen_res_size = len(gen_res)

        res = OrderedDict()
        for i in range(gen_res_size):
            res[i] = [self._wrap_sentence(gen_res[i].strip().translate(self.transtab))]

        gts = OrderedDict()
        gt_res_ = [
            [self._wrap_sentence(gt_res[i][j].strip().translate(self.transtab)) for j in range(len(gt_res[i]))]
                for i in range(len(gt_res))
        ]
        for i in range(gen_res_size):
            gts[i] = gt_res_[gt_idx[i]]

        res_ = [{'image_id':i, 'caption': res[i]} for i in range(len(res))]
        _, batch_cider_scores = self.scst_cider_scorer.compute_score(gts, res_)
        scores = self.CIDER_REWARD_WEIGHT * batch_cider_scores
        return scores

    @classmethod
    def _wrap_sentence(self, s):
        # ensure the sentence ends with <eos> token
        # in order to keep consisitent with cider_cached_tokens
        r = s.strip()
        if r.endswith('.'):
            r = r[:-1]
        r += ' <eos>'
        return r

    def get_generator_out(self, model, sample):
        def decode(toks):
            hypo = toks.int().cpu()
            hypo_str = self.task.tgt_dict.string(hypo)
            hypo_str = self.task.bpe.decode(hypo_str).strip()
            return hypo, hypo_str

        model.eval()
        with torch.no_grad():
            self.task.scst_generator.model.eval()
            gen_out = self.task.scst_generator.generate([model], sample)

        gen_target = []
        gen_res = []
        gt_res = []
        for i in range(len(gen_out)):
            for j in range(len(gen_out[i])):
                hypo, hypo_str = decode(gen_out[i][j]["tokens"])
                gen_target.append(hypo)
                gen_res.append(hypo_str)
            gt_res.append(
                decode(utils.strip_pad(sample["target"][i], self.padding_idx))[1].split('&&')
            )

        return gen_target, gen_res, gt_res

    def get_reward_and_scores(self, gen_res, gt_res, device):
        batch_size = len(gt_res)
        gen_res_size = len(gen_res)
        seq_per_img = gen_res_size // batch_size

        gt_idx = [i // seq_per_img for i in range(gen_res_size)]
        scores = self._calculate_eval_scores(gen_res, gt_idx, gt_res)
        sc_ = scores.reshape(batch_size, seq_per_img)
        baseline = (sc_.sum(1, keepdims=True) - sc_) / (sc_.shape[1] - 1)
        # sample - baseline
        reward = scores.reshape(batch_size, seq_per_img)
        reward = reward - baseline
        reward = reward.reshape(gen_res_size)
        reward = torch.as_tensor(reward, device=device, dtype=torch.float64)

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
        seq_per_img = gen_target_size // batch_size

        model.train()
        sample_src_tokens = torch.repeat_interleave(
            sample['net_input']['src_tokens'], seq_per_img, dim=0
        )
        sample_src_lengths = torch.repeat_interleave(
            sample['net_input']['src_lengths'], seq_per_img, dim=0
        )
        sample_patch_images = torch.repeat_interleave(
            sample['net_input']['patch_images'], seq_per_img, dim=0
        )
        sample_patch_masks = torch.repeat_interleave(
            sample['net_input']['patch_masks'], seq_per_img, dim=0
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
            patch_images=sample_patch_images, patch_masks=sample_patch_masks,
            prev_output_tokens=gen_prev_output_tokens
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
        gen_target, gen_res, gt_res = self.get_generator_out(model, sample)
        reward, scores = self.get_reward_and_scores(gen_res, gt_res, device=sample["target"].device)
        net_output, gen_target_tokens = self.get_net_output(model, sample, gen_target)
        gen_lprobs, gen_target_tokens = self.get_lprobs_and_target(model, net_output, gen_target_tokens)
        loss, ntokens = scst_loss(gen_lprobs, gen_target_tokens, reward, ignore_index=self.padding_idx, reduce=reduce)
        nsentences = gen_target_tokens.size(0)
        gate_loss = 0.0
        gate_count = 0
        for l_aux in net_output[1]["l_aux"]:
            if l_aux is not None:
                gate_loss += l_aux
                gate_count += 1
        if self.gate_loss_combine_method == "average":
            gate_loss = gate_loss / gate_count
        if self.gate_loss_transform == "neg_log":
            gate_loss = - torch.log(gate_loss)
        gate_loss = sample_size * gate_loss
        loss += self.gate_loss_weight * gate_loss

        return loss, gate_loss, scores.sum(), ntokens, nsentences

    def get_moe_metadata(self, model):
        moe_logging_output = {}
        for key in AdjustLabelSmoothedMOECrossEntropyCriterion.moe_logging_keys:
            total_val = 0
            count = 0
            for _, module in model.named_modules():
                if isinstance(module, MOELayer):
                    total_val += module.metadata[key] if key in module.metadata else 0
                    count += 1
            moe_logging_output[key] = total_val / count
        moe_logging_output["batch_count"] = 1
        return moe_logging_output

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        score_sum = sum(log.get("score", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        moe_loss_sum = sum(log.get("moe_loss", 0) for log in logging_outputs)

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
        metrics.log_scalar(
            "moe_gate_loss", moe_loss_sum / sample_size, sample_size, round=8
        )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
