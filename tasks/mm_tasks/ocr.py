# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

import torch
from dataclasses import dataclass, field
import json
import logging
from typing import Optional
from argparse import Namespace
import Levenshtein
from fairseq import metrics, utils
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.ocr_dataset import OcrDataset
from data.file_dataset import FileDataset

EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class OcrConfig(OFAConfig):
    is_document: bool = field(
        default=False, metadata={"help": "enable special resizing for document data."}
    )
    eval_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )


@register_task("ocr", dataclass=OcrConfig)
class OcrTask(OFATask):
    def __init__(self, cfg: OcrConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(",")
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = OcrDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            is_document=self.cfg.is_document,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        gen_args = json.loads(self.cfg.eval_args)
        self.sequence_generator = self.build_generator([model], Namespace(**gen_args))

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        hyps, refs = self._inference(self.sequence_generator, sample, model)
        acc = [1.0 if hyp == ref else 0.0 for hyp, ref in zip(hyps, refs)]
        distance = [
            Levenshtein.distance(hyp, ref) / max(len(hyp), len(ref))
            for hyp, ref in zip(hyps, refs)
        ]
        logging_output["_acc_sum"] = sum(acc)
        logging_output["_acc_cnt"] = len(acc)
        logging_output["_dist_sum"] = sum(distance)
        logging_output["_dist_cnt"] = len(distance)

        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        def compute_acc(meters):
            score = meters["_acc_sum"].sum / meters["_acc_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 4)

        def compute_ned(meters):
            score = meters["_dist_sum"].sum / meters["_dist_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            score = 1.0 - score
            return round(score, 4)

        if sum_logs("_acc_cnt") > 0:
            metrics.log_scalar("_acc_sum", sum_logs("_acc_sum"))
            metrics.log_scalar("_acc_cnt", sum_logs("_acc_cnt"))
            metrics.log_derived("acc", compute_acc)
            metrics.log_scalar("_dist_sum", sum_logs("_dist_sum"))
            metrics.log_scalar("_dist_cnt", sum_logs("_dist_cnt"))
            metrics.log_derived("ned", compute_ned)

    def _inference(self, generator, sample, model):
        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.strip().replace(" ", ""))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,
                )
                .strip()
                .replace(" ", "")
            )
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))

        return hyps, refs