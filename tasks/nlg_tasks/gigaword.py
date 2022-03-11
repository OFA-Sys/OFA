# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import string
from typing import Optional
from argparse import Namespace
from fairseq import metrics
from fairseq.tasks import register_task
from fairseq.data import encoders

from tasks.ofa_task import OFATask, OFAConfig
from data.nlg_data.summary_dataset import SummaryDataset
from data.file_dataset import FileDataset
from datasets import load_metric

logger = logging.getLogger(__name__)


_tok_dict = {"(": "-lrb-", ")": "-rrb-",
             "[": "-lsb-", "]": "-rsb-",
             "{": "-lcb-", "}": "-rcb-",
             "[UNK]": "UNK", '&': '&amp;', '<': '&lt;', '>': '&gt;'}


def _is_digit(w):
    for ch in w:
        if not(ch.isdigit() or ch == ','):
            return False
    return True


def fix_tokenization(text):
    input_tokens = text.split()
    output_tokens = []
    has_left_quote = False
    has_left_single_quote = False

    i = 0
    prev_dash = False
    while i < len(input_tokens):
        tok = input_tokens[i]
        flag_prev_dash = False
        if tok in _tok_dict.keys():
            output_tokens.append(_tok_dict[tok])
            i += 1
        elif tok == "\"":
            if has_left_quote:
                output_tokens.append("''")
            else:
                output_tokens.append("``")
            has_left_quote = not has_left_quote
            i += 1
        elif tok == "'" and len(output_tokens) > 0 and output_tokens[-1].endswith("n") and i < len(input_tokens) - 1 and input_tokens[i + 1] == "t":
            output_tokens[-1] = output_tokens[-1][:-1]
            output_tokens.append("n't")
            i += 2
        elif tok == "'" and i < len(input_tokens) - 1 and input_tokens[i + 1] in ("s", "d", "ll"):
            output_tokens.append("'"+input_tokens[i + 1])
            i += 2
        elif tok == "'":
            if has_left_single_quote:
                output_tokens.append("'")
            else:
                output_tokens.append("`")
            has_left_single_quote = not has_left_single_quote
            i += 1
        elif tok == "." and i < len(input_tokens) - 2 and input_tokens[i + 1] == "." and input_tokens[i + 2] == ".":
            output_tokens.append("...")
            i += 3
        elif tok == "," and len(output_tokens) > 0 and _is_digit(output_tokens[-1]) and i < len(input_tokens) - 1 and _is_digit(input_tokens[i + 1]):
            # $ 3 , 000 -> $ 3,000
            output_tokens[-1] += ','+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and output_tokens[-1].isdigit() and i < len(input_tokens) - 1 and input_tokens[i + 1].isdigit():
            # 3 . 03 -> $ 3.03
            output_tokens[-1] += '.'+input_tokens[i + 1]
            i += 2
        elif tok == "." and len(output_tokens) > 0 and len(output_tokens[-1]) == 1 and output_tokens[-1].isupper() and i < len(input_tokens) - 2 and len(input_tokens[i + 1]) == 1 and input_tokens[i + 1].isupper() and input_tokens[i + 2] == '.':
            # U . N . -> U.N.
            k = i+3
            while k+2 < len(input_tokens):
                if len(input_tokens[k + 1]) == 1 and input_tokens[k + 1].isupper() and input_tokens[k + 2] == '.':
                    k += 2
                else:
                    break
            output_tokens[-1] += ''.join(input_tokens[i:k])
            i += 2
        elif tok == "-":
            if i < len(input_tokens) - 1 and input_tokens[i + 1] == "-":
                output_tokens.append("--")
                i += 2
            elif i == len(input_tokens) - 1 or i == 0:
                output_tokens.append("-")
                i += 1
            elif output_tokens[-1] not in string.punctuation and input_tokens[i + 1][0] not in string.punctuation:
                output_tokens[-1] += "-"
                i += 1
                flag_prev_dash = True
            else:
                output_tokens.append("-")
                i += 1
        elif prev_dash and len(output_tokens) > 0 and tok[0] not in string.punctuation:
            output_tokens[-1] += tok
            i += 1
        else:
            output_tokens.append(tok)
            i += 1
        prev_dash = flag_prev_dash
    return " ".join(output_tokens)


@dataclass
class GigawordConfig(OFAConfig):
    # options for reporting Rouge during validation
    eval_rouge: bool = field(
        default=False, metadata={"help": "evaluation with rouge scores"}
    )
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU or CIDEr (e.g., 'moses'); "
                    "required if using --eval-bleu or --eval-cider; "
                    "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    noise_ratio: float = field(
        default=0.0, metadata={"help": "noise ratio for prev output"}
    )


@register_task("gigaword", dataclass=GigawordConfig)
class GigawordTask(OFATask):
    def __init__(self, cfg: GigawordConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = SummaryDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            noise_ratio=self.cfg.noise_ratio
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        if self.cfg.eval_rouge:
            detok_args = json.loads(self.cfg.eval_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.cfg.eval_detok, **detok_args)
            )

            gen_args = json.loads(self.cfg.eval_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
            self.metric = load_metric('../../utils/rouge.py')

        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_rouge:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            result = self.metric.compute(predictions=hyps, references=refs, use_agregator=False, use_stemmer=True)
            result_recall = {key: sum([item.recall for item in value]) * 100 for key, value in result.items()}
            result_f1 = {key: sum([item.fmeasure for item in value]) * 100 for key, value in result.items()}

            logging_output['_rouge1_recall_sum'] = result_recall['rouge1']
            logging_output['_rouge2_recall_sum'] = result_recall['rouge2']
            logging_output['_rougeL_recall_sum'] = result_recall['rougeL']
            logging_output['_rouge1_f1_sum'] = result_f1['rouge1']
            logging_output['_rouge2_f1_sum'] = result_f1['rouge2']
            logging_output['_rougeL_f1_sum'] = result_f1['rougeL']
            logging_output['_rouge_cnt'] = len(hyps)
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if sum_logs("_rouge_cnt") > 0:
            metrics.log_scalar("_rouge1_recall_sum", sum_logs("_rouge1_recall_sum"))
            metrics.log_scalar("_rouge2_recall_sum", sum_logs("_rouge2_recall_sum"))
            metrics.log_scalar("_rougeL_recall_sum", sum_logs("_rougeL_recall_sum"))
            metrics.log_scalar("_rouge1_f1_sum", sum_logs("_rouge1_f1_sum"))
            metrics.log_scalar("_rouge2_f1_sum", sum_logs("_rouge2_f1_sum"))
            metrics.log_scalar("_rougeL_f1_sum", sum_logs("_rougeL_f1_sum"))
            metrics.log_scalar("_rouge_cnt", sum_logs("_rouge_cnt"))
            metrics.log_derived("rouge1_recall", lambda x: x["_rouge1_recall_sum"].sum / x["_rouge_cnt"].sum)
            metrics.log_derived("rouge2_recall", lambda x: x["_rouge2_recall_sum"].sum / x["_rouge_cnt"].sum)
            metrics.log_derived("rougeL_recall", lambda x: x["_rougeL_recall_sum"].sum / x["_rouge_cnt"].sum)
            metrics.log_derived("rouge1_f1", lambda x: x["_rouge1_f1_sum"].sum / x["_rouge_cnt"].sum)
            metrics.log_derived("rouge2_f1", lambda x: x["_rouge2_f1_sum"].sum / x["_rouge_cnt"].sum)
            metrics.log_derived("rougeL_f1", lambda x: x["_rougeL_f1_sum"].sum / x["_rouge_cnt"].sum)

    def _inference(self, generator, sample, model):
        def decode(toks):
            s = self.tgt_dict.string(toks.int().cpu())
            if self.bpe:
                s = self.bpe.decode(s)
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyp = decode(gen_out[i][0]["tokens"]).lower().strip()
            hyp = fix_tokenization(hyp).replace('<unk>', ' unk').replace('1', '#')
            ref = sample["target_strs"][i]
            hyps.append(hyp)
            refs.append(ref)
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])

        return hyps, refs
