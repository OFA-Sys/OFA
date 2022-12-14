# Copyright 2022 The OFA-Sys Team.
# All rights reserved.
# This source code is licensed under the Apache 2.0 license
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
from typing import Optional
from fairseq.tasks import register_task
from fairseq.data import FairseqDataset, iterators

from tasks.ofa_task import OFATask, OFAConfig
from data.s2t_data.unify_dataset import UnifyDataset
from data.file_dataset import FileDataset
from omegaconf import DictConfig
import string
from fairseq import metrics, utils
import editdistance
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class UnifySpeechTextConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": ""}
    )
    text_data: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data"},
    )
    audio_data: Optional[str] = field(
        default=None,
        metadata={"help": "pure audio data"},
    )
    speech_text_data: Optional[str] = field(
        default=None,
        metadata={"help": "speech text data"},
    )
    valid_data: Optional[str] = field(
        default=None,
        metadata={"help": "valid data"}
    )
    text_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data selected cols"},
    )
    audio_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure audio data selected cols"},
    )
    speech_text_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "speech_text data selected cols"},
    )

    pretrain_seed: int = field(
        default=7,
        metadata={"help": "pretrain seed"},
    )

    train_stage: int = field(
        default=2,
        metadata={"help": "train stage"}
    )
    audio_code_dict_size: int = field(
        default=30000,
        metadata={"help": "audio_code_dict_size"}
    )
    lang: str = field(
        default="zh",
        metadata={"help": "language"}
    )
    n_frames_per_step: int = field(
        default=1,
        metadata={"help": "n_frames_per_step of fbank"}
    )
    sample_rate: int = field(
        default=16000,
        metadata={"help": "sample rate of wav"}
    )
    phone_dict_path: Optional[str] = field(
        default=None,
        metadata={"help": "phone_dict_path"}
    )
    config_yaml_path: Optional[str] = field(
        default=None,
        metadata={"help": "config_yaml_path"}
    )
    eval_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU or CIDEr",
            "argparse_const": "@@ ",
        },
    )
    eval_wer: bool = field(
        default=False, metadata={"help": "evaluation with WER scores"}
    )
    text2phone_path: Optional[str] = field(
        default=None,
        metadata={"help": "text2phone_path"}
    )


@register_task("unify_speech_text_task", dataclass=UnifySpeechTextConfig)
class UnifySpeechTextTask(OFATask):
    def __init__(self, cfg: UnifySpeechTextConfig, src_dict, tgt_dict, phone_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.cfg = cfg
        self.phone_dict = phone_dict
        self.train_stage = cfg.train_stage
        self.text2phone_path = cfg.text2phone_path

        self.pure_text_dataset = None
        self.pure_audio_dataset = None
        self.speech_text_dataset = None
        if self.train_stage < 4:
            if self.cfg.text_data is not None:
                self.pure_text_dataset = FileDataset(self.cfg.text_data, self.cfg.text_selected_cols)
            if self.cfg.audio_data is not None and self.train_stage == 2:
                self.pure_audio_dataset = FileDataset(self.cfg.audio_data, self.cfg.audio_selected_cols)
            if self.cfg.speech_text_data is not None:
                self.speech_text_dataset = FileDataset(self.cfg.speech_text_data, self.cfg.speech_text_selected_cols)
        if self.train_stage == 1:
            self.valid_dataset = FileDataset(self.cfg.valid_data, self.cfg.text_selected_cols)
        else:
            self.valid_dataset = FileDataset(self.cfg.valid_data, self.cfg.speech_text_selected_cols)

    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""
        begin = 0
        end = 0
        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        end = len(src_dict)
        print("text_dict:", begin, end)
        begin = len(src_dict)
        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        src_dict.add_symbol("?")
        tgt_dict.add_symbol("?")
        src_dict.add_symbol("<blank>")
        tgt_dict.add_symbol("<blank>")
        
        # phone
        phone_dict = cls.load_dictionary(cfg.phone_dict_path)
        if cfg.text2phone_path is None:
            phone_dict.add_symbol("<blank>")
            phone_dict.add_symbol("<mask>")

        # src_dict.add_symbol("<phone_blank>")
        # tgt_dict.add_symbol("<phone_blank>")
        # for line in open(cfg.phone_dict_path, "r"):
        #     line = line.strip().split(" ")[0]
        #     src_dict.add_symbol("<phone_{}>".format(line))
        #     tgt_dict.add_symbol("<phone_{}>".format(line))
        # src_dict.add_symbol("<phone_mask_idx>")
        # tgt_dict.add_symbol("<phone_mask_idx>")

        # audio code
        for i in range(cfg.audio_code_dict_size):
            src_dict.add_symbol("<audio_{}>".format(i))
            tgt_dict.add_symbol("<audio_{}>".format(i))
        end = len(src_dict)
        print("audio_code_dict:", begin, end)
        begin = len(src_dict)

        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))
        end = len(src_dict)
        print("code_dict:", begin, end)
        begin = len(src_dict)
        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))
        end = len(src_dict)
        print("bin_dict:", begin, end)
        begin = len(src_dict)

        logger.info("source dictionary: {} types".format(len(src_dict)))
        logger.info("target dictionary: {} types".format(len(tgt_dict)))
        logger.info("phone dictionary: {} types".format(len(phone_dict)))
        return cls(cfg, src_dict, tgt_dict, phone_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        # paths = self.cfg.data.split(',')
        # assert len(paths) > 0
        #
        # file_path = paths[(epoch - 1) % (len(paths))]
        # dataset = FileDataset(file_path, self.cfg.selected_cols)

        if split != "train":
            dataset = self.valid_dataset
        else:
            if self.train_stage == 1:
                dataset = self.pure_text_dataset
            elif self.train_stage == 2:
                dataset = self.pure_audio_dataset
            else:
                dataset = self.speech_text_dataset

        self.datasets[split] = UnifyDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            self.phone_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            seed=self.cfg.pretrain_seed,
            code_dict_size=self.cfg.code_dict_size,
            audio_code_dict_size=self.cfg.audio_code_dict_size,
            num_bins=self.cfg.num_bins,
            pure_text_dataset=self.pure_text_dataset,
            pure_audio_dataset=self.pure_audio_dataset,
            speech_text_dataset=self.speech_text_dataset,
            config_yaml_path=self.cfg.config_yaml_path,
            lang=self.cfg.lang,
            text2phone_path=self.text2phone_path,
            train_stage=self.train_stage,
            n_frames_per_step=self.cfg.n_frames_per_step,
            sample_rate=self.cfg.sample_rate,
        )

    def get_batch_iterator(
            self,
            dataset,
            max_tokens=None,
            max_sentences=None,
            max_positions=None,
            ignore_invalid_inputs=False,
            required_batch_size_multiple=1,
            seed=1,
            num_shards=1,
            shard_id=0,
            num_workers=0,
            epoch=1,
            data_buffer_size=0,
            disable_iterator_cache=False,
    ):
        assert isinstance(dataset, FairseqDataset)

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        # create mini-batches with given size constraints
        batch_sampler = [
            [j for j in range(i, min(i + max_sentences, len(dataset)))]
            for i in range(0, len(dataset), max_sentences)
        ]
        total_row_count = dataset.dataset.get_total_row_count()
        num_batches = math.ceil(math.ceil(total_row_count / num_shards) / max_sentences)
        if len(batch_sampler) < num_batches:
            batch_sampler.append([1])

        # return a reusable, sharded iterator
        epoch_iter = iterators.EpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_sampler=batch_sampler,
            seed=seed,
            num_shards=1,
            shard_id=0,
            num_workers=num_workers,
            epoch=epoch,
            buffer_size=data_buffer_size
        )

        return epoch_iter

    def valid_step(self, sample, model, criterion, **extra_kwargs):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.cfg.eval_wer:
            hyps, refs = self._inference(self.sequence_generator, sample, model)
            scores = self._calculate_error_rate(hyps, refs)
            logging_output["_wer_score_sum"] = scores.sum()
            logging_output["_wer_cnt"] = scores.size
        return loss, sample_size, logging_output

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        if self.cfg.eval_wer:
            def compute_wer(meters):
                wer = meters["_wer_score_sum"].sum / meters["_wer_cnt"].sum
                wer = wer if isinstance(wer, float) else wer.item()
                return round(wer, 3)

            if sum_logs("_wer_cnt") > 0:
                metrics.log_scalar("_wer_score_sum", sum_logs("_wer_score_sum"))
                metrics.log_scalar("_wer_cnt", sum_logs("_wer_cnt"))
                metrics.log_derived("wer", compute_wer)

    def _inference(self, generator, sample, model):

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.cfg.eval_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        transtab = str.maketrans({key: None for key in string.punctuation})
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.translate(transtab).strip())
            refs.append(
                [
                    sent.translate(transtab).strip()
                    for sent in decode(
                        utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                        escape_unk=True,  # don't count <unk> as matches to the hypo
                    ).split('&&')
                ]
            )

        return hyps, refs

    def _calculate_error_rate(self, hyps, refs, unit="word"):
        """each line is "<text> (None-<index>)" """
        assert (len(hyps) == len(refs))
        refs = [x[0] for x in refs]
        err_rates = [
            editdistance.eval(hyp.split(), ref.split()) / len(ref.split()) for hyp, ref in zip(hyps, refs)
        ]
        err_rates = np.asarray(err_rates)
        return err_rates

    @property
    def phone_dictionary(self):
        """Return the phone :class:`~fairseq.data.Dictionary`."""
        return self.phone_dict

