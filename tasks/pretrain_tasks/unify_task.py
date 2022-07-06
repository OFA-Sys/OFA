# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import json
import logging
import os
from typing import Optional
from fairseq.tasks import register_task

from tasks.ofa_task import OFATask, OFAConfig
from data.pretrain_data.unify_dataset import UnifyDataset
from data.file_dataset import FileDataset

logger = logging.getLogger(__name__)


@dataclass
class UnifyConfig(OFAConfig):
    max_image_size: int = field(
        default=512, metadata={"help": ""}
    )
    text_data: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data"},
    )
    image_data: Optional[str] = field(
        default=None,
        metadata={"help": "pure image data"},
    )
    detection_data: Optional[str] = field(
        default=None,
        metadata={"help": "detection data"},
    )
    text_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure text data selected cols"},
    )
    image_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "pure image data selected cols"},
    )
    detection_selected_cols: Optional[str] = field(
        default=None,
        metadata={"help": "detection data selected cols"},
    )
    neg_sample_dir: Optional[str] = field(
        default=None,
        metadata={"help": "negative sample directory, which contains captions (taken from all image-text pairs), "
                          "answers (taken from VQA), "
                          "objects (taken form OpenImages) "},
    )
    code_image_size: int = field(
        default=128, metadata={"help": "the resolution of the generated image in the image infilling task"}
    )

    pretrain_seed: int = field(
        default=7,
        metadata={"help": "pretrain seed"},
    )

    mask_ratio: float = field(
        default=0.3,
        metadata={"help": "fraction of words/subwords that will be masked"},
    )
    random_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], use random token this often"},
    )
    keep_ratio: float = field(
        default=0.0,
        metadata={"help": "instead of using [MASK], keep original token this often"},
    )
    mask_length: str = field(
        default="span-poisson",
        metadata={"help": "mask length to choose ['subword', 'word', 'span-poisson']"},
    )
    poisson_lambda: float = field(
        default=3.0,
        metadata={"help": "randomly shuffle sentences for this proportion of inputs"},
    )
    replace_length: int = field(
        default=1,
        metadata={"help": "when masking N tokens, replace with 0, 1, or N tokens (use -1 for N)"},
    )


@register_task("unify_task", dataclass=UnifyConfig)
class UnifyTask(OFATask):
    def __init__(self, cfg: UnifyConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

        self.type2ans_dict = json.load(open(os.path.join(self.cfg.neg_sample_dir, 'type2ans.json')))
        self.ans2type_dict = {}
        for type, answer_list in self.type2ans_dict.items():
            if type == 'other':
                continue
            for answer in answer_list:
                self.ans2type_dict[answer] = type

        self.all_object_list = [
            row.strip() for row in open(os.path.join(self.cfg.neg_sample_dir, 'object.txt')) if row.strip() != ''
        ]
        self.all_caption_list = [
            row.strip() for row in open(os.path.join(self.cfg.neg_sample_dir, 'all_captions.txt')) if row.strip() != ''
        ]

        self.pure_text_dataset = None
        self.pure_image_dataset = None
        self.detection_dataset = None
        if self.cfg.text_data is not None:
            self.pure_text_dataset = FileDataset(self.cfg.text_data, self.cfg.text_selected_cols)
        if self.cfg.image_data is not None:
            self.pure_image_dataset = FileDataset(self.cfg.image_data, self.cfg.image_selected_cols)
        if self.cfg.detection_data is not None:
            self.detection_dataset = FileDataset(self.cfg.detection_data, self.cfg.detection_selected_cols)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        file_path = paths[(epoch - 1) % (len(paths))]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = UnifyDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            seed=self.cfg.pretrain_seed,
            code_dict_size=self.cfg.code_dict_size,
            num_bins=self.cfg.num_bins,
            patch_image_size=self.cfg.patch_image_size,
            code_image_size=self.cfg.code_image_size,
            pure_text_dataset=self.pure_text_dataset,
            pure_image_dataset=self.pure_image_dataset,
            detection_dataset=self.detection_dataset,
            all_object_list=self.all_object_list,
            all_caption_list=self.all_caption_list,
            type2ans_dict=self.type2ans_dict,
            ans2type_dict=self.ans2type_dict,
            max_image_size=self.cfg.max_image_size,
            mask_ratio=self.cfg.mask_ratio,
            random_ratio=self.cfg.random_ratio,
            keep_ratio=self.cfg.keep_ratio,
            mask_length=self.cfg.mask_length,
            poisson_lambda=self.cfg.poisson_lambda,
            replace_length=self.cfg.replace_length
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
