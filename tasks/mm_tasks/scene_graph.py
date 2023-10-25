# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import copy
from dataclasses import dataclass, field
import itertools
import json
import logging
import os
from typing import Optional
from argparse import Namespace
from itertools import zip_longest
from collections import OrderedDict

import numpy as np
from omegaconf import DictConfig
import sacrebleu
import string
from fairseq import metrics, utils
from fairseq.tasks import register_task
from data.mm_data.sg_dataset import SGDataset
import torch
from torch import distributed
from torchvision.ops import nms, box_iou

from tasks.ofa_task import OFATask, OFAConfig
from data.mm_data.caption_dataset import CaptionDataset
from data.file_dataset import FileDataset
from utils.sg_eval import BasicSceneGraphEvaluator
from utils.sgg_eval import SGRecall, SGMeanRecall
from utils.eval_utils import toks2triplets


EVAL_BLEU_ORDER = 4

logger = logging.getLogger(__name__)


@dataclass
class SGClsConfig(OFAConfig):
    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for BLUE or CIDEr scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )
    vg_json_dir: Optional[str] = field(
        default=None, metadata={"help": "path to the directory of visual genome json files"}
    )
    sg_mode: Optional[str] = field(
        default='sgdet', metadata={"help": "sgdet, sgcls, predcls"}
    )


@register_task("scene_graph", dataclass=SGClsConfig)
class SceneGraphTask(OFATask):
    def __init__(self, cfg: SGClsConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)
        self.valid_count = 0

        if cfg.vg_json_dir is not None:
            with open(cfg.vg_json_dir) as f:
                self.vg_json = json.load(f)
        else:
            with open('/data/hulab/zcai75/visual_genome/VG-SGG-dicts-with-attri.json') as f:
                self.vg_json = json.load(f)
        
        self.sg_mode = cfg.sg_mode
        self.eval_args = Namespace(**json.loads(cfg.eval_args))
        # self.sg_evaluator = BasicSceneGraphEvaluator('sgdet')
        self.result_dict = {}
        self.sgRecall = SGRecall(self.result_dict)
        idx_to_predicate = [None] * (len(self.vg_json['idx_to_predicate']) + 1)
        for idx, pred in self.vg_json['idx_to_predicate'].items():
            idx_to_predicate[int(idx)] = pred
        self.sgMeanRecall = SGMeanRecall(self.result_dict, len(idx_to_predicate), idx_to_predicate)
        self.sgRecall.register_container(self.sg_mode)
        self.sgMeanRecall.register_container(self.sg_mode)
    
    @classmethod
    def setup_task(cls, cfg: DictConfig, **kwargs):
        """Setup the task."""

        # load dictionaries
        src_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )
        tgt_dict = cls.load_dictionary(
            os.path.join(cfg.bpe_dir, "dict.txt")
        )

        for symbol in ['<sub>', '<obj>', '<pred>']:
            src_dict.add_symbol(symbol)
            tgt_dict.add_symbol(symbol)

        src_dict.add_symbol("<mask>")
        tgt_dict.add_symbol("<mask>")
        for i in range(cfg.code_dict_size):
            src_dict.add_symbol("<code_{}>".format(i))
            tgt_dict.add_symbol("<code_{}>".format(i))

        # quantization
        for i in range(cfg.num_bins):
            src_dict.add_symbol("<bin_{}>".format(i))
            tgt_dict.add_symbol("<bin_{}>".format(i))

        logger.info("sg setup: source dictionary: {} types".format(len(src_dict)))
        logger.info("sg setup: target dictionary: {} types".format(len(tgt_dict)))
        return cls(cfg, src_dict, tgt_dict)


    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = SGDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            max_tgt_length=self.cfg.max_tgt_length,
            patch_image_size=self.cfg.patch_image_size,
            num_bins=self.cfg.num_bins,
            imagenet_default_mean_and_std=self.cfg.imagenet_default_mean_and_std,
            mode=self.cfg.sg_mode,
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)
        gen_args = json.loads(self.cfg.eval_args)
        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )

        return model

    def valid_step(self, sample, model, criterion):
        def decode(toks):
            s = self.tgt_dict.string(
                toks.int().cpu()
            )
            return s

        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()

        out = self.inference_step(self.sequence_generator, [model], sample)
        sents = [
            [decode(out[i][j]['tokens']) for j in range(len(out[i]))]
        for i in range(len(out))]
        sents = [sent[:1] for sent in sents]

        if self.valid_count % 100 == 0:
            try:
                print([self.bpe.decode(decode(o['tokens'])) for o in out[0]])
            except:
                pass

        for i in range(len(sample["id"])):
            image_id = sample["id"][i]
            img_size = sample["img_size"][i]

            local_container = {}
            # calculate recall for each image
            label_to_idx = self.vg_json['label_to_idx']
            predicate_to_idx = self.vg_json['predicate_to_idx']
            gt_boxes = np.array(sample['gt_boxes'][i])
            gt_relations = np.array(sample['gt_relations'][i])
            gt_classes = np.array(sample['gt_classes'][i])
            gt_entry = {'gt_boxes': gt_boxes, 'gt_classes': gt_classes, 'gt_relations': gt_relations, 'gt_rels': gt_relations}
            local_container.update(gt_entry)

            triplets = []
            for sent in sents[i]:
                triplets.extend(toks2triplets(sent.split(), self.bpe, self.cfg.num_bins, img_size))
            
            pred_boxes = []
            pred_rel_inds = []
            rel_scores = []
            pred_classes_all = []
            for ((sub_label, sub_box), rel_label, (obj_label, obj_box)) in triplets:
                if sub_label not in label_to_idx or obj_label not in label_to_idx or rel_label not in predicate_to_idx:
                    print('label not found', sub_label, obj_label, rel_label)
                    continue

                pred_rel_inds.append([])

                pred_boxes.append(sub_box)
                pred_classes_all.append(label_to_idx[sub_label])
                pred_rel_inds[-1].append(len(pred_boxes) - 1)
                
                pred_boxes.append(obj_box)
                pred_classes_all.append(label_to_idx[obj_label])
                pred_rel_inds[-1].append(len(pred_boxes) - 1)
                
                vec = np.zeros(len(predicate_to_idx) + 1)
                vec[predicate_to_idx[rel_label]] = 1.0
                rel_scores.append(vec)
            
            if len(pred_boxes) > 0:
                # cluster the boxes
                pred_boxes = torch.atleast_2d(torch.tensor(pred_boxes))
                nms_keep = nms(pred_boxes, torch.ones(len(pred_boxes)), 0.99)
                pred_boxes_nms = pred_boxes[nms_keep]
                close = box_iou(pred_boxes, pred_boxes_nms) > 0.99
                nms_map = {m[0]: m[1] for m in close.nonzero().tolist()} # maps from pred_boxes id to pred_boxes_nms id

                rel_keep = []
                for j, rel in enumerate(pred_rel_inds):
                    if rel[0] in nms_map and rel[1] in nms_map:
                        rel_keep.append(j)
                        pred_rel_inds[j] = [nms_map[i] for i in rel]
                pred_rel_inds = np.array(pred_rel_inds)[rel_keep]
                rel_scores = np.array(rel_scores)[rel_keep]

                pred_boxes = np.array(pred_boxes_nms)

                pred_classes = np.array(pred_classes_all)[nms_keep.numpy()]
                obj_scores = np.zeros((len(pred_boxes), len(label_to_idx)))
                for j, c in enumerate(pred_classes):
                    obj_scores[j, c-1] = 1.0

                if len(pred_rel_inds) > 0:
                    assert pred_rel_inds.max() < len(pred_boxes), (pred_boxes.shape, pred_rel_inds)
            else:
                print('no prediction')
                pred_boxes = np.zeros((0, 4))
                pred_classes = np.zeros(0)
                obj_scores = np.zeros((0, len(label_to_idx)))
                pred_rel_inds = np.zeros((0, 2), dtype=np.int64)
                rel_scores = np.zeros((0, len(predicate_to_idx)))

            pred_entry = {'pred_boxes': pred_boxes, 'pred_rel_inds': pred_rel_inds, 'rel_scores': rel_scores,
                          'pred_classes': pred_classes, 'obj_scores': obj_scores}
            local_container.update(pred_entry)

            # self.sg_evaluator.evaluate_scene_graph_entry(gt_entry, pred_entry)

            try:
                self.sgRecall.collect_recall({'iou_thres': 0.5}, local_container, self.sg_mode)
                self.sgMeanRecall.collect_mean_recall_items({'iou_thres': 0.5}, local_container, self.sg_mode)
            except Exception as e:
                print(e)
                print(local_container)
            
            # draw the gt and pred bounding boxes and save the image

            # from torchvision.transforms import functional as F
            # from torchvision.utils import draw_bounding_boxes
            # from PIL import Image

            # img = sample['net_input']['patch_images'][i].cpu().squeeze()
            # img = F.convert_image_dtype(img, torch.uint8)
            # img_size = img_size[::-1]
            # gt_boxes_scaled = torch.tensor(gt_boxes, dtype=torch.float32)
            # gt_boxes_scaled[0::2] *= img_size[0] / 1024
            # gt_boxes_scaled[1::2] *= img_size[1] / 1024
            # pred_boxes_scaled = torch.tensor(pred_boxes)
            # pred_boxes_scaled[0::2] *= img_size[0] / 1024
            # pred_boxes_scaled[1::2] *= img_size[1] / 1024
            # img = F.resize(img, img_size)
            # img = draw_bounding_boxes(img, gt_boxes_scaled, width=2, colors='green')
            # img = draw_bounding_boxes(img, pred_boxes_scaled, width=2, colors='red')
            # img = Image.fromarray(img.numpy().transpose(1, 2, 0))
            # img.save(f'{image_id}.jpg')

            self.valid_count += 1
            # exit()
        
        # logging_output.update(self.sg_evaluator.result_dict)
        # for k, recs in self.sg_evaluator.result_dict[f'{self.sg_mode}_recall'].items():
        #     logging_output[f'{self.sg_mode}_recall@{k}_sum'] = sum(recs)
        #     logging_output[f'{self.sg_mode}_recall@{k}_count'] = len(recs)

        # logging_output.update(self.sgRecall.result_dict)
        # logging_output.update(self.sgMeanRecall.result_dict)

        logging_output['trip_count'] = len(triplets)

        return loss, sample_size, logging_output
    
    def get_valid_stats(self, cfg, trainer, agg_val):
        # if torch.distributed.get_rank() == 0:
        #     self.sg_evaluator.print_stats()
        # print('post validate', stats)

        group_size = torch.distributed.get_world_size()
        gather_list = [None] * group_size
        distributed.all_gather_object(gather_list, self.result_dict)

        # recursively merge dictionaries and concat lists
        def merge_dict(d1, d2):
            for k, v in d2.items():
                if isinstance(v, dict):
                    merge_dict(d1[k], v)
                else:
                    if type(d1[k]) == list and type(v) == list:
                        d1[k] += v
                    elif type(v) == np.float64 or type(d1[k]) == float:
                        assert d1[k] == 0.0 and v == 0.0, (k, d1[k], v)

        # self.result_dict = copy.deepcopy(gather_list[0])
        for i in range(1, group_size):
            merge_dict(self.result_dict, gather_list[i])
        
        self.sgRecall.calculate_recall(self.sg_mode)
        self.sgMeanRecall.calculate_mean_recall(self.sg_mode)

        print_str = ''
        print_str += self.sgRecall.generate_print_string(self.sg_mode)
        print_str += self.sgMeanRecall.generate_print_string(self.sg_mode)
        print(print_str)

        for key, value in self.result_dict.items():
            if type(value) == dict:
                for k, v in value.items():
                    if type(v) == np.float64 or type(v) == float:
                        agg_val[f'{key}@{k}'] = v
                    else:
                        # print(key, len(v))
                        pass
    
        self.sgRecall.register_container(self.sg_mode)
        self.sgMeanRecall.register_container(self.sg_mode)

        return agg_val

    def post_validate(self, model, stats, agg):
        self.valid_count = 0

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)

        def sum_logs(key):
            import torch
            result = sum(log.get(key, 0) for log in logging_outputs)
            if torch.is_tensor(result):
                result = result.cpu()
            return result

        # print('reduce metrics', logging_outputs)
        # recalls = [log[f'{self.sg_mode}_recall'] for log in logging_outputs if f'{self.sg_mode}_recall' in log] # skip the training logs
        # if len(recalls) == 0:
        #     return
        # agg = {}
        # for k in self.sg_evaluator.result_dict[f'{self.sg_mode}_recall']:
        #     # agg[k] = list(itertools.chain(*[r[k] for r in recalls]))
        #     # self.sg_evaluator.result_dict[f'{self.sg_mode}_recall'].update(agg)
        #     recall_sum = sum_logs(f'{self.sg_mode}_recall@{k}_sum')
        #     recall_count = sum_logs(f'{self.sg_mode}_recall@{k}_count')
        #     if recall_count > 0:
        #         metrics.log_scalar(f'{self.sg_mode}_recall@{k}', recall_sum / recall_count * 100, round=3)
    
    def logging_outputs_can_be_summed(criterion):
        return True

    def _inference(self, generator, sample, model):

        def decode(toks):
            s = self.tgt_dict.string(
                toks.int().cpu()
            )
            if self.bpe:
                s = self.bpe.decode(s)
            return s

        gen_out = self.inference_step(generator, [model], sample)
        hyps, refs = [], []
        # transtab = str.maketrans({key: None for key in string.punctuation})
        for i in range(len(gen_out)):
            decode_tokens = decode(gen_out[i][0]["tokens"])
            hyps.append(decode_tokens.strip())
            refs.append(
                decode(
                        utils.strip_pad(sample["target"][i], self.tgt_dict.pad())
                    ).split('&&')
            )
        if self.cfg.eval_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + ' && '.join(refs[0]))

        return hyps, refs
