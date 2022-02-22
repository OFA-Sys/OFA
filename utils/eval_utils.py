import string
import math
import json
from itertools import chain
import os

import torch
import torch.distributed as dist

from data import data_utils


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


def eval_caption(task, generator, models, sample):
    transtab = str.maketrans({key: None for key in string.punctuation})
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        results.append({"image_id": str(sample_id), "caption": detok_hypo_str.translate(transtab).strip()})
    return results, None


def eval_refcoco(task, generator, models, sample):
    def _calculate_ap_score(hyps, refs, thresh=0.5):
        interacts = torch.cat(
            [torch.where(hyps[:, :2] < refs[:, :2], refs[:, :2], hyps[:, :2]),
             torch.where(hyps[:, 2:] < refs[:, 2:], hyps[:, 2:], refs[:, 2:])],
            dim=1
        )
        area_predictions = (hyps[:, 2] - hyps[:, 0]) * (hyps[:, 3] - hyps[:, 1])
        area_targets = (refs[:, 2] - refs[:, 0]) * (refs[:, 3] - refs[:, 1])
        interacts_w = interacts[:, 2] - interacts[:, 0]
        interacts_h = interacts[:, 3] - interacts[:, 1]
        area_interacts = interacts_w * interacts_h
        ious = area_interacts / (area_predictions + area_targets - area_interacts + 1e-6)
        return ((ious >= thresh) & (interacts_w > 0) & (interacts_h > 0)).float()

    gen_out = task.inference_step(generator, models, sample)
    hyps = []
    for i in range(len(gen_out)):
        hyps.append(gen_out[i][0]["tokens"][:-1] - len(task.src_dict) + task.cfg.num_bins)
    hyps = torch.stack(hyps, dim=0)
    hyps = hyps / (task.cfg.num_bins - 1) * task.cfg.max_image_size
    hyps[:, ::2] /= sample['w_resize_ratios'].unsqueeze(1)
    hyps[:, 1::2] /= sample['h_resize_ratios'].unsqueeze(1)

    results = [
        {"uniq_id": sample_id,
         "box": [hyps[i][0].item(), hyps[i][1].item(), hyps[i][2].item(), hyps[i][3].item()]}
        for i, sample_id in enumerate(sample["id"].tolist())
    ]
    scores = _calculate_ap_score(hyps, sample['region_coords'].float())
    return results, scores


def eval_image_gen(task, generator, models, sample):
    hypos, _ = task.inference_image(generator, sample, models)
    tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
    caption = task.bpe.decode(task.tgt_dict.string([token for token in tokens if token >= 4]))[
              38:].replace('/', '')

    text_similarity_score, indices = task.compute_text_similarity(hypos, caption,
                                                                  sample['net_input']['src_tokens'].device)
    results = []
    for i, indice in enumerate(indices):
        results.append({"sample_id": str(sample["id"][0]), "score": text_similarity_score[i], "image": hypos[indice]})

    scores = [max(text_similarity_score).item()]
    return results, scores


def eval_step(task, generator, models, sample):
    if task.cfg._name == 'caption':
        return eval_caption(task, generator, models, sample)
    if task.cfg._name == 'image_gen':
        return eval_image_gen(task, generator, models, sample)
    elif task.cfg._name == 'refcoco':
        return eval_refcoco(task, generator, models, sample)
    else:
        raise NotImplementedError


def merge_results(task, cfg, logger, score_cnt, score_sum, results):
    if task.cfg._name == 'image_gen':
        if cfg.distributed_training.distributed_world_size > 1:
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))
    else:
        gather_results = None
        if cfg.distributed_training.distributed_world_size > 1:
            gather_results = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(gather_results, results)
            dist.all_reduce(score_sum.data)
            dist.all_reduce(score_cnt.data)
        if score_cnt.item() > 0:
            logger.info("score_sum: {}, score_cnt: {}, score: {}".format(
                score_sum, score_cnt, round(score_sum.item() / score_cnt.item(), 4)
            ))

        if cfg.distributed_training.distributed_world_size == 1 or dist.get_rank() == 0:
            os.makedirs(cfg.common_eval.results_path, exist_ok=True)
            output_path = os.path.join(cfg.common_eval.results_path, "{}_predict.json".format(cfg.dataset.gen_subset))
            gather_results = list(chain(*gather_results)) if gather_results is not None else results
            with open(output_path, 'w') as fw:
                json.dump(gather_results, fw)
