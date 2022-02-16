import string
import math

import torch

from data import data_utils


def get_symbols_to_strip_from_output(generator):
    if hasattr(generator, "symbols_to_strip_from_output"):
        return generator.symbols_to_strip_from_output
    else:
        return {generator.eos}


def decode_fn(x, tgt_dict, bpe, generator):
    x = tgt_dict.string(x.int().cpu(), extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator))
    if bpe is not None:
        x = bpe.decode(x)
    return x


def eval_vqa_gen(task, generator, models, sample):
    hypos = task.inference_step(generator, models, sample)
    results = []
    for i, sample_id in enumerate(sample["id"].tolist()):
        detok_hypo_str = decode_fn(hypos[i][0]["tokens"], task.tgt_dict, task.bpe, generator)
        results.append({"question_id": sample_id, "answer": detok_hypo_str.strip()})
    scores = [ref_dict.get(result['answer'], 0) for ref_dict, result in zip(sample['ref_dict'], results)]
    return results, scores


def zero_shot_step(task, generator, models, sample):
    generator.zero_shot = True
    if task.cfg._name == 'vqa_gen':
        generator.constraint_trie = None
        return eval_vqa_gen(task, generator, models, sample)
    else:
        raise NotImplementedError
