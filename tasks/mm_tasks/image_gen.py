# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

from dataclasses import dataclass, field
import json
import logging
import os
import math
import base64
from typing import Optional
from argparse import Namespace
from omegaconf import DictConfig, OmegaConf
from torchvision import transforms
from PIL import Image
from io import BytesIO

import torch
import numpy as np
from fairseq import metrics
from fairseq.tasks import register_task
from fairseq.dataclass import ChoiceEnum

from models import search, clip
from models.taming.models.vqgan import GumbelVQ
from data.mm_data.image_gen_dataset import ImageGenDataset
from data.file_dataset import FileDataset

from tasks.ofa_task import OFATask, OFAConfig

logger = logging.getLogger(__name__)


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


EVAL_CLIP_METHOD = ChoiceEnum(["ii_sim", "ti_sim"])

@dataclass
class ImageGenConfig(OFAConfig):
    sampling_times: int = field(
        default=1, metadata={"help": "sample times"}
    )

    code_image_size: int = field(
        default=256, metadata={"help": "code image size"}
    )

    # options for reporting CLIP score during validation
    eval_clip_method: EVAL_CLIP_METHOD = field(
        default='ti_sim',
        metadata={
            "help": "evaluation with CLIP scores. ii_sim means Similarity between generated Images and ref Images, ti_sim means Similarity between generated Images and input Text"}
    )

    eval_args: Optional[str] = field(
        default='{}',
        metadata={
            "help": 'generation args for clip scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )

    scst: bool = field(
        default=False, metadata={"help": "Self-critical sequence training"}
    )
    scst_args: str = field(
        default='{}',
        metadata={
            "help": 'generation args for Self-critical sequence training, as JSON string'
        },
    )

    vqgan_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "path of vqgan model"}
    )
    vqgan_config_path: Optional[str] = field(
        default=None,
        metadata={"help": "path of vqgan config"}
    )
    clip_model_path: Optional[str] = field(
        default=None,
        metadata={"help": "clip model path"}
    )
    gen_images_path: str = field(
        default='', metadata={"help": "where to store generated images during evalution. Don't dump images if None. "}
    )


@register_task("image_gen", dataclass=ImageGenConfig)
class ImageGenTask(OFATask):
    def __init__(self, cfg: ImageGenConfig, src_dict, tgt_dict):
        super().__init__(cfg, src_dict, tgt_dict)

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = self.cfg.data.split(',')
        assert len(paths) > 0

        if split == 'train':
            file_path = paths[(epoch - 1) % (len(paths) - 1)]
        else:
            file_path = paths[-1]
        dataset = FileDataset(file_path, self.cfg.selected_cols)

        self.datasets[split] = ImageGenDataset(
            split,
            dataset,
            self.bpe,
            self.src_dict,
            self.tgt_dict,
            max_src_length=self.cfg.max_src_length,
            code_dict_size=self.cfg.code_dict_size,
            code_image_size=self.cfg.code_image_size
        )

    def build_model(self, cfg):
        model = super().build_model(cfg)

        device = torch.cuda.current_device()
        clip_model, clip_preprocess = clip.load(self.cfg.clip_model_path, device=device)
        self.clip_model = clip_model
        self.clip_preprocess = clip_preprocess
        self.clip_model.to(device)
        self.clip_model.eval()

        vqgan_config = OmegaConf.load(self.cfg.vqgan_config_path)
        vqgan = GumbelVQ(**vqgan_config.model.params)
        sd = torch.load(self.cfg.vqgan_model_path, map_location="cpu")["state_dict"]
        missing, unexpected = vqgan.load_state_dict(sd, strict=False)
        for k, v in vqgan.named_parameters():
            v.requires_grad = False
        self.image_tokenizer = vqgan
        self.image_tokenizer.to(device)
        self.image_tokenizer.eval()

        gen_args = json.loads(self.cfg.eval_args)
        self.sequence_generator = self.build_generator(
            [model], Namespace(**gen_args)
        )
        if self.cfg.scst:
            scst_args = json.loads(self.cfg.scst_args)
            self.scst_generator = self.build_generator(
                [model], Namespace(**scst_args)
            )

        return model

    def build_generator(
            self, models, args, seq_gen_cls=None, extra_gen_cls_kwargs=None, prefix_allowed_tokens_fn=None,
    ):
        """
        Build a :class:`~fairseq.SequenceGenerator` instance for this
        task.

        Args:
            models (List[~fairseq.models.FairseqModel]): ensemble of models
            args (fairseq.dataclass.configs.GenerationConfig):
                configuration object (dataclass) for generation
            extra_gen_cls_kwargs (Dict[str, Any]): extra options to pass
                through to SequenceGenerator
            prefix_allowed_tokens_fn (Callable[[int, torch.Tensor], List[int]]):
                If provided, this function constrains the beam search to
                allowed tokens only at each step. The provided function
                should take 2 arguments: the batch ID (`batch_id: int`)
                and a unidimensional tensor of token ids (`inputs_ids:
                torch.Tensor`). It has to return a `List[int]` with the
                allowed tokens for the next generation step conditioned
                on the previously generated tokens (`inputs_ids`) and
                the batch ID (`batch_id`). This argument is useful for
                constrained generation conditioned on the prefix, as
                described in "Autoregressive Entity Retrieval"
                (https://arxiv.org/abs/2010.00904) and
                https://github.com/facebookresearch/GENRE.
        """
        from models.sequence_generator import SequenceGenerator

        # Choose search strategy. Defaults to Sampling.
        self.sampling_times = self.cfg.sampling_times
        sampling = True  # we have to use sampling instead of beam search in image generation task
        sampling_topk = getattr(args, "sampling_topk", -1)
        sampling_topp = getattr(args, "sampling_topp", -1.0)

        assert sampling_topk < 0 or sampling, "--sampling-topk requires --sampling"
        assert sampling_topp < 0 or sampling, "--sampling-topp requires --sampling"

        search_strategy = search.Sampling(
            self.target_dictionary, sampling_topk, sampling_topp
        )
        extra_gen_cls_kwargs = extra_gen_cls_kwargs or {}

        return SequenceGenerator(
            models,
            self.target_dictionary,
            beam_size=getattr(args, "beam", 5),
            max_len_a=getattr(args, "max_len_a", 0),
            max_len_b=getattr(args, "max_len_b", 200),
            min_len=getattr(args, "min_len", 1),
            normalize_scores=(not getattr(args, "unnormalized", False)),
            len_penalty=getattr(args, "lenpen", 1),
            unk_penalty=getattr(args, "unkpen", 0),
            temperature=getattr(args, "temperature", 1.0),
            match_source_len=getattr(args, "match_source_len", False),
            no_repeat_ngram_size=getattr(args, "no_repeat_ngram_size", 0),
            search_strategy=search_strategy,
            constraint_range=self.cfg.constraint_range,
            gen_code=True,
            **extra_gen_cls_kwargs,
        )

    def compute_ref_image_similarity(self, hyps, ref, device):
        hyp_images = torch.stack(
            [self.clip_preprocess(hyp_image) for hyp_image in hyps], dim=0
        ).to(device)

        ref_images = self.clip_preprocess(ref).unsqueeze(0).to(device)
        with torch.no_grad():
            hyp_image_features = self.clip_model.encode_image(hyp_images)
            ref_image_features = self.clip_model.encode_image(ref_images)
        hyp_image_features /= hyp_image_features.norm(dim=-1, keepdim=True)
        ref_image_features /= ref_image_features.norm(dim=-1, keepdim=True)
        similarity = hyp_image_features @ ref_image_features.T
        # scores.append(similarity.max().item())
        sorted_score, indices = torch.sort(similarity.view(-1), descending=True)
        return sorted_score, indices

    def compute_text_similarity(self, hyps, text, device):
        hyp_images = torch.stack(
            [self.clip_preprocess(hyp_image) for hyp_image in hyps], dim=0
        ).to(device)

        clip_input = clip.tokenize([text]).to(device)
        with torch.no_grad():
            hyp_image_features = self.clip_model.encode_image(hyp_images)
            hyp_image_features /= hyp_image_features.norm(dim=-1, keepdim=True)
            text_features = self.clip_model.encode_text(clip_input)
            text_features /= text_features.norm(dim=-1, keepdim=True)
        ti_similarity = hyp_image_features @ text_features.T
        sorted_score, indices = torch.sort(ti_similarity.view(-1), descending=True)
        return sorted_score, indices

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = criterion(model, sample)

        model.eval()
        device = sample['target'].device

        hyps, ref = self.inference_image(self.sequence_generator, sample, [model])
        scores = []

        tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
        caption = self.bpe.decode(self.tgt_dict.string([token for token in tokens if token >= 4]))[
                  38:].replace('/', '')
        if self.cfg.eval_clip_method == 'ii_sim':
            similarity_score, indices = self.compute_ref_image_similarity(hyps, ref, device)
        elif self.cfg.eval_clip_method == 'ti_sim':
            similarity_score, indices = self.compute_text_similarity(hyps, caption, device)
        else:
            raise ValueError("unsupported eval method.")

        scores.append(similarity_score.max().item())
        sorted_hyps = [hyps[indice] for indice in indices]

        if self.cfg.gen_images_path:
            caption_tokens = sample['net_input']['src_tokens'][0].view(-1).tolist()
            caption = self.bpe.decode(self.tgt_dict.string([token for token in caption_tokens if token >= 4]))[
                      38:].replace('/', '')
            self.dump_images(sorted_hyps, text=caption, path=os.path.join(self.cfg.gen_images_path, 'all_results'))
            self.dump_images(sorted_hyps, text=caption, path=os.path.join(self.cfg.gen_images_path, 'top1'), topk=1)

        logging_output["_score_sum"] = sum(scores)
        logging_output["_score_cnt"] = len(scores)

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
            score = meters["_score_sum"].sum / meters["_score_cnt"].sum
            score = score if isinstance(score, float) else score.item()
            return round(score, 3)

        if sum_logs("_score_cnt") > 0:
            metrics.log_scalar("_score_sum", sum_logs("_score_sum"))
            metrics.log_scalar("_score_cnt", sum_logs("_score_cnt"))
            metrics.log_derived("score", compute_score)

    def inference_image(self, generator, sample, models):
        hyps, ref = [], None
        for j in range(self.sampling_times):
            gen_out = self.inference_step(generator, models, sample)
            for i in range(len(gen_out)):
                with torch.no_grad():
                    tokens = torch.stack([item['tokens'][:-1] for item in gen_out[i]], dim=0)
                    tokens += -len(self.src_dict) + self.cfg.code_dict_size + self.cfg.num_bins
                    images = self.image_tokenizer.decode_code(
                        tokens.view(-1, self.cfg.code_image_size // 8, self.cfg.code_image_size // 8)
                    )
                    images = [custom_to_pil(image) for image in images]
                hyps += images
        if 'code_images' in sample:
            ref = Image.open(BytesIO(base64.urlsafe_b64decode(sample['code_images'][0]))).convert('RGB')

        return hyps, ref

    def dump_images(self, images, text, path, topk=None):
        os.makedirs(path, exist_ok=True)
        if topk:
            images = images[:topk]
        for j, image in enumerate(images):
            save_path = os.path.join(path, f'{text}_{j}.png')
            image.save(save_path)
