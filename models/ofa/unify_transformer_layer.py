# Copyright 2022 The OFA-Sys Team. 
# All rights reserved.
# This source code is licensed under the Apache 2.0 license 
# found in the LICENSE file in the root directory.

import torch.distributed as dist
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from fairseq import utils
from fairseq.distributed import utils as dist_utils, fsdp_wrap
from fairseq.modules import LayerNorm
from fairseq.modules.moe import Top1Gate, Top2Gate, MOELayer
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.fused_bias_gelu import fused_bias_gelu, has_fused_bias_gelu
from fairseq.modules.quant_noise import quant_noise
from torch import Tensor
from .unify_multihead_attention import MultiheadAttention

class Dummy_layer(nn.Module):

    def __init__(self, d_model, d_inner, activation_fn, ffn_layernorm, activation_dropout_module, dropout_module, quant_noise, quant_noise_block_size, resmoe_topk=1, resmoe_double_expert=False, resmoe_input=0, resmoe_output=1, resmoe_num_expert=4, resmoe_freeze_tuning=False):
        super().__init__()
        self.topK = resmoe_topk
        self.d_model = d_model
        self.d_inner = d_inner
        self.double_expert = resmoe_double_expert
        # 0: X, 1: X + FFN(X), 2: FFN(X)
        self.moe_input = resmoe_input
        # 1: MoE(X) + FFN(X), 2: MoE(FFN(X)) + FFN(X), 3: MoE(FFN(X))
        self.moe_output = resmoe_input
        self.num_expert = resmoe_num_expert
        self.freeze_tuning = resmoe_freeze_tuning
        self.activation_fn = activation_fn
        self.ffn_layernorm = ffn_layernorm
        self.activation_dropout_module = activation_dropout_module
        self.dropout_module = dropout_module
        self.quant_noise = quant_noise
        self.quant_noise_block_size = quant_noise_block_size

        self.fc1 = self.build_fc1(
            self.d_model,
            self.d_inner,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            self.d_inner,
            self.d_model,
            self.quant_noise,
            self.quant_noise_block_size,
        )
    
    def forward(self, x):
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        if self.ffn_layernorm is not None:
            x = self.ffn_layernorm(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        return x

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

def _linear(x, weight, bias=None):
    return F.linear(x, weight, bias)

def _ffn(
    x,
    fc1,
    activation_fn,
    activation_dropout_module,
    fc2,
    dropout_module,
):
    x_shape = x.shape
    x = x.reshape(-1, x.size(-1))
    if has_fused_bias_gelu and activation_fn == gelu:
        x = _linear(x, fc1.weight)
        x = fused_bias_gelu(x, fc1.bias)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    else:
        x = _linear(x, fc1.weight, fc1.bias)
        x = activation_fn(x)
        x = activation_dropout_module(x)
        x = _linear(x, fc2.weight, fc2.bias)
    x = x.view(x_shape)
    x = dropout_module(x)
    return x


class FeedForwardNetwork(nn.Module):
    """
        Feed Forward Network layer in the Transformer model
    """
    def __init__(self, args, embed_dim, ffn_dim, dropout_module=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)
        self.activation_fn = utils.get_activation_fn(
            activation=str(args.activation_fn)
            if getattr(args, "activation_fn", None) is not None
            else "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.fc1 = self.build_fc1(
            self.embed_dim,
            ffn_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc2(
            ffn_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.dropout_module = FairseqDropout(
                args.dropout, module_name=self.__class__.__name__
            ) if not dropout_module else dropout_module

    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def forward(self, x):
        return _ffn(
            x,
            fc1=self.fc1,
            activation_fn=self.activation_fn,
            activation_dropout_module=self.activation_dropout_module,
            fc2=self.fc2,
            dropout_module=self.dropout_module,
        )
        return x

def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    Comment by Ross Wightman: This is the same as the DropConnect impl I created for EfficientNet, etc networks,
    however, the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for changing the
    layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use 'survival rate' as the
    argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (1, x.shape[1], 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

    def extra_repr(self) -> str:
        return "p={}".format(self.drop_prob)


class TransformerEncoderLayer(nn.Module):
    """Encoder layer block.

    In the original paper each operation (multi-head attention or FFN) is
    postprocessed with: `dropout -> add residual -> layernorm`. In the
    tensor2tensor code they suggest that learning is more robust when
    preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.encoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
    """

    def __init__(self, args, drop_path_rate=0.0, is_moe_layer=False):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.ffn_dim = args.encoder_ffn_embed_dim
        self.quant_noise = getattr(args, 'quant_noise_pq', 0)
        self.quant_noise_block_size = getattr(args, 'quant_noise_pq_block_size', 8) or 8
        self.self_attn = self.build_self_attention(self.embed_dim, args)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.is_moe_layer = is_moe_layer
        if self.is_moe_layer and getattr(args, "alternate_ffn_embed_dim", 0.0) > 0:
            self.ffn_dim = getattr(args, "alternate_ffn_embed_dim", 0.0)
        self.normalize_before = args.encoder_normalize_before
        if not self.is_moe_layer or getattr(args, "alternate_ffn_embed_dim", 0.0) > 0:
            self.activation_fn = utils.get_activation_fn(
                activation=getattr(args, 'activation_fn', 'relu') or "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            
            self.fc1 = self.build_fc1(
                self.embed_dim,
                self.ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                self.ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                    getattr(args, "moe_batch_prioritized_routing", False),
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim, self.dropout_module)
            self.moe_layer = MOELayer(gate, experts, args)
            # self.doe = Dummy_layer(self.embed_dim, args.encoder_ffn_embed_dim, self.activation_fn, self.ffn_layernorm, self.activation_dropout_module, self.dropout_module, self.quant_noise, self.quant_noise_block_size)        

        self.attn_ln = LayerNorm(self.embed_dim) if getattr(args, 'scale_attn', False) else None
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        self.ffn_layernorm = LayerNorm(args.encoder_ffn_embed_dim) if getattr(args, 'scale_fc', False) else None
        self.w_resid = nn.Parameter(torch.ones(self.embed_dim, ), requires_grad=True) if getattr(args, 'scale_resids', False) else None
            
        self.final_layer_norm = LayerNorm(self.embed_dim)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()


    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            scale_factor=args.attn_scale_factor,
            scale_heads=getattr(args, 'scale_heads', False)
        )

    def residual_connection(self, x, residual):
        return residual + self.drop_path(x)

    def residual_scale_combination(self, x, residual):
        return (residual + self.drop_path(x)) / 2

    def residual_dynamic_scale_combination(self, x, residual):
        if self.training_times < 5000:
            return residual * (1 - self.training_times / 5000) + self.drop_path(x) * self.training_times / 5000

    def residual_learned_scale_combination(self, x, residual):
        if self.learned_res_coef >= 1:
            self.learned_res_coef = 1 - 1e-7
        elif self.learned_res_coef <= 0:
            self.learned_res_coef = 1e-7
        return residual * (self.learned_res_coef) + self.drop_path(x) * (1 - self.learned_res_coef)
    

    def doe_checkpoint_loader(self, name, state_dict, n_experts=4, noise_coef=0, boise_coef=0.05, ln_coef=0.05):

        # get fc1_weights, fc2_weights, fc1_bias, fc2_bias
        fc1_weights = state_dict["{}.fc1.weight".format(name)]
        fc2_weights = state_dict["{}.fc2.weight".format(name)]
        fc1_bias = state_dict["{}.fc1.bias".format(name)]
        fc2_bias = state_dict["{}.fc2.bias".format(name)]
        ffn_weights = state_dict["{}.ffn_layernorm.weight".format(name)]
        ffn_bias = state_dict["{}.ffn_layernorm.bias".format(name)]

        # copy fc1, fc2 + noise
        state_dict["{}.doe.fc1.weight".format(name)] = fc1_weights + torch.randn(fc1_weights.size()) * noise_coef
        state_dict["{}.doe.fc2.weight".format(name)] = fc2_weights + torch.randn(fc2_weights.size()) * noise_coef
        state_dict["{}.doe.fc1.bias".format(name)] = fc1_bias
        state_dict["{}.doe.fc2.bias".format(name)] = fc2_bias 
        state_dict["{}.doe.ffn_layernorm.weight".format(name)] = ffn_weights     
        state_dict["{}.doe.ffn_layernorm.bias".format(name)] = ffn_bias

    def moe_checkpoint_loader(self, name, state_dict, noise_coef=0):

        n_experts = self.args.moe_expert_count
        # get fc1_weights, fc2_weights, fc1_bias, fc2_bias
        fc1_weights = state_dict["{}.fc1.weight".format(name)]
        fc2_weights = state_dict["{}.fc2.weight".format(name)]
        fc1_bias = state_dict["{}.fc1.bias".format(name)]
        fc2_bias = state_dict["{}.fc2.bias".format(name)]
        ffn_weights = state_dict["{}.ffn_layernorm.weight".format(name)]
        ffn_bias = state_dict["{}.ffn_layernorm.bias".format(name)]

        # copy fc1, fc2 + noise
        for i in range(n_experts):
            state_dict["{}.moe_layer.experts.{}.fc1.weight".format(name, i)] = fc1_weights + torch.randn(fc1_weights.size()) * noise_coef
            state_dict["{}.moe_layer.experts.{}.fc2.weight".format(name, i)] = fc2_weights + torch.randn(fc2_weights.size()) * noise_coef
            state_dict["{}.moe_layer.experts.{}.fc1.bias".format(name, i)] = fc1_bias
            state_dict["{}.moe_layer.experts.{}.fc2.bias".format(name, i)] = fc2_bias
        
    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]
                if "{}.{}.{}".format(name, new, m) not in state_dict and "{}.{}".format(new, m) in self.state_dict():
                    state_dict[
                        "{}.{}.{}".format(name, new, m)
                    ] = self.state_dict()["{}.{}".format(new, m)]

        moe_checker = "{}.moe_layer.gate.wg.weight".format(name)
        if moe_checker not in state_dict and "moe_layer.gate.wg.weight" in self.state_dict():
            print("Loading Pre-trained checkpoints to MoE")
            self.moe_checkpoint_loader(name, state_dict)
        # if "{}.doe.fc1.weight".format(name) not in state_dict and "doe.fc1.weight" in self.state_dict():
        #     self.doe_checkpoint_loader(name, state_dict)

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict and param_name in self.state_dict():
                state_dict[prefix + param_name] = self.state_dict()[param_name]
        
            
    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
        self_attn_bias: Optional[Tensor] = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(
                attn_mask.to(torch.bool),
                -1e8 if x.dtype == torch.float32 else -1e4
            )

        # self.training_times += 1
        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        x, _ = self.self_attn(
            query=x,
            key=x,
            value=x,
            key_padding_mask=encoder_padding_mask,
            need_weights=False,
            attn_mask=attn_mask,
            attn_bias=self_attn_bias
        )
        if self.attn_ln is not None:
            x = self.attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        
        if not self.is_moe_layer or getattr(self.args, "alternate_ffn_embed_dim", 0.0) > 0:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            if self.ffn_layernorm is not None:
                x = self.ffn_layernorm(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            l_aux = None
        else:
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(self.args, "use_moe_pad_mask", False):
                x, l_aux = self.moe_layer(x, input_padding_mask=encoder_padding_mask)
            else:
                x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1) # seq_len, batch_size, model_dim

        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x, l_aux


class TransformerDecoderLayer(nn.Module):
    """Decoder layer block.

    In the original paper each operation (multi-head attention, encoder
    attention or FFN) is postprocessed with: `dropout -> add residual ->
    layernorm`. In the tensor2tensor code they suggest that learning is more
    robust when preprocessing each layer with layernorm and postprocessing with:
    `dropout -> add residual`. We default to the approach in the paper, but the
    tensor2tensor approach can be enabled by setting
    *args.decoder_normalize_before* to ``True``.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        no_encoder_attn (bool, optional): whether to attend to encoder outputs
            (default: False).
    """

    def __init__(
        self, args, no_encoder_attn=False, add_bias_kv=False, add_zero_attn=False, drop_path_rate=0.0, is_moe_layer=False
    ):
        super().__init__()
        self.args = args
        self.is_moe_layer = is_moe_layer
        self.embed_dim = args.decoder_embed_dim
        self.ffn_dim = args.decoder_ffn_embed_dim
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8)

        self.cross_self_attention = getattr(args, "cross_self_attention", False)

        self.self_attn = self.build_self_attention(
            self.embed_dim,
            args,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
        )
        self.self_attn_ln = LayerNorm(self.embed_dim) if getattr(args, 'scale_attn', False) else None
        self.cross_attn_ln = LayerNorm(self.embed_dim) if getattr(args, 'scale_attn', False) else None
        self.nh = self.self_attn.num_heads
        self.head_dim = self.self_attn.head_dim

        # use layerNorm rather than FusedLayerNorm for exporting.
        # char_inputs can be used to determint this.
        # TODO  remove this once we update apex with the fix
        export = getattr(args, "char_inputs", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        if no_encoder_attn:
            self.encoder_attn = None
            self.encoder_attn_layer_norm = None
        else:
            self.encoder_attn = self.build_encoder_attention(self.embed_dim, args)
            self.encoder_attn_layer_norm = LayerNorm(self.embed_dim, export=export)

        self.ffn_layernorm = LayerNorm(args.decoder_ffn_embed_dim) if getattr(args, 'scale_fc', False) else None
        self.w_resid = nn.Parameter(torch.ones(self.embed_dim, ), requires_grad=True) if getattr(args, 'scale_resids', False) else None

        if self.is_moe_layer and getattr(args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            self.ffn_dim = getattr(args, "alternate_decoder_ffn_embed_dim", 0.0)

        if not self.is_moe_layer or getattr(args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            self.activation_fn = utils.get_activation_fn(
                activation=str(args.activation_fn)
                if getattr(args, "activation_fn", None) is not None
                else "relu"
            )
            activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
            if activation_dropout_p == 0:
                # for backwards compatibility with models that use args.relu_dropout
                activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
            self.activation_dropout_module = FairseqDropout(
                float(activation_dropout_p), module_name=self.__class__.__name__
            )
            self.fc1 = self.build_fc1(
                self.embed_dim,
                self.ffn_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
            self.fc2 = self.build_fc2(
                self.ffn_dim,
                self.embed_dim,
                self.quant_noise,
                self.quant_noise_block_size,
            )
        else:
            if args.moe_top1_expert:
                gate = Top1Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    use_fp32=args.moe_gating_use_fp32,
                    moe_eval_capacity_token_fraction=getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                )
            else:
                gate = Top2Gate(
                    self.embed_dim,
                    args.moe_expert_count,
                    args.moe_gating_use_fp32,
                    args.moe_second_expert_policy,
                    args.moe_normalize_gate_prob_before_dropping,
                    getattr(args, "moe_eval_capacity_token_fraction", 0.25),
                    getattr(args, "moe_batch_prioritized_routing", False),
                )
            experts = make_experts(args, self.embed_dim, self.ffn_dim, self.dropout_module)
            self.moe_layer = MOELayer(gate, experts, args)

        self.normalize_before = args.decoder_normalize_before
        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.need_attn = True
        self.onnx_trace = False
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0.0 else nn.Identity()


    def build_fc1(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_fc2(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(nn.Linear(input_dim, output_dim), q_noise, qn_block_size)

    def build_self_attention(
        self, embed_dim, args, add_bias_kv=False, add_zero_attn=False
    ):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            dropout=args.attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=not getattr(args, "cross_self_attention", False),
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            scale_factor=args.attn_scale_factor,
            scale_heads=getattr(args, 'scale_heads', False)
        )

    def build_encoder_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.decoder_attention_heads,
            kdim=getattr(args, "encoder_embed_dim", None),
            vdim=getattr(args, "encoder_embed_dim", None),
            dropout=args.attention_dropout,
            encoder_decoder_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
            scale_factor=args.attn_scale_factor,
            scale_heads=getattr(args, 'scale_heads', False)
        )

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def residual_connection(self, x, residual):
        return residual + self.drop_path(x)

    def residual_scale_combination(self, x, residual):
        return (residual + self.drop_path(x)) / 2
    
    def residual_dynamic_scale_combination(self, x, residual):
        if self.training_times < 5000:
            return residual * (1 - self.training_times / 5000) + self.drop_path(x) * self.training_times / 5000
        else:
            return (residual + self.drop_path(x)) / 2

    def residual_learned_scale_combination(self, x, residual):
        if self.learned_res_coef >= 1:
            self.learned_res_coef = 1 - 1e-7
        elif self.learned_res_coef <= 0:
            self.learned_res_coef = 1e-7
        return residual * (self.learned_res_coef) + self.drop_path(x) * (1 - self.learned_res_coef)


    def forward(
        self,
        x,
        encoder_out: Optional[torch.Tensor] = None,
        encoder_padding_mask: Optional[torch.Tensor] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        prev_self_attn_state: Optional[List[torch.Tensor]] = None,
        prev_attn_state: Optional[List[torch.Tensor]] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        self_attn_padding_mask: Optional[torch.Tensor] = None,
        need_attn: bool = False,
        need_head_weights: bool = False,
        self_attn_bias: Optional[Tensor] = None,
        cross_attn_bias: Optional[Tensor] = None
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor, optional): binary
                ByteTensor of shape `(batch, src_len)` where padding
                elements are indicated by ``1``.
            need_attn (bool, optional): return attention weights
            need_head_weights (bool, optional): return attention weights
                for each head (default: return average over heads).

        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # self.training_times += 1

        if need_head_weights:
            need_attn = True

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if prev_self_attn_state is not None:
            prev_key, prev_value = prev_self_attn_state[:2]
            saved_state: Dict[str, Optional[Tensor]] = {
                "prev_key": prev_key,
                "prev_value": prev_value,
            }
            if len(prev_self_attn_state) >= 3:
                saved_state["prev_key_padding_mask"] = prev_self_attn_state[2]
            assert incremental_state is not None
            self.self_attn._set_input_buffer(incremental_state, saved_state)
        _self_attn_input_buffer = self.self_attn._get_input_buffer(incremental_state)
        if self.cross_self_attention and not (
            incremental_state is not None
            and _self_attn_input_buffer is not None
            and "prev_key" in _self_attn_input_buffer
        ):
            if self_attn_mask is not None:
                assert encoder_out is not None
                self_attn_mask = torch.cat(
                    (x.new_zeros(x.size(0), encoder_out.size(0)), self_attn_mask), dim=1
                )
            if self_attn_padding_mask is not None:
                if encoder_padding_mask is None:
                    assert encoder_out is not None
                    encoder_padding_mask = self_attn_padding_mask.new_zeros(
                        encoder_out.size(1), encoder_out.size(0)
                    )
                self_attn_padding_mask = torch.cat(
                    (encoder_padding_mask, self_attn_padding_mask), dim=1
                )
            assert encoder_out is not None
            y = torch.cat((encoder_out, x), dim=0)
        else:
            y = x

        x, attn = self.self_attn(
            query=x,
            key=y,
            value=y,
            key_padding_mask=self_attn_padding_mask,
            incremental_state=incremental_state,
            need_weights=False,
            attn_mask=self_attn_mask,
            attn_bias=self_attn_bias
        )
        if self.self_attn_ln is not None:
            x = self.self_attn_ln(x)
        x = self.dropout_module(x)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        if self.encoder_attn is not None and encoder_out is not None:
            residual = x
            if self.normalize_before:
                x = self.encoder_attn_layer_norm(x)
            if prev_attn_state is not None:
                prev_key, prev_value = prev_attn_state[:2]
                saved_state: Dict[str, Optional[Tensor]] = {
                    "prev_key": prev_key,
                    "prev_value": prev_value,
                }
                if len(prev_attn_state) >= 3:
                    saved_state["prev_key_padding_mask"] = prev_attn_state[2]
                assert incremental_state is not None
                self.encoder_attn._set_input_buffer(incremental_state, saved_state)

            x, attn = self.encoder_attn(
                query=x,
                key=encoder_out,
                value=encoder_out,
                key_padding_mask=encoder_padding_mask,
                incremental_state=incremental_state,
                static_kv=True,
                need_weights=need_attn or (not self.training and self.need_attn),
                need_head_weights=need_head_weights,
                attn_bias=cross_attn_bias
            )
            if self.cross_attn_ln is not None:
                x = self.cross_attn_ln(x)
            x = self.dropout_module(x)
            x = self.residual_connection(x, residual)
            if not self.normalize_before:
                x = self.encoder_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)

        if not self.is_moe_layer or getattr(self.args, "alternate_decoder_ffn_embed_dim", 0.0) > 0:
            x = self.activation_fn(self.fc1(x))
            x = self.activation_dropout_module(x)
            if self.ffn_layernorm is not None:
                x = self.ffn_layernorm(x)
            x = self.fc2(x)
            x = self.dropout_module(x)
            l_aux = None
        else:
            # x - seq_len, batch_size, model_dim
            x = x.transpose(0, 1) # batch_size, seq_len, model_dim
            if getattr(self.args, "use_moe_pad_mask", False):
                x, l_aux = self.moe_layer(x, input_padding_mask=self_attn_padding_mask)
            else:
                x, l_aux = self.moe_layer(x)
            x = x.transpose(0, 1)

        if self.w_resid is not None:
            residual = torch.mul(self.w_resid, residual)
        x = self.residual_connection(x, residual)
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        if self.onnx_trace and incremental_state is not None:
            saved_state = self.self_attn._get_input_buffer(incremental_state)
            assert saved_state is not None
            if self_attn_padding_mask is not None:
                self_attn_state = [
                    saved_state["prev_key"],
                    saved_state["prev_value"],
                    saved_state["prev_key_padding_mask"],
                ]
            else:
                self_attn_state = [saved_state["prev_key"], saved_state["prev_value"]]
            return x, attn, self_attn_state
        return x, attn, None, l_aux

    def make_generation_fast_(self, need_attn: bool = False, **kwargs):
        self.need_attn = need_attn

    def doe_checkpoint_loader(self, name, state_dict, n_experts=4, noise_coef=5e-5, boise_coef=0.05, ln_coef=0.05):

        # get fc1_weights, fc2_weights, fc1_bias, fc2_bias
        fc1_weights = state_dict["{}.fc1.weight".format(name)]
        fc2_weights = state_dict["{}.fc2.weight".format(name)]
        fc1_bias = state_dict["{}.fc1.bias".format(name)]
        fc2_bias = state_dict["{}.fc2.bias".format(name)]
        ffn_weights = state_dict["{}.ffn_layernorm.weight".format(name)]
        ffn_bias = state_dict["{}.ffn_layernorm.bias".format(name)]

        # copy fc1, fc2 + noise
        state_dict["{}.doe.fc1.weight".format(name)] = fc1_weights + torch.randn(fc1_weights.size()) * noise_coef
        state_dict["{}.doe.fc2.weight".format(name)] = fc2_weights + torch.randn(fc2_weights.size()) * noise_coef
        state_dict["{}.doe.fc1.bias".format(name)] = fc1_bias
        state_dict["{}.doe.fc2.bias".format(name)] = fc2_bias
        state_dict["{}.doe.ffn_layernorm.weight".format(name)] = ffn_weights     
        state_dict["{}.doe.ffn_layernorm.bias".format(name)] = ffn_bias

    def moe_checkpoint_loader(self, name, state_dict, noise_coef=0):

        n_experts = self.args.moe_expert_count
        # get fc1_weights, fc2_weights, fc1_bias, fc2_bias
        fc1_weights = state_dict["{}.fc1.weight".format(name)]
        fc2_weights = state_dict["{}.fc2.weight".format(name)]
        fc1_bias = state_dict["{}.fc1.bias".format(name)]
        fc2_bias = state_dict["{}.fc2.bias".format(name)]
        ffn_weights = state_dict["{}.ffn_layernorm.weight".format(name)]
        ffn_bias = state_dict["{}.ffn_layernorm.bias".format(name)]

        # copy fc1, fc2 + noise
        for i in range(n_experts):
            state_dict["{}.moe_layer.experts.{}.fc1.weight".format(name, i)] = fc1_weights + torch.randn(fc1_weights.size()) * noise_coef
            state_dict["{}.moe_layer.experts.{}.fc2.weight".format(name, i)] = fc2_weights + torch.randn(fc2_weights.size()) * noise_coef
            state_dict["{}.moe_layer.experts.{}.fc1.bias".format(name, i)] = fc1_bias
            state_dict["{}.moe_layer.experts.{}.fc2.bias".format(name, i)] = fc2_bias
      
    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        # update layer norms
        layer_norm_map = {
            "0": "self_attn_layer_norm",
            "1": "encoder_attn_layer_norm",
            "2": "final_layer_norm",
        }
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict[
                        "{}.{}.{}".format(name, new, m)
                    ] = state_dict[k]
                    del state_dict[k]
                if "{}.{}.{}".format(name, new, m) not in state_dict and "{}.{}".format(new, m) in self.state_dict():
                    state_dict[
                        "{}.{}.{}".format(name, new, m)
                    ] = self.state_dict()["{}.{}".format(new, m)]

        moe_checker = "{}.moe_layer.gate.wg.weight".format(name)
        if moe_checker not in state_dict and "moe_layer.gate.wg.weight" in self.state_dict():
            self.moe_checkpoint_loader(name, state_dict)
        # if "{}.doe.fc1.weight".format(name) not in state_dict and "doe.fc1.weight" in self.state_dict():
        #     self.doe_checkpoint_loader(name, state_dict)

        prefix = name + "." if name != "" else ""
        for param_name, param_tensor in self.state_dict().items():
            if (prefix + param_name) not in state_dict and param_name in self.state_dict():
                state_dict[prefix + param_name] = self.state_dict()[param_name]

def make_experts(args, embed_dim, expert_ffn_dim, dropout_module) -> nn.ModuleList:
    world_size = 1 if not torch.distributed.is_initialized() else torch.distributed.get_world_size()
    expert_list = []
    ddp_rank = dist_utils.get_data_parallel_rank()
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert args.moe_expert_count % world_size == 0, f'{args.moe_expert_count}, {world_size}'
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with utils.set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    # less experts than gpus
    else:
        assert world_size % args.moe_expert_count == 0, f'{world_size}, {args.moe_expert_count}'
        # initialize each FFN with the same seed on different GPUs
        with utils.set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(FeedForwardNetwork(args, embed_dim, expert_ffn_dim, dropout_module))
    experts = nn.ModuleList(expert_list)
    return experts