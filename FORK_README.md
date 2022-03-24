# Fork repo readme

## 1. Dense model to Sparse: **Resmo Tuning**  
create a residual combination of mixture-of-experts block and dense-ffn.
### Follow Me:
#### Model Part  
1. Go to `models/ofa/unify_transformer_layers.py`  
2. Check and modify `class MoE_layer` to adjust the tutel configuration  
3. Go to `line 197,198`, commet off `self.doe = Dummy_layer(self.embed_dim, args.encoder_ffn_embed_dim, self.activation_fn, self.ffn_layernorm, self.activation_dropout_module, self.dropout_module, self.quant_noise, self.quant_noise_block_size)`, which is for Residual FFN (Pic3 Left), comment on `# self.moe = MoE_layer(self.embed_dim, args.encoder_ffn_embed_dim, None)`, which is for MoE  
4. Go to `moe_x = self.doe(x)` in **BOTH ENCODER AND DECODER**, replace `moe_x = self.doe(x)` with  `moe_x = self.moe(x)`  
5. Go to `def upgrade_state_dict_named(self, state_dict, name)` in **BOTH ENCODER AND DECODER**, comment of doe_stuff, comment on moe_stuff  
6. Modify `def moe_checkpoint_loader(self, name, state_dict, n_gpu, n_experts=2, noise_coef=0)` to adjust the noise_coef and also match the n_experts  
7. *For aux_loss, first comment on in **BOTH ENCODER AND DECODER**, and also do settings in the `models/ofa/unify_transformer.py` file from line 764 - 805 (Encoder), 1304 - 1363 (Decoder); Go to `criterions/label_smoothed_encouraging_loss.py` file line 315 - 316 do adjusting. aux_loss在valid的部分会报错，目前还没有解决，但不会影响能不能跑起来的测试。
#### Script Part  
1. Go to `run_scripts/snli_ve/tutel_train.sh`, modify batch_size, freq, lr, GPU, ...
2. Then run it.
### Progress:  
#### Done  
- OFA local tests on snli, glue: modified -> `run_scripts/*some.sh, dataset/*, checkpoints/*`. 
- MoE_Layer implement, based on tutel: modified -> `models/ofa/unify_transformer_layers.py`. 
- loading function: give same weights of ffn + noise to MoE-ffn from the backbone checkpoints.  
- work out the expriment part and visualization.   
#### Currently  
- Get myself a new glasses
- Fix Gradient Over Flow, MoE communication, aux_loss compute problem


