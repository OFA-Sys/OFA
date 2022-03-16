## Fork repo readme

1. dense(pre-trained) model to sparse: Resmo Tuning  
create a residual combination of mixture-of-experts block and dense-ffn  
Progress:  
[Done]  - OFA local tests on snli, glue: modified -> run_scripts/*some.sh, dataset/*, checkpoints/* .  
[Currently]  - MoE_Layer implement, based on tutel: modified -> models/ofa/unify_transformer_layers.py. 
	  - loading function: give same weights of ffn + noise to MoE-ffn from the backbone checkpoints. 
	  - work out the expriment part and visualization. 
