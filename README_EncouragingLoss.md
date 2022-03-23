# Finetuning with Encouraging Loss (EL)
Below we provide methods for finetuning with label smoothed encouraging loss proposed in [_Well-classified Examples are Underestimated in Classification with Deep Neural Networks_](https://arxiv.org/pdf/2110.06537.pdf) on different downstream tasks.
The implementation is in [label_smoothed_encouraging_loss.py](criterions/label_smoothed_encouraging_loss.py).
You can set the `--criterion` to `adjust_label_smoothed_encouraging_loss` to use it. This criterion has a hyper-parameter `--log-end`. 
`--log-end < 1` results in a approximated and conservative version of the full encouraging loss. 
A high log_end will more strongly weaken the gradient vanishing, enhance the modeling of the data, and increase the growth rate of the margin, but it will also bring a larger gradient norm, which will bring challenges to the existing optimization system.
We recommend higher log_end for cases with higher performance, and 0.75 or 0.5 as your first try.
## Image Captioning
We provide procedures for image captioning with EL below. The preprocessing is identical to default setting.

<details>
    <summary><b>Finetuning</b></summary>
    <p>
        We propose two scripts for stage1. </b>
    </p>
<pre>
cd run_scripts/caption
nohup sh train_caption_stage1_el.sh > train_stage1_el.out &  # stage 1, train with encouraging loss, expected cider 1.403
nohup sh train_caption_stage1_el_db.sh > train_stage1_el.out &  # stage 1, train with encouraging loss, and drop best examples, expected cider 1.404
</pre>
</details>

## Referring Expression Comprehension 
We provide procedures for image captioning with EL below. The preprocessing is identical to default setting.
<details>
    <summary><b>Finetuning</b></summary>
<pre>
cd run_scripts/refcoco
nohup sh train_refcoco_el.sh > train_refcoco_el.out &  # finetune for refcoco
nohup sh train_refcocoplus_el.sh > train_refcocoplus_el.out &  # finetune for refcoco+
nohup sh train_refcocog_el.sh > train_refcocog_el.out &  # finetune for refcocog
</pre>
</details>
Evaluation is also the same as the default setting.
