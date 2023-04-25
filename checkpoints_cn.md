# Checkpoints (OFA-CN)

We provide checkpoints of OFA-CN, which is the Chinese version of OFA. We provide Base-size and Large-size models, including pretrained and finetuned models on image captioning and referring expression comprehension. Note that we translated the texts in the RefCOCO(-/+/g) datasets and finetuned OFA-CN on them. We plan to release the related new datasets in the near future. 
<br>

## Checkpoints
Below we provide the links for downloading the Chinese OFA checkpoints.

### Pretraining
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_cn_large.pt"> Pretrained checkpoint (OFA-CN-Large) </a> (~443M parameters)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_cn_base.pt "> Pretrained checkpoint (OFA-CN-Base) </a> (~160M parameters)

### Finetuning (OFA-Large)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_cn_large.pt"> Finetuned checkpoint for MUGE Caption (Stage 1) </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcoco_cn_large.pt"> Finetuned checkpoint for RefCOCO-CN </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocoplus_cn_large.pt"> Finetuned checkpoint for RefCOCO+-CN </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocog_cn_large.pt"> Finetuned checkpoint for RefCOCOg-CN </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_cn_ocr_large.pt"> Finetuned checkpoint for Chinese OCR (multitask finetuned)</a>

### Finetuning (OFA-Base)
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/caption_cn_base.pt"> Finetuned checkpoint for MUGE Caption (Stage 1) </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcoco_cn_base.pt"> Finetuned checkpoint for RefCOCO-CN </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocoplus_cn_base.pt"> Finetuned checkpoint for RefCOCO+-CN </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/refcocog_cn_base.pt"> Finetuned checkpoint for RefCOCOg-CN </a>
* <a href="https://ofa-beijing.oss-cn-beijing.aliyuncs.com/checkpoints/ofa_cn_ocr_base.pt"> Finetuned checkpoint for Chinese OCR (multitask finetuned) </a>
<br>

## Model Card
Below we provide the basic information of the base-size and large-size OFA-CN. 

<table border="1" width="100%">
    <tr align="center">
        <th>Model</th><th>#Params</th><th>Backbone</th><th>Hidden Size</th><th>Intermediate Size</th><th>#Heads</th><th>#Enc. Layers</th><th>#Dec. Layers</th>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub><td>160M</td><td>ResNet101</td><td>768</td></td><td>3072</td><td>12</td><td>6</td><td>6</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td>443M</td><td>ResNet152</td><td>1024</td></td><td>4096</td><td>16</td><td>12</td><td>12</td>
    </tr>
    </tr>
</table>
<br>

## Results
Below we provide the results of OFA-CN and the baselines for comparison. 

### [MUGE Caption]("https://tianchi.aliyun.com/muge")
<table border="1" width="100%">
    <tr align="center">
        <td>Model</td><td>BLEU@4</td><td>ROUGE-L</td><td>CIDEr-D</td>
    </tr>
    <tr align="center">
        <td>Trm </td><td>7.33</td><td>51.51</td><td>11.00</td>
    </tr>
    <tr align="center">
        <td>M6</td><td>16.19</td><td>55.06</td><td>30.75</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>26.23</td><td>58.95</td><td>50.70</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td><b>27.32</b></td><td><b>59.20</b></td><td><b>53.51</b></td>
    </tr>
</table>

### RefCOCO-CN Series
<table border="1" width="100%">
    <tr align="center">
        <td>Model</td><td>RefCOCO(val/testA/testB)</td><td>RefCOCO+(val/testA/testB)</td><td>RefCOCOg(val/test-u)</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub>(random-init)</td><td>30.13/35.07/25.03</td><td>17.89/20.90/15.83</td><td>20.30/20.45</td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Base</sub></td><td>82.18/86.07/<b>76.68</b></td><td>69.38/77.26/60.14</td><td><b>73.57/72.53</b></td>
    </tr>
    <tr align="center">
        <td>OFA<sub>Large</sub></td><td><b>82.84/86.54</b>/76.50</td><td><b>71.30/78.56/61.85</b></td><td>71.96/71.30</td>
    </tr>
</table>
<br>


