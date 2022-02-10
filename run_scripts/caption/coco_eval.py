import json
import sys
import os.path as op

from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def evaluate_on_coco_caption(res_file, label_file, outfile=None):
    """
    res_file: txt file, each row is [image_key, json format list of captions].
             Each caption is a dict, with fields "caption", "conf".
    label_file: JSON file of ground truth captions in COCO format.
    """
    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file)
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, 'w') as fp:
            json.dump(result, fp, indent=4)
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 4:
        evaluate_on_coco_caption(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        raise NotImplementedError