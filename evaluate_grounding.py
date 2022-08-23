import glob
import re

import cv2
import torch
import json
# from matplotlib import pyplot as plt
import numpy as np
from fairseq import checkpoint_utils
from fairseq import distributed_utils, options, tasks, utils
from fairseq.dataclass.utils import convert_namespace_to_omegaconf
from PIL import Image
from tasks.mm_tasks.refcoco import RefcocoTask
from utils.eval_utils import eval_step


def main():
    tasks.register_task('refcoco', RefcocoTask)

    # turn on cuda if GPU is available
    use_cuda = torch.cuda.is_available()
    # use fp16 only when GPU is available
    use_fp16 = False

    # specify some options for evaluation
    parser = options.get_generation_parser()
    input_args = ["", "--task=refcoco", "--beam=100", "--unnormalized", "--path=checkpoints/vizwiz_base_best.pt",
                  "--bpe-dir=utils/BPE"]
    args = options.parse_args_and_arch(parser, input_args)
    cfg = convert_namespace_to_omegaconf(args)

    # Load pretrained ckpt & config
    task = tasks.setup_task(cfg.task)
    models, cfg = checkpoint_utils.load_model_ensemble(utils.split_paths(cfg.common_eval.path), task=task)

    # Move models to GPU
    for model in models:
        model.eval()
        if use_fp16:
            model.half()
        if use_cuda and not cfg.distributed_training.pipeline_model_parallel:
            model.cuda()
        model.prepare_for_inference_(cfg)

    # Initialize generator
    generator = task.build_generator(models, cfg.generation)

    # Image transform
    from torchvision import transforms
    mean = [0.5, 0.5, 0.5]
    std = [0.5, 0.5, 0.5]

    patch_resize_transform = transforms.Compose([
        lambda image: image.convert("RGB"),
        transforms.Resize((cfg.task.patch_image_size, cfg.task.patch_image_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    # Normalize the question
    def pre_question(question, max_ques_words):
        question = question.lower().lstrip(",.!?*#:;~").replace('-', ' ').replace('/', ' ')
        question = re.sub(r"\s{2,}", ' ', question, )
        question = question.rstrip('\n')
        question = question.strip(' ')
        # truncate question
        question_words = question.split(' ')
        if len(question_words) > max_ques_words:
            question = ' '.join(question_words[:max_ques_words])
        return question

    def encode_text(text, length=None, append_bos=False, append_eos=False):
        s = task.tgt_dict.encode_line(line=task.bpe.encode(text), add_if_not_exist=False, append_eos=False).long()
        if length is not None:
            s = s[:length]
        if append_bos:
            s = torch.cat([bos_item, s])
        if append_eos:
            s = torch.cat([s, eos_item])
        return s

    patch_image_size = cfg.task.patch_image_size

    # Construct input for open-domain VQA task
    def construct_sample(image: Image, text: str):
        w, h = image.size
        w_resize_ratio = torch.tensor(patch_image_size / w).unsqueeze(0)
        h_resize_ratio = torch.tensor(patch_image_size / h).unsqueeze(0)
        patch_image = patch_resize_transform(image).unsqueeze(0)
        patch_mask = torch.tensor([True])
        src_text = encode_text(' which region does the text " {} " describe?'.format(text), append_bos=True,
                               append_eos=True).unsqueeze(0)
        src_length = torch.LongTensor([s.ne(pad_idx).long().sum() for s in src_text])
        sample = {
            "id": np.array(['42']),
            "net_input": {
                "src_tokens": src_text,
                "src_lengths": src_length,
                "patch_images": patch_image,
                "patch_masks": patch_mask,
            },
            "w_resize_ratios": w_resize_ratio,
            "h_resize_ratios": h_resize_ratio,
            "region_coords": torch.randn(1, 4)
        }
        return sample

    # Function to turn FP32 to FP16
    def apply_half(t):
        if t.dtype is torch.float32:
            return t.to(dtype=torch.half)
        return t

    # Function to get bounding box
    def get_bounding(image: str):
        img = cv2.cvtColor(cv2.imread(image), cv2.COLOR_BGR2GRAY)
        x, y, w, h = cv2.boundingRect(img)
        # ret, thresh = cv2.threshold(img, 127, 255, 0)
        # contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        # rect = cv2.minAreaRect(contours[0])
        # (x, y), (w, h), a = rect
        # return (x, y), (w, h), a, rect
        return (x,y) ,(w,h)

    def coord_to_dict(x1,x2,y1,y2):
        return {'x1':x1,'x2':x2,'y1':y1,'y2':y2}
    def get_iou(bb1, bb2):
        """
        Calculate the Intersection over Union (IoU) of two bounding boxes.

        Parameters
        ----------
        bb1 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x1, y1) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner
        bb2 : dict
            Keys: {'x1', 'x2', 'y1', 'y2'}
            The (x, y) position is at the top left corner,
            the (x2, y2) position is at the bottom right corner

        Returns
        -------
        float
            in [0, 1]
        """
        assert bb1['x1'] < bb1['x2']
        assert bb1['y1'] < bb1['y2']
        assert bb2['x1'] < bb2['x2']
        assert bb2['y1'] < bb2['y2']

        # determine the coordinates of the intersection rectangle
        x_left = max(bb1['x1'], bb2['x1'])
        y_top = max(bb1['y1'], bb2['y1'])
        x_right = min(bb1['x2'], bb2['x2'])
        y_bottom = min(bb1['y2'], bb2['y2'])

        if x_right < x_left or y_bottom < y_top:
            return 0.0

        # The intersection of two axis-aligned bounding boxes is always an
        # axis-aligned bounding box
        intersection_area = (x_right - x_left) * (y_bottom - y_top)

        # compute the area of both AABBs
        bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
        bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        assert iou >= 0.0
        assert iou <= 1.0
        return iou

    def get_results(gt_box, pred_box, iou_thr):
        """Calculates number of true_pos, false_pos, false_neg from single batch of boxes.
        Args:
            gt_box(dict):
            {
            'image_id': {'x1', 'x2', 'y1', 'y2'},
            'image_id': ...
            }
            pred_box(dict): same as gt_box
            iou_thr (float): value of IoU to consider as threshold for a
                true prediction.
        Returns:
            dict: true positives (int), false positives (int), false negatives (int)
        """
        fp,tp,fn = [0]*3
        for idx in pred_box.keys():
            # Empty response from the classifier
            if len(pred_box[idx].values()) == 0:
                fn+=1
            # No object in the GT image
            elif len(gt_box[idx].values()) ==0:
                fp+=1
            else:
                iou = get_iou(gt_box[idx],pred_box[idx])

                if iou >= iou_thr:
                    tp+=1
                else:
                    fp+=1
        return {'true_pos': tp, 'false_pos': fp, 'false_neg': fn}

    def calc_precision_recall(result):
        """
        Calculate Precision-Recall for a given result

        Args:
            result(dict): {
            'true_pos': ,
            'false_pos': ,
            'false_neg':
            }
        """
        try:
            precision = result['true_pos'] / (result['true_pos'] + result['false_pos'])
        except ZeroDivisionError:
            precision = 0.0
        try:
            recall = result['true_pos'] / (result['true_pos'] + result['false_neg'])
        except ZeroDivisionError:
            recall = 0
        return precision,recall
    # Text preprocess
    bos_item = torch.LongTensor([task.src_dict.bos()])
    eos_item = torch.LongTensor([task.src_dict.eos()])
    pad_idx = task.src_dict.pad()

    annotations = json.loads(open('dataset/val_grounding.json', 'r').read())
    filename = [p.split('\\')[-1][:-4] for p in glob.glob('dataset/vizwiz_data/val/*.jpg')]
    iou_scores = []
    mask_coord_dict,output_coord_dict = {}, {}
    for f in filename:
        try:
            image = Image.open(f'dataset/vizwiz_data/val/{f}.jpg')
            mask = Image.open(f'dataset/vizwiz_data/binary/{f}.png')

            question = annotations[f'{f}.jpg']['question']
            (x,y),(w,h) = get_bounding(f'dataset/vizwiz_data/binary/{f}.png')
            mask_coord = coord_to_dict(x,x+w,y,y+h)
            # Construct input sample & preprocess for GPU if cuda available
            sample = construct_sample(image, question)
            sample = utils.move_to_cuda(sample) if use_cuda else sample
            sample = utils.apply_to_sample(apply_half, sample) if use_fp16 else sample

            # Run eval step for open-domain VQA
            with torch.no_grad():
                result, scores = eval_step(task, generator, models, sample)
            output_coord = coord_to_dict(int(result[0]["box"][0]),int(result[0]["box"][2]),int(result[0]["box"][1]),int(result[0]["box"][3]))
            iou_scores.append(get_iou(output_coord,mask_coord))
            image = np.asarray(mask)
            output_coord_dict[f] = output_coord
            mask_coord_dict[f] = mask_coord

            cv2.rectangle(
                image,
                (int(x),int(y)),
                (int(x+w),int(y+h)),
                (255, 255, 0),
                3
            )
            cv2.rectangle(
                image,
                (int(result[0]["box"][0]), int(result[0]["box"][1])),
                (int(result[0]["box"][2]), int(result[0]["box"][3])),
                (0, 255, 0),
                3
            )
            # print(f,get_iou(output_coord,mask_coord),mask_coord,output_coord)
            # plt.imshow(image)
            # plt.show()
        except:
            continue
    result_5 = get_results(mask_coord_dict,output_coord_dict,0.5)
    result_75 = get_results(mask_coord_dict, output_coord_dict, 0.75)
    result_9 = get_results(mask_coord_dict, output_coord_dict, 0.9)
    print(f'Average Precision @ 0.5 - {calc_precision_recall(result_5)}')
    print(f'Average Precision @ 0.75 - {calc_precision_recall(result_75)}')
    print(f'Average Precision @ 0.9 - {calc_precision_recall(result_9)}')
    print(f'Mean IOU Score - {sum(iou_scores)/len(iou_scores)}')
if __name__ == "__main__":
    main()
