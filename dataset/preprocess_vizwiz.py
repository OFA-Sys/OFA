"""
A script to transform the VizWiz dataset into the format of the VQA dataset.
The format is a csv file with the following columns:
question-id, image-id, question text (lowercase), answer with confidence (like 1.0|!+no),
object label (can be blank), image as base64 encoded string
For all-candidate inference, a trainval_ans2label.pkl file has to be generated,
which is just a pickled mapping of answers to their label; not necessary for beam-search inference.

Author: Jan Willruth
"""

import base64
import glob
import json
import sys
import pandas as pd
from io import BytesIO
from PIL import Image
from tqdm import tqdm


def img2base64(fn):
    """
    Convert an image to base64 encoded string. Code from a maintainer of the OFA repo
    (https://github.com/OFA-Sys/OFA/issues/56).

    :param fn: filename of image
    :return: base64 encoded string
    """

    img = Image.open(fn)
    img_buffer = BytesIO()
    img.save(img_buffer, format=img.format)
    byte_data = img_buffer.getvalue()
    return base64.b64encode(byte_data).decode('utf-8')


def main():
    # Dict to map answer confidence to value
    conf = {'yes': '1.0', 'maybe': '0.5', 'no': '0.0'}
    # Iterate over subsets
    for subset in ['train', 'val', 'test']:
        print(f'Creating tsv file for {subset} subset...')
        # Load corresponding json file
        annotations = json.load(open(f'vizwiz_data/Annotations/{subset}.json', encoding='utf-8'))
        # Create empty set to store data
        tsv_set = set()
        # Iterate over all images in subset
        for fn in tqdm(glob.glob(f'vizwiz_data/{subset}/*.jpg'), file=sys.stdout):
            # Some string manipulation to get img_id
            fn = fn.replace('\\', '/')
            img_id = int(fn.split('/')[-1].split('_')[-1][3:-4])
            # Get corresponding question
            try:
                question = annotations[img_id]['question'].lower()
            except IndexError:
                continue
            # If test subset, use placeholder answer, else iterate over all questions
            if subset == 'test':
                tsv_set.add((img_id, img_id, question, '1.0|!+no', '', img2base64(fn)))
            else:
                for ans in annotations[img_id]['answers']:
                    ans_conf = f'{conf[ans["answer_confidence"]]}|!+{ans["answer"]}'
                    tsv_set.add((img_id, img_id, question, ans_conf, '', img2base64(fn)))
        # Write to tsv file
        with open(f'vizwiz_data/vizwiz_{subset}.tsv', 'w', encoding='utf-8') as f:
            for line in tsv_set:
                f.write('\t'.join(map(str, line)) + '\n')
    return 'All done!'


if __name__ == '__main__':
    main()
