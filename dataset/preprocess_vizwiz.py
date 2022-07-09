"""
A script to transform the VizWiz dataset into the format of the VQA dataset.
The format is a csv file with the following columns:
question-id, image-id, question text (lowercase), answer with confidence (like 1.0|!+no),
object label (can be blank), image as base64 encoded string
Further, a trainval_ans2label2.pkl file has to be generated which is simply a pickled dict,
mapping the most frequent answers to a (random) label.

Author: Jan Willruth
"""

import base64
import glob
import json
import _pickle as pickle
import sys
from io import BytesIO
from collections import Counter
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
    """print('Generating trainval_ans2label2.pkl file...')
    # Load annotations
    train = json.load(open('vizwiz_data/Annotations/train.json', encoding='utf-8'))
    val = json.load(open('vizwiz_data/Annotations/val.json', encoding='utf-8'))
    annotations = train + val
    # Extract answers
    answers = [ans['answer'] for question in annotations for ans in question['answers']]
    # Count occurrences of answers
    answer_counts = {}
    for ca in answers:
        if ca not in answer_counts:
            answer_counts[ca] = 1
        else:
            answer_counts[ca] += 1
    # Pick top x most frequent answers (VQA uses 3129)
    x = 3129
    freq_answers = sorted(answer_counts, key=answer_counts.get, reverse=True)[:x]
    # Create dict to map answers to labels
    trainval_ans2label = {answer: i for i, answer in enumerate(freq_answers)}
    # Save to file
    with open('vizwiz_data/trainval_ans2label2.pkl', 'wb') as f:
        pickle.dump(trainval_ans2label, f)

    print('Finished generating trainval_ans2label2.pkl file...')"""

    # Dict to map answer confidence to value
    conf = {'yes': '1.0', 'maybe': '0.5', 'no': '0.0'}
    # Iterate over subsets
    for subset in ['train', 'val', 'test']:
        print(f'Generating rows for {subset} tsv file...')
        # Load corresponding json file
        annotations = json.load(open(f'vizwiz_data/Annotations/{subset}.json', encoding='utf-8'))
        if subset == 'train':
            annotations += json.load(open('vizwiz_data/Annotations/val.json', encoding='utf-8'))
        # Create empty set to store data
        tsv_set = set()
        # Iterate over all images in subset
        file_names = glob.glob(f'vizwiz_data/{subset}/*.jpg')
        for fn in tqdm(file_names, file=sys.stdout):
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
                answers = [ca['answer'] for ca in annotations[img_id]['answers']]
                conf, ans = 1.0, Counter(answers).most_common(1)[0][0]
                ans_conf = f'{conf}|!+{ans}&&'
                """ans_conf = ''
                conf_ans = annotations[img_id]['answers']
                for ca in conf_ans:
                    ans_conf += f'{conf[ca["answer_confidence"]]}|!+{ca["answer"]}&&'"""
                tsv_set.add((img_id, img_id, question, ans_conf[:-2], '', img2base64(fn)))
        # Write to tsv file
        print(f'Writing {subset} tsv file...')
        with open(f'vizwiz_data/vizwiz_{subset}.tsv', 'w', encoding='utf-8') as f:
            for line in tsv_set:
                f.write('\t'.join(map(str, line)) + '\n')
    return 'Finished creating tsv files!'


if __name__ == '__main__':
    print(main())
    print('All done!')
