import csv
from io import BytesIO
import os
import base64
from tqdm import tqdm
import h5py
import json
from PIL import Image
import numpy as np

data_dir = '/data/hulab/zcai75/OFA_data/vg'
vg_dir = '/data/hulab/zcai75/visual_genome'
image_dir = os.path.join(vg_dir, 'VG_100K')
toy = False
version = 'toy' if toy else 'full'
toy_count = 1000

if not os.path.exists(data_dir):
	os.makedirs(data_dir)

with h5py.File(os.path.join(vg_dir, 'VG-SGG-with-attri.h5'), 'r') as f, \
	 open(os.path.join(vg_dir, 'VG-SGG-dicts-with-attri.json'), 'r') as d, \
	 open(os.path.join(vg_dir, 'image_data.json')) as img_data:
	d = json.load(d)
	img_data = json.load(img_data)
	print(f.keys())
	# print(f['boxes_1024'][0])
	with open(os.path.join(data_dir, f'vg_train_{version}.tsv'), 'w+', newline='\n') as f_train, \
		 open(os.path.join(data_dir, f'vg_val_{version}.tsv'), 'w+', newline='\n') as f_val:
		writer_train = csv.writer(f_train, delimiter='\t', lineterminator='\n')
		writer_val = csv.writer(f_val, delimiter='\t', lineterminator='\n')

		assert len(f['img_to_first_rel']) == len(f['img_to_last_rel']) == len(f['img_to_first_box']) == len(f['img_to_last_box']) == len(f['split'])
		data = enumerate(zip(
			f['img_to_first_rel'], f['img_to_last_rel'],
			f['img_to_first_box'], f['img_to_last_box'],
			f['split']))
		tqdm_obj = tqdm(data, total=len(f['split']))

		train_count = 0
		val_count = 0
		skip_count = 0
		rel_count = 0
		for i, (first_rel, last_rel, first_box, last_box, split) in tqdm_obj:
			if toy and ((train_count > toy_count and split == 0) or (val_count > toy_count and split != 0)):
				continue
			try:
				if first_rel < 0 or last_rel < 0 or last_rel - first_rel < 0:
					skip_count += 1
					continue
				rel_count += last_rel - first_rel + 1

				image_id = img_data[i]['image_id']
				with Image.open(os.path.join(image_dir, f'{image_id}.jpg'), 'r') as img_f:
					img_rels = (f['relationships'][first_rel : last_rel+1] - first_box).tolist()

					pred_ids = np.atleast_1d(f['predicates'][first_rel : last_rel+1].squeeze()).tolist()
					boxes = np.atleast_2d(f['boxes_1024'][first_box : last_box+1].squeeze()).tolist()
					boxes = [list(map(round, [box[0]-box[2]/2, box[1]-box[3]/2, box[0]+box[2]/2, box[1]+box[3]/2])) for box in boxes]
					assert all([box[0] >= 0 and box[1] >= 0 and box[2] <= 1024 and box[3] <= 1024 for box in boxes]), boxes
					assert len(pred_ids) == len(img_rels) and len(pred_ids) > 0, (first_rel, last_rel, pred_ids, img_rels)
					box_ids = np.atleast_1d(f['labels'][first_box : last_box+1].squeeze()).tolist()
					pred_label = [d['idx_to_predicate'][str(i)] for i in pred_ids]
					box_label = [d['idx_to_label'][str(i)] for i in box_ids]

					assert len(pred_ids) == len(pred_label) == len(img_rels)
					assert len(boxes) == len(box_ids) == len(box_label)

					buf = BytesIO()
					img_f.save(buf, format='jpeg')
					buf.seek(0)
					img_str = base64.urlsafe_b64encode(buf.read()).decode('utf-8')

					row = [image_id, ','.join(map(str, pred_ids)), ','.join(map(str, box_ids)), ','.join([' '.join(map(str, rel)) for rel in img_rels]), ','.join([' '.join(map(str, box)) for box in boxes]), ','.join(pred_label), ','.join(box_label), img_str]
					# print(row[:-1])
					if split == 0:
						if toy and train_count > toy_count:
							continue
						writer_train.writerow(row)
						train_count += 1
					else:
						if toy and val_count > toy_count:
							continue
						writer_val.writerow(row)
						val_count += 1
			except FileNotFoundError:
				print('Cannot find ' + f'{image_id}.jpg')
			# break
		print('Train:', train_count, 'Val:', val_count, 'Skipped:', skip_count)
		assert rel_count == len(f['relationships']), (rel_count, len(f['relationships']))
