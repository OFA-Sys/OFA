import pandas as pd
import _pickle as pickle

vqa_val = pd.read_csv('dataset/vqa_data/vqa_val.tsv', sep='\t')
vizwiz_val = pd.read_csv('dataset/vizwiz_data/vizwiz_val.tsv', sep='\t')
ans2label = pickle.load(open('dataset/vizwiz_data/trainval_ans2label.pkl', 'rb'))
print(ans2label)
