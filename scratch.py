import pandas as pd

train_tsv = './dataset/vizwiz_data/vizwiz_train.tsv'
val_tsv = './dataset/vizwiz_data/vizwiz_val.tsv'
train_df = pd.read_csv(open(train_tsv), sep='\t')
val_df = pd.read_csv(open(val_tsv), sep='\t')
print("")
