import pandas as pd
# this is to extract the data from that .tgz file
import tarfile
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# get all of the data out of that .tgz
yelp_reviews = tarfile.open('data/yelp_review_polarity_csv.tgz')
yelp_reviews.extractall('data')
yelp_reviews.close()

# check out what the data looks like before you get started
# look at the training data set
train_df = pd.read_csv('data/yelp_review_polarity_csv/train.csv', header=None)
# print(train_df.head())

# look at the test data set
test_df = pd.read_csv('data/yelp_review_polarity_csv/test.csv', header=None)
# test_df = pd.read_csv('data/yelp_review_polarity_csv/testbert.csv', header=None)
# print(test_df.head())

train_df[0] = (train_df[0] == 2).astype(int)
test_df[0] = (test_df[0] == 2).astype(int)

print(train_df.head())
print(test_df.head())

bert_df = pd.DataFrame({
    'id': range(len(train_df)),
    'label': train_df[0],
    'alpha': ['q']*train_df.shape[0],
    'text': train_df[1].replace(r'\n', ' ', regex=True)
})

train_bert_df, dev_bert_df = train_test_split(bert_df, test_size=0.01)

print(train_bert_df.head())

test_bert_df = pd.DataFrame({
    'id': range(len(test_df)),
    'text': test_df[1].replace(r'\n', ' ', regex=True)
})

test_bert_df.head()

train_bert_df.to_csv('data/train.tsv', sep='\t', index=False, header=False)
dev_bert_df.to_csv('data/dev.tsv', sep='\t', index=False, header=False)
test_bert_df.to_csv('data/test.tsv', sep='\t', index=False, header=True)
# test_bert_df.to_csv('data/test2.tsv', sep='\t', index=False, header=False)


# python run_classifier.py --task_name=cola --do_predict=true --data_dir=./data --vocab_file=./multi_cased_L-12_H-768_A-12/vocab.txt --init_checkpoint=./multi_cased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --output_dir=./bert_output --bert_config_file=./multi_cased_L-12_H-768_A-12/bert_config.json --do_lower_case=False