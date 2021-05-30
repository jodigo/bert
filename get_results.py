import fire
import pandas as pd

def get_results(outcome):
    scores = pd.read_csv(f"./bert_output/test_results2.tsv", sep="\t", header=None)
    scores.columns = ['negative', 'positive']
    scores['id'] = range(1, len(scores) + 1)

    texts = pd.read_csv(f'./data/test2.tsv', sep="\t")
    texts.columns = ['id', 'text']
    d = pd.merge(texts, scores, on='id', how='outer')
    print(d)

    d.to_csv('./merged_result_text.csv', sep='\t', index=False, header=False)

if __name__ == '__main__':
  fire.Fire(get_results)