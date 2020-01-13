import tensorflow as tf
import pandas as pd
import os
import re
import json

pattern = re.compile('</?(POS)?(NEG)?>')

# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    for file_path in os.listdir(directory):
        with tf.io.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
    return pd.DataFrame.from_dict(data)

def annotated_doc(doc_fname, evidences, polarity):
    with open(doc_fname, 'r') as fin:
        doc_content = fin.readlines()
    doc_content = [l.strip() for l in doc_content]
    doc_content = ' '.join(doc_content).split()
    evidences_sorted = []
    if len(evidences) != 0:
        if len(evidences) == 1:
            evidences = evidences[0]
        else:
            evidences = [e[0] for e in evidences]            
        evidences_sorted = sorted(evidences, key=lambda x: x['start_token'])
        evi_texts = [e['text'] for e in evidences_sorted]
        evidences_sorted = [(e['start_token'], e['end_token']) for e in evidences_sorted]
    evidences_sorted.append((len(doc_content), -1))
    ret = doc_content[:evidences_sorted[0][0]]  
    tag = 'POS' if polarity == 1 else 'NEG'
    
    for i in range(len(evidences_sorted) - 1):
        ret.append('<{}>'.format(tag))
        evi = doc_content[evidences_sorted[i][0]: evidences_sorted[i][1]]
        try:
            assert ' '.join(evi) == evi_texts[i]
        except AssertionError:
            print(evi)
            print(evi_texts[i])
        ret += doc_content[evidences_sorted[i][0]: evidences_sorted[i][1]]
        ret.append('</{}>'.format(tag))
        ret += doc_content[evidences_sorted[i][1]: evidences_sorted[i+1][0]]
    return ' '.join(ret)
        
# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(docs_folder, dataset_fname):
    with open(dataset_fname, 'r') as fin:
        d = fin.readlines()
    ret = {'sentence': [], 'polarity': [], 'annotation_id':[]}
    for a in d:
        annotation = json.loads(a)
        ret['annotation_id'].append(annotation['annotation_id'])
        polarity = 1 if annotation['classification'] == 'POS' else 0
        doc_fname = os.path.join(docs_folder, ret['annotation_id'][-1])
        ret['sentence'].append(annotated_doc(doc_fname, annotation['evidences'], polarity))
        ret['polarity'].append(polarity)
    return pd.DataFrame(ret)
    #return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
      fname="movies.tar.gz",
      origin="http://www.eraserbenchmark.com/zipped/movies.tar.gz", 
      extract=True)
    dataset_folder = os.path.join(os.path.dirname(dataset), 'movies')
    docs_folder = os.path.join(dataset_folder, 'docs')
    df_train = load_dataset(docs_folder, os.path.join(dataset_folder, 'train.jsonl'))
    df_val = load_dataset(docs_folder, os.path.join(dataset_folder, 'val.jsonl'))
    df_test = load_dataset(docs_folder, os.path.join(dataset_folder, 'test.jsonl'))
    return df_train, df_val, df_test