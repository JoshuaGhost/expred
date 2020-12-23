import tensorflow as tf
import os
import re
import pandas as pd

def load_dataset(file_path):
    data = {}
    data["sentence"] = []
    data["polarity"] = []
    with tf.gfile.GFile(file_path, "r") as f:
        for l in f.readlines():
            ls = l.strip()
            ls = l.split('\t', 2)
            if len(ls) < 3:
                continue
            if ls[1] == 'neutral':
                continue
            data["sentence"].append(ls[2])
            if ls[1] == 'positive':
                data['polarity'].append(1)
            else:
                data['polarity'].append(0)
    return pd.DataFrame.from_dict(data)

def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
        fname='semeval17.zip',
        origin='https://www.dropbox.com/s/byzr8yoda6bua1b/2017_English_final.zip?dl=1',
        extract=True)
    train_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                        '2017_English_final', 'GOLD', 'Subtask_A', 'twitter-2016train-A.txt'))
    dev_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      '2017_English_final', 'GOLD', 'Subtask_A', 'twitter-2016devtest-A.txt'))
    test_df = load_dataset(os.path.join(os.path.dirname(dataset),
                                      '2017_English_final', 'GOLD', 'Subtask_A', 'twitter-2016test-A.txt'))
    return pd.concat([train_df, dev_df]).sample(frac=1).reset_index(drop=True), test_df.sample(frac=1).reset_index(drop=True)
   