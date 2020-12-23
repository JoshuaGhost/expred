#@title ACL IMDB
import tensorflow as tf
import os
import re
import pandas as pd


pattern = re.compile('</?(POS)?(NEG)?>')
# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = {}
    data["sentence"] = []
    for file_path in os.listdir(directory):
        with tf.gfile.GFile(os.path.join(directory, file_path), "r") as f:
            data["sentence"].append(f.read())
            re.sub(pattern, '', data['sentence'][-1])
    return pd.DataFrame.from_dict(data)

# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(directory):
    pos_df = load_directory_data(directory + "pos")
    neg_df = load_directory_data(directory + "neg")
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)

# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
    dataset = tf.keras.utils.get_file(
      fname="zaidan2007.zip",
      origin="https://www.dropbox.com/s/7b6logxi9o4nvio/zaidan2007annotation.zip?dl=1", 
      extract=True)

    df = load_dataset(os.path.join(os.path.dirname(dataset), 
                                   'zaidan2007annotation', 'withRats_'))
    df.drop_duplicates(keep=False)
    test = df.sample(frac=0.2)
    train = pd.merge(df, test, how='outer', indicator=True)
    train = df.loc[train._merge == 'left_only', ['sentence', 'polarity']]
    return train, test