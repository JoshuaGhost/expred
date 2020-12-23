import tensorflow as tf
import os
import re


def load_dataset(file_path):
    with open(file_path, 'r') as f:
        df = pd.read_csv(f, index_col='Tweet index', sep='\t')
    return df

def download_and_load_datasets(force_download=False):
    dataset_train = tf.keras.utils.get_file(
        fname='SemEval2018-T3-train-taskA.txt',
        origin='https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/train/SemEval2018-T3-train-taskA.txt',
        extract=False)
    dataset_test = tf.keras.utils.get_file(
        fname='SemEval2018-T3_input_test_taskA.txt',
        origin='https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/test_TaskA/SemEval2018-T3_input_test_taskA.txt',
        extract=False)
    dataset_golden = tf.keras.utils.get_file(
        fname='SemEval2018-T3_gold_test_taskA_emoji.txt',
        origin='https://raw.githubusercontent.com/Cyvhee/SemEval2018-Task3/master/datasets/goldtest_TaskA/SemEval2018-T3_gold_test_taskA_emoji.txt',
        extract=False)
    
    with open(dataset_train, 'r') as f:
        train_df = pd.read_csv(f, index_col='Tweet index', sep='\t')
    with open(dataset_test, 'r') as f:
        test_df = pd.read_csv(f, index_col='tweet index', sep='\t')
    with open(dataset_golden, 'r') as f:
        golden_df = pd.read_csv(f, index_col='Tweet index', sep='\t')[['Label',]]
    test_df = test_df.merge(golden_df, left_index=True, right_index=True).rename(columns={'tweet text': 'Tweet text'})
    return train_df, test_df