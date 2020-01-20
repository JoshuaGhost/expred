#!/usr/bin/env python
# coding: utf-8
# In[ ]:


# imports
from utils import *
from display_rational import convert_res_to_htmls
from config import *
from losses import imbalanced_bce_bayesian, imbalanced_bce_resampling, exp_interval_loss
from metrices import *
from tqdm import tqdm_notebook
from bert import optimization
from bert import run_classifier
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.backend import set_session
from tensorflow.keras import backend as K
import numpy as np
import pandas as pd
import tensorflow_hub as hub
import os
import pickle

from datetime import datetime

import bert

import tensorflow
if tensorflow.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf
    tf.disable_v2_behavior()
else:
    import tensorflow as tf


# The loss of the multi-task learning is:
# $$
# \mathcal{L}_{total} = \mathcal{L}_{cls} + \lambda\mathcal{L}_{exp}\text{,}
# $$
# where $\mathcal{L}_{cls}$ is the loss of the classification task, $\mathcal{L}_{exp}$ is the token-wise average of explaination loss (averaged cross entropy) and $S$ indicates the length of the input text. The hyper-parameter $\lambda$ is written as ```par_lambda``` in the following cells since 'lambda' is a reserved word in python.
# 
# The loss function has been changed since 2020.01.08. The old formulation of the loss function tried to balance between cls loss and exp loss, which down-scale both term as well as their learning rate. The new schema respect the global learning rate w.r.t. cls part while one can modify the learning rate of explaination part through $\lambda$

# In[ ]:


# variable hyper-parameters
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--par_lambda', type=float)
parser.add_argument('--gpu_id', type=str)
parser.add_argument('--batch_size', type=int)
parser.add_argument('--num_epochs', type=int)
parser.add_argument('--dataset', type=str, choices='fever multirc movies'.split()) # fever, multirc, movie
parser.add_argument("--do_train", action='store_true')
parser.add_argument('--exp_visualize', action='store_true')
parser.add_argument('--evaluate', action='store_true')
parser.add_argument('--exp_benchmark', action='store_true')
parser.add_argument('--exp_structure', type=str, default='gru', choices='gru rnr'.split()) # gru, rnr
parser.add_argument('--delete_checkpoints', action='store_true')
parser.add_argument('--merge_evidences', action='store_true')

args = ['--par_lambda', '0.01', 
        '--gpu_id', '0', 
        '--batch_size', '2', 
        '--num_epochs', '10',
        '--dataset', 'movies',
        '--do_train',
        '--evaluate',
        '--exp_benchmark',
        '--exp_structure', 'rnr',
        '--delete_checkpoints',
        '--merge_evidences']

args = parser.parse_args(args)
#args = parser.parse_args()

BATCH_SIZE = args.batch_size
par_lambda = args.par_lambda
NUM_EPOCHS = args.num_epochs
gpu_id = args.gpu_id
exp_structure = args.exp_structure
dataset = args.dataset
DO_DELETE = args.delete_checkpoints
do_train = args.do_train
load_best = not do_train
evaluate = args.evaluate
exp_visualize = args.exp_visualize
exp_benchmark = args.exp_benchmark
merge_evidences = args.merge_evidences

LEARNING_RATE = 1e-5


# In[ ]:


# static hyper-parameters
MAX_SEQ_LENGTH = 512
HARD_SELECTION_COUNT = None
HARD_SELECTION_THRESHOLD = 0.5


# In[ ]:


# Set the output directory for saving model file
# Optionally, set a GCP bucket location
from losses import rnr_matrix_loss


bert_size = 'base'
#bert_size = 'large'

if bert_size == 'base':
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

#pooling = 'mean'
pooling = 'first'

EXP_OUTPUT = exp_structure

rebalance_approach = 'resampling'
#rebalance_approach = 'bayesian'
if EXP_OUTPUT == 'gru':
    if rebalance_approach == 'resampling':
        loss_function = imbalanced_bce_resampling
    else:
        loss_function = imbalanced_bce_bayesian
elif EXP_OUTPUT == 'rnr':
    loss_function = rnr_matrix_loss
    
suffix = ''
#suffix = 'cls_only'
#suffix = 'transfer_cls_to_exp'
#suffix = 'transfer_exp_to_cls'

OUTPUT_DIR = ['bert_{}_seqlen_{}_{}_exp_output_{}'.format(
    bert_size, MAX_SEQ_LENGTH, dataset, EXP_OUTPUT)]
OUTPUT_DIR.append('merged_evidences' if merge_evidences else 'separated_evidences')
DATASET_CACHE_NAME = '_'.join(OUTPUT_DIR) + '_inputdata_cache'
if par_lambda is None:
    OUTPUT_DIR.append('no_weight')
else:
    OUTPUT_DIR.append('par_lambda_{}'.format(par_lambda))
OUTPUT_DIR.append('no_padding_imbalanced_bce_{}_pooling_{}_learning_rate_{}'.format(rebalance_approach, pooling, LEARNING_RATE))
OUTPUT_DIR.append(suffix)
OUTPUT_DIR = '_'.join(OUTPUT_DIR)
MODEL_NAME = OUTPUT_DIR
OUTPUT_DIR = os.path.join('model_checkpoints', MODEL_NAME)

# @markdown Whether or not to clear/delete the directory and create a new one
#DO_DELETE = False  # @param {type:"boolean"}
# @markdown Set USE_BUCKET and BUCKET if you want to (optionally) store model output on GCP bucket.
USE_BUCKET = False  # @param {type:"boolean"}
BUCKET = 'bert-base-uncased-test0'  # @param {type:"string"}

if USE_BUCKET:
    OUTPUT_DIR = 'gs://{}/{}'.format(BUCKET, OUTPUT_DIR)
    from google.colab import auth
    auth.authenticate_user()

if DO_DELETE:
    try:
        tf.gfile.DeleteRecursively(OUTPUT_DIR)
    except:
        # Doesn't matter if the directory didn't exist
        pass
if tensorflow.__version__.startswith('2'):
    tf.io.gfile.makedirs(OUTPUT_DIR)
else:
    tf.gfile.MakeDirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))


# In[ ]:


# data loading and preprocessing
'''
if dataset == 'semeval18':
    from load_data_semeval18 import download_and_load_datasets
    DATA_COLUMNS = 'Tweet text'
    LABEL_COLUMN = 'Label'
elif dataset == 'eraser_multirc':
    from load_data_eraser_multirc import download_and_load_datasets
    DATA_COLUMNS = ['query', 'passage']
    LABEL_COLUMN = 'classification'
elif dataset == 'eraser_fever':
    from load_data_eraser_fever import download_and_load_datasets
    DATA_COLUMNS = ['query', 'passage']
    LABEL_COLUMN = 'classification'
else:
    if dataset == 'acl_imdb' or dataset == 'acl_imdb_cls':
        from load_data_acl_imdb import download_and_load_datasets
    if dataset == 'semeval16' or dataset == 'semeval16_cls':
        from load_data_semeval16 import download_and_load_datasets
    if dataset == 'zaidan07_cls':
        from load_data_imdb_zaidan07_cls import download_and_load_datasets
    if dataset == 'zaidan07_seq' or dataset == 'zaidan07_mtl':
        from load_data_imdb_zaidan07_seq import download_and_load_datasets
    if dataset == 'eraser_movie_mtl':
        from load_data_imdb_zaidan07_eraser import download_and_load_datasets
    DATA_COLUMNS = ['sentence']
    LABEL_COLUMN = 'polarity'

ret = download_and_load_datasets()
if len(ret) == 3:
    train, val, test = ret
else:
    train, test = ret
    val = train.sample(frac=0.2)
    train = pd.merge(train, val, how='outer', indicator=True)
    train = train.loc[train._merge == 'left_only', ['sentence', 'polarity']]
label_list = [0, 1]

# data preprocessing
from bert_data_preprocessing_rational import load_bert_features, convert_bert_features


def preprocess(data, label_list, dataset_name):
    features = load_bert_features(
        data, label_list, MAX_SEQ_LENGTH, DATA_COLUMNS, LABEL_COLUMN)

    with_rations = ('cls' not in dataset_name)
    with_lable_id = ('seq' not in dataset_name)

    return convert_bert_features(features, with_lable_id, with_rations, EXP_OUTPUT)


@cache_decorator(os.path.join('cache', DATASET_NAME))
def preprocess_wrapper(*data_inputs):
    ret = []
    for data in data_inputs:
        ret.append(preprocess(data, label_list, dataset))
    return ret


rets_train, rets_val, rets_test = preprocess_wrapper(train, val, test)

train_input_ids, train_input_masks, train_segment_ids, train_rations, train_labels = rets_train
val_input_ids, val_input_masks, val_segment_ids, val_rations, val_labels = rets_val
test_input_ids, test_input_masks, test_segment_ids, test_rations, test_labels = rets_test
'''
pass


# In[ ]:


# initializing graph and session
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = gpu_id
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)
set_session(sess)


# In[ ]:


# data loading and preprocessing from eraser
from eraserbenchmark.rationale_benchmark.utils import load_datasets, load_documents
from eraserbenchmark.eraser_utils import extract_doc_ids_from_annotations
from itertools import chain

if dataset == 'movies':
    label_list = ['POS', 'NEG']
elif dataset == 'multirc':
    label_list = ['True', 'False']
elif dataset == 'fever':
    label_list = ['SUPPORTS', 'REFUTES']

data_dir = f'/home/zzhang/.keras/datasets/{dataset}/'
train, val, test = load_datasets(data_dir)
#train, val, test = [expand_on_evidences(data) for data in [train, val, test]]
docids = set(chain.from_iterable(extract_doc_ids_from_annotations(d) for d in [train, val, test]))
docs = load_documents(data_dir, docids)

from bert_data_preprocessing_rational_eraser import preprocess
@cache_decorator(os.path.join('cache', DATASET_CACHE_NAME + '_eraser_format'))
def preprocess_wrapper(*data_inputs, docs=docs):
    ret = []
    for data in data_inputs:
        ret.append(preprocess(data, docs, label_list, dataset, MAX_SEQ_LENGTH, EXP_OUTPUT, merge_evidences))
    return ret

rets_train, rets_val, rets_test = preprocess_wrapper(train, val, test, docs=docs)

train_input_ids, train_input_masks, train_segment_ids, train_rations, train_labels = rets_train
val_input_ids, val_input_masks, val_segment_ids, val_rations, val_labels = rets_val
test_input_ids, test_input_masks, test_segment_ids, test_rations, test_labels = rets_test

def expand_on_evidences(data):
    from copy import deepcopy
    from eraserbenchmark.rationale_benchmark.utils import Annotation
    expanded_data = []
    for ann in tqdm_notebook(data):
        for ev_group in ann.evidences:
            new_ann = Annotation(annotation_id=ann.annotation_id,
                                 query=ann.query,
                                 evidences=frozenset([ev_group]),
                                 classification=ann.classification)
            expanded_data.append(new_ann)
    return expanded_data
#expanded_test = expand_on_evidences(test)


# In[ ]:


# hyper-parameters of BERT's
WARMUP_PROPORTION = 0.1

num_train_steps = int(len(train_input_ids) / BATCH_SIZE * float(NUM_EPOCHS))
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)


# In[ ]:


from bert_utils import get_vocab
vocab = get_vocab(config)


# In[ ]:


# building models
from model import BertLayer
from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Multiply, Concatenate, RepeatVector, Dot, Lambda, Add

from metrices import sp_precision_wrapper, sp_recall_wrapper

DIM_DENSE_CLS = 256
NUM_GRU_UNITS_BERT_SEQ = 128
NUM_INTERVAL_LSTM_WIDTH = 100


def build_model(par_lambda=None):
    in_id = Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")
    in_mask = Input(shape=(MAX_SEQ_LENGTH,), name="input_masks")
    in_segment = Input(shape=(MAX_SEQ_LENGTH,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]

    bert_cls_output, bert_exp_output = BertLayer(
        n_fine_tune_layers=10)(bert_inputs)

    outputs = []
    if 'seq' not in dataset:
        # Classifier output
        dense = Dense(DIM_DENSE_CLS, activation='tanh')(bert_cls_output)
        cls = Dense(1, activation='sigmoid', name='cls_output')(dense)
        outputs.append(cls)
    if 'cls' not in dataset:
        # Explainer output
        if EXP_OUTPUT == 'gru':
            gru = CuDNNGRU(
                NUM_GRU_UNITS_BERT_SEQ, kernel_initializer='random_uniform', return_sequences=True)(bert_exp_output)
            exp = Dense(1, activation='sigmoid')(gru)
            output_mask = Reshape((512, 1))(in_mask)
            exp_outputs = Multiply(name='exp_output')([output_mask, exp])
        elif EXP_OUTPUT == 'rnr':
            M1 = Bidirectional(layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True),
                               merge_mode='concat')(bert_exp_output)
            p_starts = Dense(1, activation='sigmoid')(Concatenate(axis=-1)([bert_exp_output, M1]))

            m1_tilde = Dot(axes=-2)([p_starts, M1])
            M1_tilde = Lambda(lambda x: tf.tile(x, (1, MAX_SEQ_LENGTH, 1)))(m1_tilde)
            x = Multiply()([M1, M1_tilde])
            M2 = Bidirectional(layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True),
                               merge_mode='concat')(Concatenate(axis=-1)([bert_exp_output, M1, M1_tilde, x]))
            p_end_given_start = Dense(MAX_SEQ_LENGTH, activation='softmax')(Concatenate(axis=-1)([bert_exp_output, M2]))
            p_end_given_start = Lambda(lambda x: tf.linalg.band_part(x, 0, -1))(p_end_given_start)
            exp_outputs = Concatenate(axis=-1, name='exp_output')([p_starts, p_end_given_start])
            #exp_outputs = Lambda(lambda x: tf.reduce_sum(x, axis=-1, keepdims=True), name='exp_output')(p_dist)
        outputs.append(exp_outputs)

    model = Model(inputs=bert_inputs, outputs=outputs)
    optimizer = Adam(LEARNING_RATE)
    
    if par_lambda is None:
        loss_weights = None
    else:
        loss_weights = {'cls_output': 1,
                        'exp_output': par_lambda}
    metrics = {'cls_output': 'accuracy',
               'exp_output': [f1_wrapper(EXP_OUTPUT),
                              sp_precision_wrapper(EXP_OUTPUT),
                              sp_recall_wrapper(EXP_OUTPUT),
                              precision_wrapper(EXP_OUTPUT),
                              recall_wrapper(EXP_OUTPUT)]}
    loss = {'cls_output': 'binary_crossentropy',
            'exp_output': loss_function()}
    '''
    metrics = [f1_wrapper(EXP_OUTPUT),
                              sp_precision_wrapper(EXP_OUTPUT),
                              sp_recall_wrapper(EXP_OUTPUT),
                              precision_wrapper(EXP_OUTPUT),
                              recall_wrapper(EXP_OUTPUT)]
    loss = loss_function()
    '''
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model_exp = Model(inputs=bert_inputs, outputs=exp_outputs)
    optimizer = Adam(LEARNING_RATE)
    model_exp.compile(loss=loss_function(), optimizer=optimizer)

    return model, model_exp


# In[ ]:


# training, evaluation and inference
import os
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

checkpoint_path = os.path.join(OUTPUT_DIR, 'cp-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)
cls_output_file = os.path.join(OUTPUT_DIR, 'output.txt')

BENCHMARK_SPLIT_NAME = 'test'
RES_FOR_BENCHMARK_FNAME = MODEL_NAME + '_' + BENCHMARK_SPLIT_NAME

with graph.as_default():
    set_session(sess)
    model, model_exp = build_model(par_lambda)
    model.summary()
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())

    training_inputs = [train_input_ids, train_input_masks, train_segment_ids]
    # training_inputs = [train_input_ids[:10],
    #                   train_input_masks[:10], train_segment_ids[:10]]
    val_inputs = [val_input_ids, val_input_masks, val_segment_ids]
    test_inputs = [test_input_ids, test_input_masks, test_segment_ids]

    training_outputs, test_outputs, val_outputs = {}, {}, {}

    if 'seq' not in dataset:
        training_outputs['cls_output'] = train_labels
        #training_outputs['cls_output'] = train_labels[:10]
        test_outputs['cls_output'] = test_labels
        val_outputs['cls_output'] = val_labels
    if 'cls' not in dataset:
        training_outputs['exp_output'] = train_rations
        #training_outputs['exp_output'] = train_rations[:10]
        test_outputs['exp_output'] = test_rations
        val_outputs['exp_output'] = val_rations

    initial_epoch = 0
    if load_best:
        if dataset == 'fever':
            best_epoch = 3
        else:
            with open(cls_output_file, 'r') as fin:
                log = fin.readlines()
            history = eval(log[2])
            best_epoch = np.argmin(history['loss'])+1
        model.load_weights(checkpoint_path.format(epoch=best_epoch))
    else:
        for ckpt_i in range(NUM_EPOCHS, 0, -1):
            if os.path.isfile(checkpoint_path.format(epoch=ckpt_i)):
                initial_epoch = ckpt_i
                model.load_weights(checkpoint_path.format(epoch=ckpt_i))
                # assert False  # dumm proof, most of the case the training is end-to-end, without disturbance and reloading
                break

    if do_train:
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=False,
                                                         verbose=1,
                                                         period=1)
        es_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, restore_best_weights=True)

        with open(cls_output_file, 'a+') as fw:
            fw.write("=============== {} ===============\n".format(datetime.now()))

        history = model.fit(
            training_inputs,
            training_outputs,

            validation_data=(val_inputs,
                             val_outputs),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[cp_callback, es_callback],
            initial_epoch=initial_epoch
        )

        with open(cls_output_file, 'a+') as fw:
            fw.write("{}:\n".format(datetime.now()))
            fw.write(str(history.history) + '\n')

    if evaluate:
        evaluation_res = model.evaluate(x=test_inputs,
                                        y=test_outputs,
                                        batch_size=BATCH_SIZE,
                                        verbose=1)
        with open(cls_output_file, 'a+') as fw:
            fw.write("{}:\n".format(datetime.now()))
            fw.write(str(evaluation_res) + '\n')

    if exp_visualize:
        len_head = 100
        test_inputs_head = [x[:len_head] for x in test_inputs]
        pred = model_exp.predict(test_inputs_head)
        pred = np.round(np.array(pred)).astype(np.int32)
        exp_output_folder = os.path.join(OUTPUT_DIR, 'exp_outputs/')
        tf.gfile.MakeDirs(exp_output_folder)
        print('marked rationals are saved under {}'.format(exp_output_folder))
        vocab = get_vocab(BERT_MODEL_HUB)
        for i, l in enumerate(tqdm_notebook(test.iterrows())):
            if i == len_head:
                break
            input_ids = test_input_ids[i]
            pred_intp = pred[i].reshape([-1])
            label = test_labels[i]
            gt = test_rations[i].reshape([-1])
            html = convert_res_to_htmls(input_ids, pred_intp, gt, vocab)
            with open(exp_output_folder + str(l[0]) + '.html', "w+") as f:
                f.write(
                    '<h1>label: {}</h1>\n'.format('pos' if label == 1 else 'neg'))
                f.write(html[1] + '<br/><br/>\n' + html[0])

    if exp_benchmark:
        from eraser_benchmark import pred_to_results
        result_fname = RES_FOR_BENCHMARK_FNAME + '.jsonl'
        result_fname = os.path.join(
            'eraserbenchmark', 'annotated_by_exp', result_fname)

        if BENCHMARK_SPLIT_NAME == 'test':
            benchmark_inputs, raw_input, benchmark_input_ids = test_inputs, test, test_input_ids
        elif BENCHMARK_SPLIT_NAME == 'val':
            benchmark_inputs, raw_input, benchmark_input_ids = val_inputs, val, val_input_ids
        elif BENCHMARK_SPLIT_NAME == 'train':
            benchmark_inputs, raw_input, benchmark_input_ids = training_inputs, train, train_input_ids

        pred = model.predict(x=benchmark_inputs)

        # results = [pred_to_results(raw_input.loc[i], benchmark_input_ids[i], (pred[0][i], pred[1][i]), HARD_SELECTION_COUNT, HARD_SELECTION_THRESHOLD)
        #           for i in range(len(raw_input))]
        from eraser_benchmark import pred_to_results, get_cls_score, add_cls_scores, remove_rations, extract_rations
        results = [pred_to_results(raw_input[i], benchmark_input_ids[i], 
                                   (pred[0][i], pred[1][i]), 
                                   HARD_SELECTION_COUNT, 
                                   HARD_SELECTION_THRESHOLD,
                                   vocab, docs, label_list, EXP_OUTPUT)
                   for i in range(len(pred[0]))]
        pred_softmax = np.hstack([1-pred[0], pred[0]])
        c_pred_softmax = get_cls_score(
            model, results, docs, label_list, dataset, remove_rations, MAX_SEQ_LENGTH, EXP_OUTPUT)
        s_pred_softmax = get_cls_score(
            model, results, docs, label_list, dataset, extract_rations, MAX_SEQ_LENGTH, EXP_OUTPUT)

        results = [add_cls_scores(res, cls_score, c_cls_score, s_cls_score, label_list) for res, cls_score,
                   c_cls_score, s_cls_score in zip(results, pred_softmax, c_pred_softmax, s_pred_softmax)]
        anns_saved = set()
        real_results = []
        for ann in results:
            if ann['annotation_id'] not in anns_saved:
                anns_saved.add(ann['annotation_id'])
                real_results.append(ann)
        with open(result_fname+'pkl3', "wb+") as pfout:
            pickle.dump(real_results, pfout)
        from eraserbenchmark.eraser import evaluate
        evaluate(MODEL_NAME, dataset)

    with open(cls_output_file, 'a+') as fw:
        fw.write('/////////////////experiment ends//////////////////\n\n\n')

