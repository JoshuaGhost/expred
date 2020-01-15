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

#import tensorflow.compat.v1 as tf
# tf.disable_v2_behavior()
import tensorflow as tf

# hyper-parameters
MAX_SEQ_LENGTH = 512
HARD_SELECTION_COUNT = None
HARD_SELECTION_THRESHOLD = 0.5

BATCH_SIZE = 2   # for jupyter-notebook
# BATCH_SIZE = 16  # for stand-alone python script

#par_lambda = None
#par_lambda = 1e-5
par_lambda = 1

NUM_EPOCHS = 10

LEARNING_RATE = 1e-5

# Set the output directory for saving model file
# Optionally, set a GCP bucket location

#dataset = 'eraser_movie_mtl'
#dataset = 'eraser_multirc'
dataset = 'fever'

rebalance_approach = 'resampling'
#rebalance_approach = 'bayesian'

bert_size = 'base'
#bert_size = 'large'

if bert_size == 'base':
    BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

#pooling = 'mean'
pooling = 'first'

if rebalance_approach == 'resampling':
    loss_function = imbalanced_bce_resampling
else:
    loss_function = imbalanced_bce_bayesian

EXP_OUTPUT = 'gru'
#EXP_OUTPUT = 'interval'

suffix = ''
#suffix = 'cls_only'
#suffix = 'transfer_cls_to_exp'
#suffix = 'transfer_exp_to_cls'

OUTPUT_DIR = ['bert_{}_seqlen_{}_{}_exp_output_{}'.format(
    bert_size, MAX_SEQ_LENGTH, dataset, EXP_OUTPUT)]
DATASET_NAME = '_'.join(OUTPUT_DIR) + '_inputdata_cache'
if par_lambda is None:
    OUTPUT_DIR.append('no_weight')
else:
    OUTPUT_DIR.append('par_lambda_{}'.format(par_lambda))
OUTPUT_DIR.append('no_padding_imbalanced_bce_{}_pooling_{}_learning_rate_{}'.format(
    rebalance_approach, pooling, LEARNING_RATE))
OUTPUT_DIR.append(suffix)

OUTPUT_DIR = '_'.join(OUTPUT_DIR)
MODEL_NAME = OUTPUT_DIR
OUTPUT_DIR = os.path.join('model_checkpoints', MODEL_NAME)

# @markdown Whether or not to clear/delete the directory and create a new one
DO_DELETE = False  # @param {type:"boolean"}
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
tf.io.gfile.makedirs(OUTPUT_DIR)
print('***** Model output directory: {} *****'.format(OUTPUT_DIR))

# data loading
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

# data loading from eraser
if dataset == 'movie':
    label_list = ['POS', 'NEG']
elif dataset == 'multirc':
    label_list = ['True', 'False']
elif dataset == 'fever':
    label_list = ['SUPPORTS', 'REFUTES']

def extract_doc_ids_from_annotations(anns):
    ret = set()
    for ann in anns:
        for ev_group in ann.evidences:
            for ev in ev_group:
                ret.add(ev.docid)
    return ret

from eraserbenchmark.rationale_benchmark.utils import load_datasets, load_documents
from itertools import chain
data_dir = f'/home/zzhang/.keras/datasets/{dataset}/'
train, val, test = load_datasets(data_dir)
docids = set(chain.from_iterable(extract_doc_ids_from_annotations(d) for d in [train, val, test]))
docs = load_documents(data_dir, docids)

# data preprocessing eraser
from bert_data_preprocessing_rational_eraser import load_bert_features, convert_bert_features

def preprocess(data, docs, label_list, dataset_name):
    features = load_bert_features(data, docs, label_list, MAX_SEQ_LENGTH)

    with_rations = ('cls' not in dataset_name)
    with_lable_id = ('seq' not in dataset_name)

    return convert_bert_features(features, with_lable_id, with_rations, EXP_OUTPUT)


@cache_decorator(os.path.join('cache', DATASET_NAME + '_eraser_format'))
def preprocess_wrapper(*data_inputs, docs=docs):
    ret = []
    for data in data_inputs:
        ret.append(preprocess(data, docs, label_list, dataset))
    return ret

rets_train, rets_val, rets_test = preprocess_wrapper(train, val, test, docs=docs)

train_input_ids, train_input_masks, train_segment_ids, train_rations, train_labels = rets_train
val_input_ids, val_input_masks, val_segment_ids, val_rations, val_labels = rets_val
test_input_ids, test_input_masks, test_segment_ids, test_rations, test_labels = rets_test

# initializing graph and session
graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)
set_session(sess)

# hyper-parameters of BERT's
WARMUP_PROPORTION = 0.1

num_train_steps = int(len(train_input_ids) / BATCH_SIZE * float(NUM_EPOCHS))
num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

# building models
from model import BertLayer
from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Multiply, Concatenate, RepeatVector, Dot, Lambda

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
        elif EXP_OUTPUT == 'interval':
            M1 = Bidirectional(layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True),
                               merge_mode='concat')(bert_exp_output)
            p_start = Dense(1, activation='sigmoid')(
                Concatenate(axis=-1)([bert_exp_output, M1]))

            m1_tilde = Dot(axes=-2)([p_start, M1])
            M1_tilde = Lambda(lambda x: tf.tile(
                x, (1, MAX_SEQ_LENGTH, 1)))(m1_tilde)
            x = Multiply()([M1, M1_tilde])
            M2 = Bidirectional(layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True),
                               merge_mode='concat')(Concatenate(axis=-1)([bert_exp_output, M1, M1_tilde, x]))
            p_end = Dense(1, activation='sigmoid')(
                Concatenate(axis=-1)([bert_exp_output, M2]))
            exp_outputs = Concatenate(
                axis=-1, name='exp_output')([p_start, p_end])
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
                              precision_wrapper(EXP_OUTPUT),
                              recall_wrapper(EXP_OUTPUT)]}
    if EXP_OUTPUT == 'gru':
        loss = {'cls_output': 'binary_crossentropy',
                'exp_output': loss_function()}
    elif EXP_OUTPUT == 'interval':
        loss = {'cls_output': 'binary_crossentropy',
                'exp_output': exp_interval_loss()}
    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model_exp = Model(inputs=bert_inputs, outputs=exp_outputs)
    optimizer = Adam(LEARNING_RATE)
    model_exp.compile(loss=loss_function(), optimizer=optimizer)

    return model, model_exp

# convert to eraser benchmark (TODO: support for interval exp output)
from eraserbenchmark.rationale_benchmark.utils import Annotation, Evidence
from eraserbenchmark.rationale_benchmark.utils import annotations_to_jsonl
from bert.tokenization import FullTokenizer, BasicTokenizer, WordpieceTokenizer,\
    convert_to_unicode, whitespace_tokenize, convert_ids_to_tokens
import re
from utils import *
from copy import deepcopy

pattern = re.compile('</?(POS)?(NEG)?>')
vocab = None


def convert_ids_to_token_list(input_ids):
    global vocab
    if vocab is None:
        from bert.tokenization import load_vocab
        with tf.Graph().as_default():
            bert_module = hub.Module(BERT_MODEL_HUB)
            tokenization_info = bert_module(
                signature="tokenization_info", as_dict=True)
            with tf.Session(config=config) as sess:
                vocab_file = sess.run(tokenization_info["vocab_file"])
        vocab = load_vocab(vocab_file)

    iv_vocab = {input_id: wordpiece for wordpiece, input_id in vocab.items()}
    token_list = convert_ids_to_tokens(iv_vocab, input_ids)
    return token_list


def convert_subtoken_ids_to_tokens(ids, exps=None, raw_sentence=None):
    subtokens = convert_ids_to_token_list(ids)
    tokens, exps_output = [], []
    exps_input = [0 for i in ids] if exps is None else exps
    raw_sentence = subtokens if raw_sentence is None else raw_sentence
    subtokens = list(
        reversed([t[2:] if t.startswith('##') else t for t in subtokens]))
    exps_input = list(reversed(exps_input))
    for ref_token in raw_sentence:
        t, e = '', 0
        while t != ref_token and len(subtokens) > 0:
            t += subtokens.pop()
            e = max(e, exps_input.pop())
        tokens.append(t)
        exps_output.append(e)
        if len(subtokens) == 0:
            # the last sub-token is incomplete, ditch it directly
            if ref_token != tokens[-1]:
                tokens = tokens[:-1]
                exps_output = exps_output[:-1]
            break
    if exps is None:
        return tokens
    return tokens, exps_output

# [SEP] == 102
# [CLS] == 101
# [PAD] == 0


def extract_texts(tokens, exps=None, text_a=True, text_b=False):
    if tokens[0] == 101:
        endp_text_a = tokens.index(102)
        if text_b:
            endp_text_b = endp_text_a + 1 + \
                tokens[endp_text_a + 1:].index(102)
    else:
        endp_text_a = tokens.index('[SEP]')
        if text_b:
            endp_text_b = endp_text_a + 1 + \
                tokens[endp_text_a + 1:].index('[SEP]')
    ret_token = []
    if text_a:
        ret_token += tokens[1: endp_text_a]
    if text_b:
        ret_token += tokens[endp_text_a + 1: endp_text_b]
    if exps is None:
        return ret_token
    else:
        ret_exps = []
        if text_a:
            ret_exps += exps[1: endp_text_a]
        if text_b:
            ret_exps += exps[endp_text_a + 1: endp_text_b]
        return ret_token, ret_exps


def pred_to_exp_mask(exp_pred, count=None, threshold=0.5):
    if count is None:
        return (np.array(exp_pred) >= threshold).astype(np.int32)
    temp = [(i, p) for i, p in enumerate(exp_pred)]
    temp = sorted(temp, key=lambda x: x[1], reverse=True)
    ret = np.zeros_like(exp_pred).astype(np.int32)
    for i, _ in temp[:count]:
        ret[i] = 1
    return ret


def rational_bits_to_ev_generator(token_list, raw_input, exp_pred, hard_selection_count, hard_selection_threshold):
    in_rationale = False
    ev = {'docid': raw_input['docids'][0],
          'start_token': -1, 'end_token': -1, 'text': ''}
    exp_masks = pred_to_exp_mask(
        exp_pred, hard_selection_count, hard_selection_threshold)
    for i, p in enumerate(exp_masks):
        if p == 0 and in_rationale:  # leave rational zone
            in_rationale = False
            ev['end_token'] = i
            ev['text'] = ' '.join(
                token_list[ev['start_token']: ev['end_token']])
            yield deepcopy(ev)
        elif p == 1 and not in_rationale:  # enter rational zone
            in_rationale = True
            ev['start_token'] = i
    if in_rationale:  # the final non-padding token is rational
        ev['end_token'] = len(exp_pred)
        ev['text'] = ' '.join(token_list[ev['start_token']: ev['end_token']])
        yield deepcopy(ev)


def pred_to_results(raw_input, input_ids, pred, hard_selection_count, hard_selection_threshold):
    cls_pred, exp_pred = pred
    exp_pred = exp_pred.reshape((-1,)).tolist()
    if 'sentence' in raw_input:
        raw_sentence = raw_input['sentence']
    else:
        raw_sentence = raw_input['passage']
    raw_sentence = re.sub(pattern, '', raw_sentence)
    raw_sentence = re.sub('\x12', '', raw_sentence)
    raw_sentence = raw_sentence.lower().split()
    if dataset == 'eraser_movie_mtl':
        token_ids, exp_pred = extract_texts(input_ids, exp_pred, text_a=True, text_b=False)
    else:
        token_ids, exp_pred = extract_texts(input_ids, exp_pred, text_a=False, text_b=True)
    token_list, exp_pred = convert_subtoken_ids_to_tokens(token_ids, exp_pred, raw_sentence)
    result = {'annotation_id': raw_input['annotation_id']}
    ev_groups = []
    if dataset == 'eraser_movie_mtl':
        docids = None
    elif dataset == 'eraser_fever' or dataset == 'eraser_multirc':
        docids = raw_input['docids']
    result['docids'] = docids
    result['rationales'] = [{'docid': docids[0]}]
    for ev in rational_bits_to_ev_generator(token_list, raw_input, exp_pred, hard_selection_count, hard_selection_threshold):
        ev_groups.append(ev)
    result['rationales'][-1]['hard_rationale_predictions'] = ev_groups
    
    
    
    
    
    
# Q-A dataset has its question on text_b, need to eliminate them from exp_pred6  
    
    
    
    
    
    
    
    
    
    
    result['rationales'][-1]['soft_rationale_predictions'] = exp_pred + \
        [0] * (len(raw_sentence) - len(token_list))
    if dataset == 'eraser_movie_mtl': 
        POS_LABEL = 'POS'
        NEG_LABEL = 'NEG'
    elif dataset == 'eraser_fever':
        POS_LABEL = 'SUPPORTS'
        NEG_LABEL = 'REFUTES'
    elif dataset == 'eraser_multirc':
        POS_LABEL = 'True'
        NEG_LABEL = 'False'
    result['classification'] = NEG_LABEL if round(
            cls_pred[0]) < (NEG+POS)/2 else POS_LABEL
    return result

# training, evaluation and inference
import os
SAVE_CHECKPOINTS_STEPS = 500
SAVE_SUMMARY_STEPS = 100

checkpoint_path = os.path.join(OUTPUT_DIR, 'cp-{epoch:04d}.ckpt')
checkpoint_dir = os.path.dirname(checkpoint_path)

train = True
load_best = False
assert train ^ load_best
evaluate = False
exp_visualize = False
exp_benchmark = True
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
    #training_inputs = [train_input_ids[:10],
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
        if dataset == 'eraser_fever':
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
                model.load_weights(checkpoint_path.format(epoch=ckpt_i))
                assert False  # dumm proof, most of the case the training is end-to-end, without disturbance and reloading
                break

    if train:
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

        def get_vocab(bert_model_hub):
            from bert.tokenization import load_vocab
            import tensorflow as tf
            with tf.Graph().as_default():
                bert_module = hub.Module(BERT_MODEL_HUB)
                tokenization_info = bert_module(signature="tokenization_info",
                                                as_dict=True)
                with tf.Session() as sess:
                    vocab_file = sess.run(tokenization_info["vocab_file"])
            vocab = load_vocab(vocab_file)
            return vocab

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

        results = [pred_to_results(raw_input.loc[i], benchmark_input_ids[i], (pred[0][i], pred[1][i]), HARD_SELECTION_COUNT, HARD_SELECTION_THRESHOLD)
                   for i in range(len(raw_input))]

        def remove_rations(line, args):
            instance_id = line.name
            instance = args[instance_id]
            rationales = instance['rationales'][0]['hard_rationale_predictions']
            if 'sentence' in line:
                sentence = line.sentence
            else:
                sentence = line.passage
            sentence = re.sub(pattern, '', sentence).lower().split()
            rationales = [{'end_token': 0, 'start_token': 0}] \
                + sorted(rationales, key=lambda x: x['start_token']) \
                + [{'start_token': len(sentence), 'end_token': len(sentence)}]
            ret = []
            for rat_id, rat in enumerate(rationales[:-1]):
                ret += ['.'] * (rat['end_token'] - rat['start_token']) \
                    + sentence[rat['end_token']
                        : rationales[rat_id + 1]['start_token']]
            if 'sentence' in line:
                line.sentence = ' '.join(ret)
            else:
                line.passage = ' '.join(ret)
            return line

        def extract_rations(line, args):
            instance_id = line.name
            instance = args[instance_id]
            rationales = instance['rationales'][0]['hard_rationale_predictions']
            if 'sentence' in line:
                sentence = line.sentence
            else:
                sentence = line.passage
            sentence = re.sub(pattern, '', sentence).lower().split()
            rationales = [{'end_token': 0, 'start_token': 0}] \
                + sorted(rationales, key=lambda x: x['start_token']) \
                + [{'start_token': len(sentence), 'end_token': len(sentence)}]
            ret = []
            for rat_id, rat in enumerate(rationales[:-1]):
                ret += sentence[rat['start_token']: rat['end_token']] \
                    + ['.'] * (rationales[rat_id + 1]
                               ['start_token'] - rat['end_token'])
            if 'sentence' in line:
                line.sentence = ' '.join(ret)
            else:
                line.passage = ' '.join(ret)
            return line

        def get_cls_score(model, raw_input, label_list, dataset, r_function, rationales):
            _input = deepcopy(raw_input)
            _input = _input.apply(r_function, axis=1, args=(rationales,))
            rets = preprocess(_input, label_list, dataset)
            _input_ids, _input_masks, _segment_ids, _rations, _labels = rets

            _inputs = [_input_ids, _input_masks, _segment_ids]
            _pred = model.predict(_inputs)
            return(np.hstack([1-_pred[0], _pred[0]]))

        def add_cls_scores(res, cls, c, s):
            res['classification_scores'] = {'NEG': cls[0], 'POS': cls[1]}
            res['comprehensiveness_classification_scores'] = {
                'NEG': c[0], 'POS': c[1]}
            res['sufficiency_classification_scores'] = {
                'NEG': s[0], 'POS': s[1]}
            return res

        pred_softmax = np.hstack([1-pred[0], pred[0]])
        c_pred_softmax = get_cls_score(
            model, raw_input, label_list, dataset, remove_rations, results)
        s_pred_softmax = get_cls_score(
            model, raw_input, label_list, dataset, extract_rations, results)

        results = [add_cls_scores(res, cls_score, c_cls_score, s_cls_score) for res, cls_score,
                   c_cls_score, s_cls_score in zip(results, pred_softmax, c_pred_softmax, s_pred_softmax)]

        with open(result_fname+'pkl3', "wb+") as pfout:
            pickle.dump(results, pfout)

        from eraserbenchmark.eraser import evaluate
        evaluate(MODEL_NAME)

    with open(cls_output_file, 'a+') as fw:
        fw.write('/////////////////experiment ends//////////////////\n\n\n')
