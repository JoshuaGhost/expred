import pickle
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

import os
from model import BertLayer
import numpy as np

cls_only = False

identifier = 'exp'
#identifier = 'mtl'

dataset = 'movies'
#dataset = 'fever'
dataset = 'multirc'

cache_fname = f'bert_base_seqlen_512_{dataset}_exp_output_gru_merged_evidences_inputdata_cache_eraser_format'

if identifier == 'exp':
    exp_only = True
    identifier_model_name = f'seqlen_512_dataset_{dataset}_exp_structure_gru_complete_sentences_merged_evidences_bert_base_bce_rebalance_approach_resampling_pooling_first_learning_rate_1e-05_no_weight_exp_only'
    classifier_model_name = f'seqlen_512_dataset_{dataset}_exp_structure_none_train_on_annotations_only_machine_rationale_exp_only_merged_evidences_bert_base_bce_rebalance_approach_resampling_pooling_first_learning_rate_1e-05_no_weight_cls_only_eval_annotations'
    if dataset == 'movies':
        label_list = ['POS', 'NEG']
        identifier_ckpt_num = '0003'
        classifier_ckpt_num = '0005'
    elif dataset == 'multirc':
        label_list = ['True', 'False']
        identifier_ckpt_num = '0001'
        classifier_ckpt_num = '0001'
    elif dataset == 'fever':
        label_list = ['SUPPORTS', 'REFUTES']
        par_lambda = 20
        identifier_ckpt_num = '0002'
        classifier_ckpt_num = '0001'
elif identifier == 'mtl':
    exp_only = False
    identifier_model_name = f'seqlen_512_dataset_{dataset}_exp_structure_gru_complete_sentences_merged_evidences_bert_base_bce_rebalance_approach_resampling_pooling_first_learning_rate_1e-05_par_lambda_{par_lambda}_'
    classifier_model_name = f'seqlen_512_dataset_{dataset}_exp_structure_none_train_on_annotations_only_machine_annotation_mtl_hard_lambda_{par_lambda}_merged_evidences_bert_base_bce_rebalance_approach_resampling_pooling_first_learning_rate_1e-05_no_weight_cls_only_eval_annotations'
    if dataset == 'movies':
        label_list = ['POS', 'NEG']
        par_lambda = 5
        identifier_ckpt_num = '0003'
        classifier_ckpt_num = '0010'
    elif dataset == 'multirc':
        label_list = ['True', 'False']
        par_lambda = 2
        identifier_ckpt_num = '0001'
        classifier_ckpt_num = '0001'
    elif dataset == 'fever':
        label_list = ['SUPPORTS', 'REFUTES']
        par_lambda = 20
        identifier_ckpt_num = '0002'
        classifier_ckpt_num = '0002'


with open(f'cache/{cache_fname}', 'br') as fin:
    rets_train, rets_val, rets_test = pickle.load(fin)
train_input_ids, train_input_masks, train_segment_ids, train_rations, train_labels = rets_train
val_input_ids, val_input_masks, val_segment_ids, val_rations, val_labels = rets_val
test_input_ids, test_input_masks, test_segment_ids, test_rations, test_labels = rets_test


graph = tf.get_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = '0'
config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
sess = tf.Session(config=config)

set_session(sess)
training_inputs = [train_input_ids, train_input_masks, train_segment_ids]
val_inputs = [val_input_ids, val_input_masks, val_segment_ids]
test_inputs = [test_input_ids, test_input_masks, test_segment_ids]
sess.run(tf.local_variables_initializer())
sess.run(tf.global_variables_initializer())
sess.run(tf.tables_initializer())


from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Multiply, Concatenate, Dot, Lambda, Softmax
from bert_utils import get_vocab
from eraser_benchmark import pred_to_results
from metrices import *
from losses import *

vocab = get_vocab(config)

# prepare for the model
loss_function = imbalanced_bce_resampling
DIM_DENSE_CLS = 256
NUM_GRU_UNITS_BERT_SEQ = 128
NUM_INTERVAL_LSTM_WIDTH = 100
exp_structure = 'gru'
loss_function = imbalanced_bce_resampling

if exp_only:
    loss_weights = None
else:
    loss_weights = {'cls_output': 1,
                    'exp_output': par_lambda}
metrics, loss = {}, {}
if not exp_only:
    metrics['cls_output'] = 'accuracy'
    loss['cls_output'] = 'binary_crossentropy'
if exp_structure != 'none':
    metrics['exp_output'] = [f1_wrapper(exp_structure),
                             sp_precision_wrapper(exp_structure),
                             sp_recall_wrapper(exp_structure),
                             precision_wrapper(exp_structure),
                             recall_wrapper(exp_structure)]
    loss['exp_output'] = loss_function()

def build_model():
    in_id = Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")
    in_mask = Input(shape=(MAX_SEQ_LENGTH,), name="input_masks")
    in_segment = Input(shape=(MAX_SEQ_LENGTH,), name="segment_ids")
    bert_inputs = [in_id, in_mask, in_segment]
    bert_cls_output, bert_exp_output = BertLayer(
        n_fine_tune_layers=10, name='bert')(bert_inputs)
    outputs = []
    model_cls, model_exp = None, None
    if 'seq' not in dataset and not exp_only:
        # Classifier output
        dense = Dense(DIM_DENSE_CLS, activation='tanh', name='cls_dense')(bert_cls_output)
        cls_output = Dense(1, activation='sigmoid', name='cls_output')(dense)
        outputs.append(cls_output)
        model_cls = Model(inputs=bert_inputs, outputs=cls_output)
        optimizer = Adam(LEARNING_RATE)
        model_cls.compile(loss=loss['cls_output'], optimizer=optimizer, metrics=[metrics['cls_output']])
    else:
        model_cls = None
    if 'cls' not in dataset and exp_structure != 'none' and not cls_only:
        # Explainer output
        if exp_structure == 'gru':
            gru = CuDNNGRU(
                NUM_GRU_UNITS_BERT_SEQ, kernel_initializer='random_uniform', return_sequences=True,
                name='exp_gru_gru')(
                bert_exp_output)
            exp = Dense(1, activation='sigmoid', name='exp_gru_dense')(gru)
            output_mask = Reshape((MAX_SEQ_LENGTH, 1), name='exp_gru_reshape')(in_mask)
            exp_outputs = Multiply(name='exp_output')([output_mask, exp])
        elif exp_structure == 'rnr':
            M1 = Bidirectional(
                layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True, name='exp_rnr_lstm1'),
                merge_mode='concat', name='exp_rnr_bidirectional1')(bert_exp_output)
            p_starts = Dense(1, activation='sigmoid', name='exp_rnr_starts')(
                Concatenate(axis=-1)([bert_exp_output, M1]))
            start_mask = Reshape((MAX_SEQ_LENGTH, 1))(in_mask)
            p_starts = Multiply()([p_starts, start_mask])
            m1_tilde = Dot(axes=-2)([p_starts, M1])
            M1_tilde = Lambda(lambda x: tf.tile(x, (1, MAX_SEQ_LENGTH, 1)))(m1_tilde)
            x = Multiply()([M1, M1_tilde])
            M2 = Bidirectional(
                layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True, name='exp_rnr_lstm2'),
                merge_mode='concat', name='exp_rnr_bidirecitonal2')(
                Concatenate(axis=-1)([bert_exp_output, M1, M1_tilde, x]))
            p_end_given_start = Dense(MAX_SEQ_LENGTH, activation='linear', name='exp_rnr_end')(
                Concatenate(axis=-1)([bert_exp_output, M2]))
            end_mask = Lambda(lambda x: tf.tile(x, (1, MAX_SEQ_LENGTH, 1)))(Reshape((1, MAX_SEQ_LENGTH))(in_mask))
            p_end_given_start = Multiply()([p_end_given_start, end_mask])
            p_end_given_start = Lambda(lambda x: tf.linalg.band_part(x, 0, -1))(p_end_given_start)
            p_end_given_start = Softmax(axis=-1)(p_end_given_start)
            exp_outputs = Concatenate(axis=-1, name='exp_output')([p_starts, p_end_given_start])
        outputs.append(exp_outputs)
        model_exp = Model(inputs=bert_inputs, outputs=exp_outputs)
        optimizer = Adam(LEARNING_RATE)
        model_exp.compile(loss=loss['exp_output'], optimizer=optimizer, metrics=[metrics['exp_output']])
    else:
        model_exp = None
    model = Model(inputs=bert_inputs, outputs=outputs)
    optimizer = Adam(LEARNING_RATE)
    model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)
    return model, model_cls, model_exp

LEARNING_RATE = 1e-5
model, model_cls, model_exp = build_model()
model.load_weights(f"model_checkpoints/emnlp2020/two_stages_{identifier}+cls/{identifier_model_name}/cp-{identifier_ckpt_num}.ckpt")

# prepare the data
from eraserbenchmark.rationale_benchmark.utils import load_datasets
train, val, test = load_datasets(f'/home/zzhang/.keras/datasets/{dataset}')
training_outputs, test_outputs, val_outputs = {}, {}, {}
if not exp_only:
    training_outputs['cls_output'] = train_labels
    test_outputs['cls_output'] = test_labels
    val_outputs['cls_output'] = val_labels
training_outputs['exp_output'] = train_rations
test_outputs['exp_output'] = test_rations
val_outputs['exp_output'] = val_rations

# expred-stage-1
print(model.evaluate(x=test_inputs, y=test_outputs, batch_size=10, verbose=1))

# annotated test for the 2nd stage
benchmark_inputs, raw_input, benchmark_input_ids, benchmark_outputs = test_inputs, test, test_input_ids, test_outputs
pred = model.predict(x=benchmark_inputs)
if exp_only:
    cls_pred = np.array([[1] for i in pred])
    exp_pred = pred
    print(cls_pred.shape, exp_pred.shape)
else:
    cls_pred, exp_pred = pred
from eraserbenchmark.eraser_utils import extract_doc_ids_from_annotations
from itertools import chain
from eraserbenchmark.rationale_benchmark.utils import load_documents
docids = set(chain.from_iterable(extract_doc_ids_from_annotations(d) for d in [train, val, test]))
docs = load_documents(f'/home/zzhang/.keras/datasets/{dataset}', docids)
from bert.tokenization import FullTokenizer, BasicTokenizer, \
    convert_to_unicode, whitespace_tokenize, convert_ids_to_tokens


import tensorflow_hub as hub
from bert_with_ration_eraser import BasicTokenizerWithRation
gpu_id = '0'

class FullTokenizerWithRations(FullTokenizer):  # Test passed :)
    def __init__(self, vocab_file, do_lower_case=True):
        self.basic_rational_tokenizer = BasicTokenizerWithRation(do_lower_case=do_lower_case)
        super(FullTokenizerWithRations, self).__init__(vocab_file, do_lower_case)
    def tokenize(self, text, evidences=None):
        split_tokens = []
        split_rations = []
        for token, ration in self.basic_rational_tokenizer.tokenize(text, evidences):
            for sub_token in self.wordpiece_tokenizer.tokenize(token):
                split_tokens.append(sub_token)
                split_rations.append(ration)
        return list(zip(split_tokens, split_rations))
    @classmethod
    def create_tokenizer_from_hub_module(self, gpu_id):
        """Get the vocab file and casing info from the Hub module."""
        with tf.Graph().as_default():  # basically useless, but good practice to specify the graph using, even it sets the default graph as the default graph
            bert_module = hub.Module(BERT_MODEL_HUB)
            tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
            config = tf.ConfigProto()
            config.gpu_options.allow_growth = True
            config.gpu_options.visible_device_list = gpu_id
            config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
            with tf.Session(
                    config=config) as sess:  # create a new session, with session we can setup even remote computation
                vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                      tokenization_info["do_lower_case"]])
        return FullTokenizerWithRations(vocab_file=vocab_file, do_lower_case=do_lower_case)
full_tokenizer = FullTokenizerWithRations.create_tokenizer_from_hub_module(gpu_id)

results = [pred_to_results(raw_input[i], benchmark_input_ids[i],
                           (cls_pred[i], exp_pred[i]),
                           None,
                           0.5,
                           vocab, docs, label_list, 
                           exp_structure, 
                           full_tokenizer, 
                           identifier=identifier,
                           fix_empty_evidence='doc')
           for i in range(len(cls_pred))]
bert_tokens = [res[1] for res in results]
bert_tokens = {annotation_id: tokens for annotation_id, tokens in bert_tokens}
results = [res[0] for res in results]
exp_output_fname = f'model_checkpoints/emnlp2020/two_stages_{identifier}+cls/{identifier_model_name}/mtl_output_test.jsonl'
import json
with open(exp_output_fname, 'w+') as fout:
    for res in results:
        if len(res) == 0:
            continue
        json.dump(res, fout)
        fout.write('\n')

# benchmark result for the first stage

from eraser_benchmark import get_cls_score
from eraser_benchmark import remove_rations, extract_rations

exp_structure = 'gru'
if identifier == 'mtl':
    pred_softmax = np.hstack([1 - cls_pred, cls_pred])
    

    c_pred_softmax = get_cls_score(
        model, results, docs, label_list, dataset, remove_rations, MAX_SEQ_LENGTH, exp_structure,
        gpu_id=gpu_id, tokenizer=full_tokenizer)
    s_pred_softmax = get_cls_score(
        model, results, docs, label_list, dataset, extract_rations, MAX_SEQ_LENGTH, exp_structure,
        gpu_id=gpu_id, tokenizer=full_tokenizer)

    from eraser_benchmark import add_cls_scores
    results = [add_cls_scores(res,
                              cls_score,
                              c_cls_score,
                              s_cls_score,
                              label_list) for res, cls_score, c_cls_score, s_cls_score in zip(results,
                                                                                              pred_softmax,
                                                                                              c_pred_softmax,
                                                                                              s_pred_softmax)]
anns_saved = set()
benchmark_results = []
for ann in results:
    if ann['annotation_id'] not in anns_saved:
        anns_saved.add(ann['annotation_id'])
        benchmark_results.append(ann)
benchmark_result_fname = f'model_checkpoints/emnlp2020/two_stages_{identifier}+cls/{identifier_model_name}/test_encoded.'
with open(benchmark_result_fname + 'pkl3', "wb+") as pfout:
    pickle.dump(benchmark_results, pfout)
from eraserbenchmark.eraser import evaluate
evaluate(benchmark_result_fname, dataset, 'test', 0, f'/home/zzhang/.keras/datasets/{dataset}')


# beginning of phase 2


# prepare for the cls only model
cls_only = True
exp_only = False
exp_structure = 'none'
metrics = {}
loss = {}
if not exp_only:
    metrics['cls_output'] = 'accuracy'
    loss['cls_output'] = 'binary_crossentropy'
if exp_structure != 'none':
    metrics['exp_output'] = [f1_wrapper(exp_structure),
                             sp_precision_wrapper(exp_structure),
                             sp_recall_wrapper(exp_structure),
                             precision_wrapper(exp_structure),
                             recall_wrapper(exp_structure)]
    loss['exp_output'] = loss_function()
loss_weights = None
model_cls_only, model_cls_cls_only, model_exp_cls_only = build_model()
model_cls_only.load_weights(f'model_checkpoints/emnlp2020/two_stages_{identifier}+cls/{classifier_model_name}/cp-{classifier_ckpt_num}.ckpt')

# prepare for the data for the cls only model
from eraser_benchmark import flatten_rations
def extract_rations_collapse_wildcards(sentence, rations, tokenizer=None, \
                                       sub='.', combine_subtokens=False, \
                                       rep_count=0):
    sentence = sentence.lower().split()
    if isinstance(rations, list): # a list of Evidence-s
        rations = [e.__dict__ for e in rations]
    else: # an Annotation
        rations = rations['rationales'][0]['hard_rationale_predictions']
    rations = flatten_rations(rations, len(sentence))
    ret = []
    for rat_id, rat in enumerate(rations[:-1]):
        ret += sentence[rat['start_token']: rat['end_token']] + [sub] * (rep_count)
    return ' '.join(ret)
# training_data_decorator = extract_rations_collapse_whildcards
training_data_decorator = extract_rations
from bert_data_preprocessing_rational_eraser import preprocess
# rets_test_cls_only = preprocess(test, docs, label_list, dataset, 512, 'none', True, data_decorator=training_data_decorator, gpu_id='0')
# test_input_ids_cls_only, test_input_masks_cls_only, test_segment_ids_cls_only, test_rations_cls_only, test_labels_cls_only = rets_test_cls_only
# test_inputs_cls_only = [test_input_ids_cls_only, test_input_masks_cls_only, test_segment_ids_cls_only]
# training_outputs_cls_only, test_outputs_cls_only, val_outputs_cls_only = {}, {}, {}
# test_outputs['cls_output'] = test_labels

from eraserbenchmark.rationale_benchmark.utils import Annotation
from eraserbenchmark.rationale_benchmark.utils import Evidence
annotated_test = []
for res, orig in zip(results, test):
    assert res['annotation_id'] == orig.annotation_id
    evidences = []
    for ev in res['rationales'][0]['hard_rationale_predictions']:
        evidences.append(Evidence(text=ev['text'], start_token=ev['start_token'], end_token=ev['end_token'], docid=ev['docid']))
    
    ann = Annotation(annotation_id=res['annotation_id'], classification=orig.classification, query=res['query'], docids=res['docids'], evidences=[evidences])
    annotated_test.append(ann)
rets_test_cls_only = preprocess(annotated_test, docs, label_list, dataset, 512, 'none', True, training_data_decorator, '0')
test_input_ids_cls_only, test_input_masks_cls_only, test_segment_ids_cls_only, test_rations_cls_only, test_labels_cls_only = rets_test_cls_only
test_inputs_cls_only = [test_input_ids_cls_only, test_input_masks_cls_only, test_segment_ids_cls_only]
benchmark_inputs_cls_only, raw_input_cls_only, benchmark_input_ids_cls_only, benchmark_outputs_cls_only = test_inputs_cls_only, annotated_test, test_input_ids_cls_only, test_outputs
import numpy as np

# evaluate phase-2 model
preds_cls_only = model_cls_only.predict(x=benchmark_inputs_cls_only)
#cls_pred_cls_only = pred_cls_only
#exp_pred_cls_only = np.array([[0 for j in range(MAX_SEQ_LENGTH)] for i in pred])
# empty_idxs = [8,12,28,34,41,61,64]
from sklearn.metrics import classification_report
preds_cls_only = list(chain.from_iterable(np.round(preds_cls_only).astype(np.int)))
# preds_cls_only[8] = 1
# preds_cls_only[12] = 1
# preds_cls_only[28] = 1
# preds_cls_only[34] = 1
# preds_cls_only[41] =1
# preds_cls_only[61] =1
# preds_cls_only[64] =1
flat_test_labels = list(chain.from_iterable(test_labels))
with open(f'model_checkpoints/emnlp2020/two_stages_{identifier}+cls/{classifier_model_name}/stage_2_performance.txt', 'w+') as fout:
    fout.write(str(classification_report(flat_test_labels, preds_cls_only, output_dict=True)))
