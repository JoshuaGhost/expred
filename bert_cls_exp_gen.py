# imports
import argparse
from datetime import datetime

import numpy as np
import shutil
import torch
from transformers import BertModel, BertTokenizer
from tqdm import tqdm_notebook

from display_rational import convert_res_to_htmls
from eraser_benchmark import rnr_matrix_to_rational_mask
from losses import imbalanced_bce_bayesian, imbalanced_bce_resampling
from losses import rnr_matrix_loss
from metrices import *
from utils import *

LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 512
HARD_SELECTION_COUNT = None
HARD_SELECTION_THRESHOLD = 0.5


def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--par_lambda', type=float)
    parser.add_argument('--gpu_id', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--len_viz_head', type=int, default=100)
    parser.add_argument('--num_epochs', type=int)
    parser.add_argument('--dataset', type=str, choices='fever multirc movies'.split())  # fever, multirc, movie
    parser.add_argument("--do_train", action='store_true')
    parser.add_argument('--exp_visualize', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--exp_benchmark', action='store_true')
    parser.add_argument('--freeze_cls', action='store_true')
    parser.add_argument('--freeze_exp', action='store_true')
    parser.add_argument('--train_cls_first', action='store_true')
    parser.add_argument('--train_exp_first', action='store_true')
    parser.add_argument('--use_bucket', action='store_true')
    parser.add_argument('--exp_structure', type=str, default='gru', choices='gru rnr'.split())  # gru, rnr
    parser.add_argument('--delete_checkpoints', action='store_true')
    parser.add_argument('--merge_evidences', action='store_true')
    parser.add_argument('--benchmark_split', type=str, default='test', choices='test train val'.split())  # gru, rnr
    parser.add_argument('--train_on_portion', type=float, default=0)
    parser.add_argument('--start_from_phase1', action='store_true')
    parser.add_argument('--load_phase1', action='store_true')
    parser.add_argument('--pooling', type=str, default='first', choices=['first', 'mean'])
    parser.add_argument('--bert_size', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--rebalance_approach', type=str, default='resampling', choices=['resampling', 'bayesian'])
    parser.add_argument('--data_dir', type=str)

    return parser


class Config():
    def __init__(self, args):
        self.BATCH_SIZE = args.batch_size
        self.par_lambda = args.par_lambda
        self.NUM_EPOCHS = args.num_epochs
        self.gpu_id = args.gpu_id
        self.exp_structure = args.exp_structure
        self.dataset_name = args.dataset
        self.DO_DELETE = args.delete_checkpoints
        self.do_train = args.do_train
        self.load_best = not self.do_train
        self.evaluate = args.evaluate
        self.exp_visualize = args.exp_visualize
        self.exp_benchmark = args.exp_benchmark
        self.merge_evidences = args.merge_evidences
        self.BENCHMARK_SPLIT_NAME = args.benchmark_split
        self.train_on_portion = args.train_on_portion
        self.freeze_cls = args.freeze_cls
        self.freeze_exp = args.freeze_exp
        self.train_cls_first = args.train_cls_first
        self.train_exp_first = args.train_exp_first
        self.start_from_phase1 = args.start_from_phase1
        self.load_phase1 = args.load_phase1
        self.pooling = args.pooling
        self.EXP_OUTPUT = self.exp_structure
        self.bert_size = args.bert_size
        self.rebalance_approach = args.rebalance_approach
        self.USE_BUCKET = args.use_bucket
        self.data_dir = args.data_dir
        self.len_head = args.len_viz_head

        self.loss_function = self.decide_loss_function()
        self.suffix = self.decide_suffix()
        self.montage_output_dir()

        if config.par_lambda is None:
            self.loss_weights = None
        else:
            self.loss_weights = {'cls_output': 1,
                            'exp_output': self.par_lambda}
        self.metrics = {'cls_output': 'accuracy',
                   'exp_output': [f1_wrapper(self.EXP_OUTPUT),
                                  sp_precision_wrapper(self.EXP_OUTPUT),
                                  sp_recall_wrapper(self.EXP_OUTPUT),
                                  precision_wrapper(self.EXP_OUTPUT),
                                  recall_wrapper(self.EXP_OUTPUT)]}
        self.loss = {'cls_output': 'binary_crossentropy',
                     'exp_output': self.loss_function()}

    def decide_loss_function(self):
        if self.EXP_OUTPUT == 'gru':
            if self.rebalance_approach == 'resampling':
                return imbalanced_bce_resampling
            else:
                return imbalanced_bce_bayesian
        elif self.EXP_OUTPUT == 'rnr':
            return rnr_matrix_loss

    def decide_suffix(self):
        if self.freeze_cls:
            return 'freeze_cls'
        elif self.freeze_exp:
            return 'freeze_exp'
        elif self.train_cls_first:
            return 'train_cls_first'
        elif self.train_exp_first:
            return 'train_exp_first'
        else:
            return ''

    def montage_output_dir(self):
        OUTPUT_DIR = ['bert_{}_seqlen_{}_{}_exp_output_{}'.format(self.bert_size, MAX_SEQ_LENGTH, dataset, self.EXP_OUTPUT)]
        OUTPUT_DIR.append('merged_evidences' if self.merge_evidences else 'separated_evidences')
        if self.train_on_portion != 0:
            OUTPUT_DIR += ['train_on_portion', str(self.train_on_portion)]
        self.DATASET_CACHE_NAME = '_'.join(OUTPUT_DIR) + '_inputdata_cache'
        if self.par_lambda is None:
            OUTPUT_DIR.append('no_weight')
        else:
            OUTPUT_DIR.append('par_lambda_{}'.format(self.par_lambda))
        OUTPUT_DIR.append(
            'no_padding_imbalanced_bce_{}_pooling_{}_learning_rate_{}'.format(self.rebalance_approach, self.pooling, LEARNING_RATE))
        OUTPUT_DIR.append(self.suffix)
        OUTPUT_DIR = '_'.join(OUTPUT_DIR)
        self.MODEL_NAME = OUTPUT_DIR
        self.OUTPUT_DIR = os.path.join('model_checkpoints', self.MODEL_NAME)


class Dataset():
    def __init__(self, config):
        from eraserbenchmark.rationale_benchmark.utils import load_datasets, load_documents
        from eraserbenchmark.eraser_utils import extract_doc_ids_from_annotations
        from itertools import chain
        from bert_data_preprocessing_rational_eraser import preprocess

        if config.dataset_name == 'movies':
            self.label_list = ['POS', 'NEG']
        elif config.dataset == 'multirc':
            self.label_list = ['True', 'False']
        elif config.dataset == 'fever':
            self.label_list = ['SUPPORTS', 'REFUTES']

        self.train, self.val, self.test = load_datasets(config.data_dir)
        if config.train_on_portion != 0:
            self.train = self.train[:int(len(self.train) * config.train_on_portion)]
        self.docids = set(chain.from_iterable(extract_doc_ids_from_annotations(d) for d in [self.train, self.val, self.test]))
        self.docs = load_documents(config.data_dir, self.docids)

        @cache_decorator(os.path.join('cache', config.DATASET_CACHE_NAME + '_eraser_format'))
        def preprocess_wrapper(*data_inputs):#, docs=self.docs):
            ret = []
            for data in data_inputs:
                ret.append(
                    preprocess(data, self.docs, self.label_list,
                               config.dataset_name, config.MAX_SEQ_LENGTH,
                               config.EXP_OUTPUT, config.merge_evidences,
                               gpu_id=config.gpu_id))
            return ret

        self.rets_train, self.rets_val, self.rets_test = preprocess_wrapper(self.train, self.val, self.test, docs=self.docs)

        self.train_input_ids, self.train_input_masks, self.train_segment_ids, self.train_rations, self.train_labels = self.rets_train
        self.val_input_ids, self.val_input_masks, self.val_segment_ids, self.val_rations, self.val_labels = self.rets_val
        self.test_input_ids, self.test_input_masks, self.test_segment_ids, self.test_rations, self.test_labels = self.rets_test

    @classmethod
    def expand_on_evidences(self, data):
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


if __name__ == '__main__':
    parser = setup_parser()

    args = ['--par_lambda', '5.0',
            '--gpu_id', '0',
            '--batch_size', '2',
            '--num_epochs', '10',
            '--dataset', 'fever',
            '--exp_visualize',
            '--exp_structure', 'gru',
            '--merge_evidences']

    # args = parser.parse_args(args)
    args = parser.parse_args()
    config = Config(args)

    assert (not (config.freeze_cls and config.freeze_exp))
    assert (not (config.train_exp_first and config.train_cls_first))
    assert not ((config.freeze_cls or config.freeze_exp) and (
            config.train_exp_first or config.train_cls_first))  # can't freeze both in the same time

    if config.DO_DELETE:
        try:
            shutil.rmtree(config.OUTPUT_DIR)
        except:
            pass

    mkdirs(config.OUTPUT_DIR)
    print('***** Model output directory: {} *****'.format(config.OUTPUT_DIR))

    dataset = Dataset(config)

    # hyper-parameters of BERT's
    WARMUP_PROPORTION = 0.1
    num_train_steps = int(len(config.train_input_ids) / config.BATCH_SIZE * float(config.NUM_EPOCHS))
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    from bert_utils import get_vocab

    vocab = get_vocab()

    # building models
    from model import BertLayer
    from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Reshape, Multiply, Concatenate, Dot, Lambda, Softmax

    from metrices import sp_precision_wrapper, sp_recall_wrapper
    from metrices_pytorch import sp_precision_wrapper, sp_recall_wrapper

    DIM_DENSE_CLS = 256
    NUM_GRU_UNITS_BERT_SEQ = 128
    NUM_INTERVAL_LSTM_WIDTH = 100

    def build_model():
        in_id = Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")
        in_mask = Input(shape=(MAX_SEQ_LENGTH,), name="input_masks")
        in_segment = Input(shape=(MAX_SEQ_LENGTH,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]

        bert_cls_output, bert_exp_output = BertLayer(
            n_fine_tune_layers=10, name='bert')(bert_inputs)

        outputs = []
        if 'seq' not in dataset:
            # Classifier output
            dense = Dense(DIM_DENSE_CLS, activation='tanh', name='cls_dense')(bert_cls_output)
            cls_output = Dense(1, activation='sigmoid', name='cls_output')(dense)
            outputs.append(cls_output)
        if 'cls' not in dataset:
            # Explainer output
            if EXP_OUTPUT == 'gru':
                gru = CuDNNGRU(
                    NUM_GRU_UNITS_BERT_SEQ, kernel_initializer='random_uniform', return_sequences=True,
                    name='exp_gru_gru')(
                    bert_exp_output)
                exp = Dense(1, activation='sigmoid', name='exp_gru_dense')(gru)
                output_mask = Reshape((MAX_SEQ_LENGTH, 1), name='exp_gru_reshape')(in_mask)
                exp_outputs = Multiply(name='exp_output')([output_mask, exp])
            elif EXP_OUTPUT == 'rnr':
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

        model = Model(inputs=bert_inputs, outputs=outputs)
        optimizer = Adam(LEARNING_RATE)

        model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)

        model_exp = Model(inputs=bert_inputs, outputs=exp_outputs)
        optimizer = Adam(LEARNING_RATE)
        model_exp.compile(loss=loss['exp_output'], optimizer=optimizer, metrics=[metrics['exp_output']])

        model_cls = Model(inputs=bert_inputs, outputs=cls_output)
        optimizer = Adam(LEARNING_RATE)
        model_cls.compile(loss=loss['cls_output'], optimizer=optimizer, metrics=[metrics['cls_output']])

        return model, model_cls, model_exp


    # training, evaluation and inference
    import os

    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    checkpoint_path = os.path.join(OUTPUT_DIR, 'cp-{epoch:04d}.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cls_output_file = os.path.join(OUTPUT_DIR, 'output.txt')

    RES_FOR_BENCHMARK_FNAME = MODEL_NAME + '_' + BENCHMARK_SPLIT_NAME

    with graph.as_default():
        set_session(sess)
        model, model_cls, model_exp = build_model(par_lambda)
        model.summary()
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())
        sess.run(tf.tables_initializer())

        training_inputs = [train_input_ids, train_input_masks, train_segment_ids]
        val_inputs = [val_input_ids, val_input_masks, val_segment_ids]
        test_inputs = [test_input_ids, test_input_masks, test_segment_ids]

        training_outputs, test_outputs, val_outputs = {}, {}, {}

        if 'seq' not in dataset:
            training_outputs['cls_output'] = train_labels
            test_outputs['cls_output'] = test_labels
            val_outputs['cls_output'] = val_labels
        if 'cls' not in dataset:
            training_outputs['exp_output'] = train_rations
            test_outputs['exp_output'] = test_rations
            val_outputs['exp_output'] = val_rations

        initial_epoch = 0
        if load_best:
            with open(cls_output_file, 'r') as fin:
                log = fin.readlines()
            history = eval(log[2])
            best_epoch = np.argmin(history['loss']) + 1
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
            es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

            with open(cls_output_file, 'a+') as fw:
                fw.write("=============== {} ===============\n".format(datetime.now()))
            if train_cls_first or train_exp_first:
                if train_cls_first:
                    phases = ['cls', 'exp']
                    model_phase0 = model_cls
                    model_phase1 = model_exp
                    training_outputs_phase0, val_outputs_phase0 = training_outputs['cls_output'], val_outputs[
                        'cls_output']
                    training_outputs_phase1, val_outputs_phase1 = training_outputs['exp_output'], val_outputs[
                        'exp_output']
                elif train_exp_first:
                    phases = ['exp', 'cls']
                    model_phase0 = model_exp
                    model_phase1 = model_cls
                    training_outputs_phase0, val_outputs_phase0 = training_outputs['exp_output'], val_outputs[
                        'exp_output']
                    training_outputs_phase1, val_outputs_phase1 = training_outputs['cls_output'], val_outputs[
                        'cls_output']
                for layer in model.layers:
                    if layer.name.startswith(phases[1]):
                        layer.trainable = False
                checkpoint_path_phase0 = os.path.join(OUTPUT_DIR, 'phase0', 'cp-{epoch:04d}.ckpt')
                mkdirs(os.path.join(OUTPUT_DIR, 'phase0'))
                cp_callback_phase0 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_phase0,
                                                                        save_weights_only=False,
                                                                        verbose=1,
                                                                        period=1)
                if start_from_phase1:
                    with open(cls_output_file, 'r') as fin:
                        log = fin.readlines()
                    history = eval(log[2])
                    best_epoch = np.argmin(history['loss']) + 1
                    model_phase0.load_weights(checkpoint_path_phase0.format(epoch=best_epoch))
                else:
                    history = model_phase0.fit(
                        training_inputs,
                        training_outputs_phase0,
                        validation_data=(val_inputs, val_outputs_phase0),
                        epochs=NUM_EPOCHS,
                        batch_size=BATCH_SIZE,
                        callbacks=[cp_callback_phase0, es_callback],
                        initial_epoch=initial_epoch
                    )
                    with open(cls_output_file, 'a+') as fw:
                        fw.write("phase 0 training history {}:\n".format(datetime.now()))
                        fw.write(str(history.history) + '\n')
                    evaluation_res = model.evaluate(x=test_inputs,
                                                    y=test_outputs,
                                                    batch_size=BATCH_SIZE,
                                                    verbose=1)
                    with open(cls_output_file, 'a+') as fw:
                        fw.write("phase 0 evaluation {}:\n".format(datetime.now()))
                        fw.write(str(evaluation_res) + '\n')
                for layer in model.layers:
                    if layer.name == 'bert' or layer.name.startswith(phases[0]):
                        layer.trainable = False
                    elif layer.name.startswith(phases[1]):
                        layer.trainable = True
                for layer in model_phase1.layers:
                    if layer.name == 'bert' or layer.name.startswith(phases[0]):
                        layer.trainable = False
                    elif layer.name.startswith(phases[1]):
                        layer.trainable = True
                optimizer = Adam(LEARNING_RATE)
                model_phase1.compile(loss=loss[phases[1] + '_output'], optimizer=optimizer,
                                     metrics=[metrics[phases[1] + '_output']])
                checkpoint_path_phase1 = os.path.join(OUTPUT_DIR, 'phase1', 'cp-{epoch:04d}.ckpt')
                mkdirs(os.path.join(OUTPUT_DIR, 'phase1'))
                cp_callback_phase1 = tf.keras.callbacks.ModelCheckpoint(checkpoint_path_phase1,
                                                                        save_weights_only=False,
                                                                        verbose=1,
                                                                        period=1)
                if load_phase1:
                    for ckpt_i in range(NUM_EPOCHS, 0, -1):
                        if os.path.isfile(checkpoint_path_phase1.format(epoch=ckpt_i)):
                            initial_epoch = ckpt_i
                            model_phase1.load_weights(checkpoint_path_phase1.format(epoch=ckpt_i))
                            # assert False  # dumm proof, most of the case the training is end-to-end, without disturbance and reloading
                            break
                    with open(cls_output_file, 'a+') as fw:
                        fw.write("phase 1 loaded, no training history {}:\n".format(datetime.now()))
                else:
                    history = model_phase1.fit(
                        training_inputs,
                        training_outputs_phase1,
                        validation_data=(val_inputs, val_outputs_phase1),
                        epochs=NUM_EPOCHS,
                        initial_epoch=initial_epoch,
                        batch_size=BATCH_SIZE,
                        callbacks=[cp_callback_phase1, es_callback],
                    )
                    with open(cls_output_file, 'a+') as fw:
                        fw.write("phase 1 training history {}:\n".format(datetime.now()))
                        fw.write(str(history.history) + '\n')
                evaluation_res = model.evaluate(x=test_inputs,
                                                y=test_outputs,
                                                batch_size=BATCH_SIZE,
                                                verbose=1)
                with open(cls_output_file, 'a+') as fw:
                    fw.write("phase 1 evaluation {}:\n".format(datetime.now()))
                    fw.write(str(evaluation_res) + '\n')
            else:
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
            test_inputs_head = [x[:len_head] for x in test_inputs]
            pred = model_exp.predict(test_inputs_head)
            pred = np.round(np.array(pred)).astype(np.int32)
            exp_output_folder = os.path.join(OUTPUT_DIR, 'exp_outputs/')
            mkdirs(exp_output_folder)
            print('marked rationals are saved under {}'.format(exp_output_folder))
            for i, l in enumerate(tqdm_notebook(test)):
                if i == len_head:
                    break
                input_ids = test_input_ids[i]
                if EXP_OUTPUT == 'rnr':
                    pred_intp = rnr_matrix_to_rational_mask(pred[i])[0]
                elif EXP_OUTPUT == 'gru':
                    pred_intp = pred[i].reshape([-1])
                label = test_labels[i]
                gt = test_rations[i].reshape([-1])
                html = convert_res_to_htmls(input_ids, pred_intp, gt, vocab)
                fname = l.annotation_id
                if l.docids is not None:
                    fname += '-' + l.docids[0]
                with open(exp_output_folder + fname + '.html', "w+") as f:
                    f.write('<h1>label: {}</h1>\n'.format('pos' if label == 1 else 'neg'))
                    f.write(html[1] + '<br/><br/>\n' + html[0])

        if exp_benchmark:
            from eraser_benchmark import pred_to_results, get_cls_score, add_cls_scores, remove_rations, \
                extract_rations, rnr_matrix_to_rational_mask
            from eraserbenchmark.eraser import evaluate

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
            results = [pred_to_results(raw_input[i], benchmark_input_ids[i],
                                       (pred[0][i], pred[1][i]),
                                       HARD_SELECTION_COUNT,
                                       HARD_SELECTION_THRESHOLD,
                                       vocab, docs, label_list, EXP_OUTPUT)
                       for i in range(len(pred[0]))]
            pred_softmax = np.hstack([1 - pred[0], pred[0]])
            c_pred_softmax = get_cls_score(
                model, results, docs, label_list, dataset, remove_rations, MAX_SEQ_LENGTH, EXP_OUTPUT, gpu_id=gpu_id)
            s_pred_softmax = get_cls_score(
                model, results, docs, label_list, dataset, extract_rations, MAX_SEQ_LENGTH, EXP_OUTPUT, gpu_id=gpu_id)

            results = [add_cls_scores(res, cls_score, c_cls_score, s_cls_score, label_list) for res, cls_score,
                                                                                                c_cls_score, s_cls_score
                       in
                       zip(results, pred_softmax, c_pred_softmax, s_pred_softmax)]
            anns_saved = set()
            real_results = []
            for ann in results:
                if ann['annotation_id'] not in anns_saved:
                    anns_saved.add(ann['annotation_id'])
                    real_results.append(ann)
            with open(result_fname + 'pkl3', "wb+") as pfout:
                pickle.dump(real_results, pfout)

            evaluate(MODEL_NAME, dataset, BENCHMARK_SPLIT_NAME, train_on_portion, data_dir)
        with open(cls_output_file, 'a+') as fw:
            fw.write('/////////////////experiment ends//////////////////\n\n\n')