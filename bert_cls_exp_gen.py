# imports
import argparse
from datetime import datetime

import numpy as np
from tensorflow.python.keras.backend import set_session
# from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm

from display_rational import convert_res_to_htmls
from eraser_benchmark import rnr_matrix_to_rational_mask
from losses import imbalanced_bce_bayesian, imbalanced_bce_resampling
from losses import rnr_matrix_loss
from metrices import *
from utils import *

if tensorflow.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf

LEARNING_RATE = 1e-5
MAX_SEQ_LENGTH = 512
HARD_SELECTION_COUNT = None
HARD_SELECTION_THRESHOLD = 0.5

if __name__ == '__main__':
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
    parser.add_argument('--benchmark_split', type=str, default='test', choices='test train val'.split())  # gru, rnr
    parser.add_argument('--freeze_cls', action='store_true')
    parser.add_argument('--freeze_exp', action='store_true')
    parser.add_argument('--train_cls_first', action='store_true')
    parser.add_argument('--train_exp_first', action='store_true')
    parser.add_argument('--cls_only', action='store_true')
    parser.add_argument('--exp_only', action='store_true')

    parser.add_argument('--train_on_annotations_only', action='store_true')
    parser.add_argument('--train_without_annotations', action='store_true')
    parser.add_argument('--eval_annotations', action='store_true')
    parser.add_argument('--annotation_source', type=str)

    parser.add_argument('--use_bucket', action='store_true')
    parser.add_argument('--exp_structure', type=str, default='none', choices='gru rnr none'.split())  # gru, rnr
    parser.add_argument('--delete_checkpoints', action='store_true')
    parser.add_argument('--merge_evidences', action='store_true')
    parser.add_argument('--train_on_portion', type=float, default=0)
    parser.add_argument('--start_from_phase1', action='store_true')
    parser.add_argument('--save_machine_rationale', action='store_true')
    parser.add_argument('--machine_rationale_folder', type=str, default='eraserbenchmark/machine_rationale')
    parser.add_argument('--load_phase1', action='store_true')
    parser.add_argument('--force_recache', action='store_true')
    parser.add_argument('--pooling', type=str, default='first', choices=['first', 'mean'])
    parser.add_argument('--cache_dir', type=str, default='/tmp/interpretation_by_design')
    parser.add_argument('--bert_size', type=str, default='base', choices=['base', 'large'])
    parser.add_argument('--rebalance_approach', type=str, default='resampling', choices=['resampling', 'bayesian'])
    parser.add_argument('--data_dir', type=str)
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

    save_machine_rationale = args.save_machine_rationale

    cache_dir = args.cache_dir
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
    BENCHMARK_SPLIT_NAME = args.benchmark_split
    train_on_portion = args.train_on_portion
    freeze_cls = args.freeze_cls
    freeze_exp = args.freeze_exp
    train_cls_first = args.train_cls_first
    train_exp_first = args.train_exp_first
    start_from_phase1 = args.start_from_phase1
    load_phase1 = args.load_phase1
    pooling = args.pooling
    exp_structure = exp_structure
    bert_size = args.bert_size
    rebalance_approach = args.rebalance_approach
    USE_BUCKET = args.use_bucket
    data_dir = os.path.join(args.data_dir, dataset)
    len_head = args.len_viz_head
    train_on_annotations_only = args.train_on_annotations_only
    train_without_annotations = args.train_without_annotations
    cls_only = args.cls_only or (train_on_annotations_only or train_without_annotations)
    exp_only = args.exp_only
    eval_annotations = args.eval_annotations
    force_recache = args.force_recache
    annotation_source = args.annotation_source
    machine_rationale_folder = args.machine_rationale_folder
    assert (not (freeze_cls and freeze_exp))
    assert (not (train_exp_first and train_cls_first))
    assert not ((freeze_cls or freeze_exp) and (
            train_exp_first or train_cls_first))  # can't freeze both in the same time
    # assert (not (human_annotation and do_train))

    if bert_size == 'base':
        BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
    elif bert_size == 'large':
        BERT_MODELL_HUB = 'https://tfhub.dev/tensorflow/bert_en_uncased_L-24_H-1024_A-16/1'

    from eraser_benchmark import remove_rations, extract_rations

    training_data_decorator = None

    # input data relevant settings

    data_cache_name = [f'seqlen_{MAX_SEQ_LENGTH}',
                       f'dataset_{dataset}',
                       f'exp_structure_{exp_structure}']
    if train_on_annotations_only:
        data_cache_name.append('train_on_annotations_only')
        training_data_decorator = extract_rations
    elif train_without_annotations:
        data_cache_name.append('train_without_annotations')
        training_data_decorator = remove_rations
    else:
        data_cache_name.append('complete_sentences')
        training_data_decorator = None
    if annotation_source is not None:
        data_cache_name.append(annotation_source)
    if train_on_portion != 0:
        data_cache_name.append(f'train_on_portion_{train_on_portion}')
    if merge_evidences:
        data_cache_name.append('merged_evidences')
    else:
        data_cache_name.append('separated_evidences')

    # input data irrelevant (model) settings

    import copy

    model_name = copy.deepcopy(data_cache_name)
    model_name += [f'bert_{bert_size}',
                   f'bce_rebalance_approach_{rebalance_approach}',
                   f'pooling_{pooling}',
                   f'learning_rate_{LEARNING_RATE}']
    if par_lambda is None:
        model_name.append('no_weight')
    else:
        model_name.append('par_lambda_{}'.format(par_lambda))
    if cls_only:
        model_suffix = 'cls_only'
    elif exp_only:
        model_suffix = 'exp_only'
    elif freeze_cls:
        model_suffix = 'freeze_cls'
    elif freeze_exp:
        model_suffix = 'freeze_exp'
    elif train_cls_first:
        model_suffix = 'train_cls_first'
    elif train_exp_first:
        model_suffix = 'train_exp_first'
    else:
        model_suffix = ''
    if eval_annotations:
        model_suffix += '_eval_annotations'
    model_name.append(model_suffix)

    data_cache_name = '_'.join(data_cache_name) + '_inputdata_cache'
    model_name = '_'.join(model_name)
    output_dir = os.path.join('model_checkpoints', model_name)

    if exp_structure == 'gru':
        if rebalance_approach == 'resampling':
            loss_function = imbalanced_bce_resampling
        else:
            loss_function = imbalanced_bce_bayesian
    elif exp_structure == 'rnr':
        loss_function = rnr_matrix_loss

    if USE_BUCKET:
        BUCKET = 'bert-base-uncased-test0'  # @param {type:"string"}
        output_dir = 'gs://{}/{}'.format(BUCKET, output_dir)
        from google.colab import auth

        auth.authenticate_user()
    if DO_DELETE:
        try:
            tf.gfile.DeleteRecursively(output_dir)
        except:
            pass

    mkdirs(output_dir)
    print('***** Model output directory: {} *****'.format(output_dir))

    # initializing graph and session
    graph = tf.get_default_graph()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = gpu_id
    config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
    sess = tf.Session(config=config)
    set_session(sess)

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

    train, val, test = load_datasets(data_dir)
    if train_on_portion != 0:
        train = train[:int(len(train) * train_on_portion)]
    docids = set(chain.from_iterable(extract_doc_ids_from_annotations(d) for d in [train, val, test]))
    docs = load_documents(data_dir, docids)

    from bert_data_preprocessing_rational_eraser import preprocess


    @cache_decorator(os.path.join(cache_dir, 'cache', data_cache_name + '_eraser_format'), force_recache=force_recache)
    def preprocess_wrapper(*data_inputs, docs=docs):
        ret = []
        for data in data_inputs:
            ret.append(
                preprocess(data, docs, label_list, dataset, MAX_SEQ_LENGTH, exp_structure, merge_evidences,
                           data_decorator=training_data_decorator, gpu_id=gpu_id))
        return ret


    rets_train, rets_val, rets_test = preprocess_wrapper(train, val, test, docs=docs)

    train_input_ids, train_input_masks, train_segment_ids, train_rations, train_labels = rets_train
    val_input_ids, val_input_masks, val_segment_ids, val_rations, val_labels = rets_val
    test_input_ids, test_input_masks, test_segment_ids, test_rations, test_labels = rets_test

    for i, input_ids in enumerate([train_input_ids, val_input_ids, test_input_ids]):
        for j, ids in enumerate(input_ids):
            try:
                a = ids[ids.index(102) + 1:].index(102)
            except ValueError:
                print(i, j)
                print(ids)
                raise ValueError


    def expand_on_evidences(data):
        from eraserbenchmark.rationale_benchmark.utils import Annotation
        expanded_data = []
        for ann in tqdm(data):
            for ev_group in ann.evidences:
                new_ann = Annotation(annotation_id=ann.annotation_id,
                                     query=ann.query,
                                     evidences=frozenset([ev_group]),
                                     classification=ann.classification)
                expanded_data.append(new_ann)
        return expanded_data


    # hyper-parameters of BERT's
    WARMUP_PROPORTION = 0.1

    num_train_steps = int(len(train_input_ids) / BATCH_SIZE * float(NUM_EPOCHS))
    num_warmup_steps = int(num_train_steps * WARMUP_PROPORTION)

    from bert_utils import get_vocab

    vocab = get_vocab(config)

    # building models
    from model import BertLayer
    from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Reshape, Multiply, Concatenate, Dot, Lambda, Softmax

    from metrices import sp_precision_wrapper, sp_recall_wrapper

    DIM_DENSE_CLS = 256
    NUM_GRU_UNITS_BERT_SEQ = 128
    NUM_INTERVAL_LSTM_WIDTH = 100

    if par_lambda is None:
        loss_weights = None
    else:
        loss_weights = {'cls_output': 1,
                        'exp_output': par_lambda}
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


    # training, evaluation and inference
    import os

    SAVE_CHECKPOINTS_STEPS = 500
    SAVE_SUMMARY_STEPS = 100

    checkpoint_path = os.path.join(output_dir, 'cp-{epoch:04d}.ckpt')
    checkpoint_dir = os.path.dirname(checkpoint_path)
    cls_output_file = os.path.join(output_dir, 'output.txt')

    RES_FOR_BENCHMARK_FNAME = model_name + '_' + BENCHMARK_SPLIT_NAME

    with graph.as_default():
        set_session(sess)
        model, model_cls, model_exp = build_model()
        if exp_structure == 'none':
            model = model_cls
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
                checkpoint_path_phase0 = os.path.join(output_dir, 'phase0', 'cp-{epoch:04d}.ckpt')
                mkdirs(os.path.join(output_dir, 'phase0'))
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
                checkpoint_path_phase1 = os.path.join(output_dir, 'phase1', 'cp-{epoch:04d}.ckpt')
                mkdirs(os.path.join(output_dir, 'phase1'))
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
            pred = model.predict(x=test_inputs)
            evaluation_output_name = ['test_loss', 'test_accuracy']
            if not exp_only:
                from sklearn.metrics import f1_score

                macro_f1 = f1_score(np.squeeze(test_outputs['cls_output']),
                                    np.round(np.squeeze(pred)),
                                    average='macro')
                evaluation_res.append(macro_f1)
                evaluation_output_name.append('macro_f1')
            with open(cls_output_file, 'a+') as fw:
                fw.write("{}:\n".format(datetime.now()))
                for name, value in zip(evaluation_output_name, evaluation_res):
                    fw.write(f"{name}: {value} ")
                fw.write('\n')

        if exp_visualize:
            test_inputs_head = [x[:len_head] for x in test_inputs]
            pred = model_exp.predict(test_inputs_head)
            pred = np.round(np.array(pred)).astype(np.int32)
            exp_vis_folder = os.path.join(output_dir, 'exp_outputs/')
            mkdirs(exp_vis_folder)
            print('marked rationals are saved under {}'.format(exp_vis_folder))
            for i, l in enumerate(tqdm(test)):
                if i == len_head:
                    break
                input_ids = test_input_ids[i]
                if exp_structure == 'rnr':
                    pred_intp = rnr_matrix_to_rational_mask(pred[i])[0]
                elif exp_structure == 'gru':
                    pred_intp = pred[i].reshape([-1])
                label = test_labels[i]
                gt = test_rations[i].reshape([-1])
                html = convert_res_to_htmls(input_ids, pred_intp, gt, vocab)
                fname = l.annotation_id
                if l.docids is not None:
                    fname += '-' + l.docids[0]
                with open(exp_vis_folder + fname + '.html', "w+") as f:
                    f.write('<h1>label: {}</h1>\n'.format('pos' if label == 1 else 'neg'))
                    f.write(html[1] + '<br/><br/>\n' + html[0])

        if exp_benchmark or save_machine_rationale:
            from eraser_benchmark import pred_to_results, get_cls_score, add_cls_scores, remove_rations, \
                extract_rations, rnr_matrix_to_rational_mask, ann_to_exp_output, convert_res_to_csv

            from eraserbenchmark.eraser import evaluate
            result_fname = RES_FOR_BENCHMARK_FNAME + '.jsonl'
            result_fname = os.path.join('eraserbenchmark', 'annotated_by_exp', result_fname)
            if BENCHMARK_SPLIT_NAME == 'test':
                benchmark_inputs, raw_input, benchmark_input_ids, benchmark_outputs = test_inputs, test, test_input_ids, test_outputs
            elif BENCHMARK_SPLIT_NAME == 'val':
                benchmark_inputs, raw_input, benchmark_input_ids, benchmark_outputs = val_inputs, val, val_input_ids, val_outputs
            elif BENCHMARK_SPLIT_NAME == 'train':
                benchmark_inputs, raw_input, benchmark_input_ids, benchmark_outputs = training_inputs, train, train_input_ids, training_outputs

            pred = model.predict(x=benchmark_inputs)
            if eval_annotations:
                cls_pred = pred if exp_structure == 'none' else pred[0]
                exp_pred = benchmark_outputs['exp_output']
            elif exp_only:
                cls_pred = [[0] for i in pred]
                exp_pred = pred
            elif cls_only:
                cls_pred = pred
                exp_pred = [[0 for j in range(MAX_SEQ_LENGTH)] for i in pred]
            else:
                cls_pred = pred[0]
                exp_pred = pred[1]

            from bert_with_ration import FullTokenizerWithRations

            full_tokenizer = FullTokenizerWithRations.create_tokenizer_from_hub_module(gpu_id)
            results = [pred_to_results(raw_input[i], benchmark_input_ids[i],
                                       (cls_pred[i], exp_pred[i]),
                                       HARD_SELECTION_COUNT,
                                       HARD_SELECTION_THRESHOLD,
                                       vocab, docs, label_list, exp_structure)
                       for i in range(len(cls_pred))]
            bert_tokens = [res[1] for res in results]
            bert_tokens = {annotation_id: tokens for annotation_id, tokens in bert_tokens}
            results = [res[0] for res in results]

            if save_machine_rationale:
                ref = {ri.annotation_id: ri for ri in raw_input}
                machine_rationale_folder = os.path.join(machine_rationale_folder, dataset)
                if not os.path.isdir(machine_rationale_folder):
                    os.mkdir(machine_rationale_folder)
                if exp_only:
                    exp_output_res = [ann_to_exp_output(ann, ref, keep_correct_predictions_only=False) for ann in results] # for two-stage model where the first stage has exp only. We don't care about the correctness of the cls prediction in the first stage
                else:
                    exp_output_res = [ann_to_exp_output(ann, ref) for ann in results]
                exp_output_res = list(filter(lambda x: len(x) > 0, exp_output_res))
                exp_output_fname = RES_FOR_BENCHMARK_FNAME + '_' \
                                   + BENCHMARK_SPLIT_NAME + '_exp_output.jsonl'
                exp_output_fname = os.path.join(machine_rationale_folder, exp_output_fname)
                import json
                with open(exp_output_fname, 'w+') as fout:
                    for res in exp_output_res:
                        res['evidences'] = [res['evidences']]
                        json.dump(res, fout)
                        fout.write('\n')
                # exp_output_docs_dir = os.path.join(exp_output_folder, 'docs')
                exp_csv_fname = exp_output_fname[:-5] + 'csv'
                exp_csv = convert_res_to_csv(results, benchmark_input_ids, bert_tokens, ref)
                exp_csv.to_csv(exp_csv_fname)
            elif exp_benchmark:
                pred_softmax = np.hstack([1 - cls_pred, cls_pred])
                c_pred_softmax = get_cls_score(
                    model, results, docs, label_list, dataset, remove_rations, MAX_SEQ_LENGTH, exp_structure,
                    gpu_id=gpu_id, tokenizer=full_tokenizer)
                s_pred_softmax = get_cls_score(
                    model, results, docs, label_list, dataset, extract_rations, MAX_SEQ_LENGTH, exp_structure,
                    gpu_id=gpu_id, tokenizer=full_tokenizer)

                print(len(c_pred_softmax))
                print((c_pred_softmax[0]))

                results = [add_cls_scores(res,
                                          cls_score,
                                          c_cls_score,
                                          s_cls_score,
                                          label_list) for res, cls_score, c_cls_score, s_cls_score in zip(results,
                                                                                                          pred_softmax,
                                                                                                          c_pred_softmax,
                                                                                                          s_pred_softmax)]
                anns_saved = set()
                real_results = []
                for ann in results:
                    if ann['annotation_id'] not in anns_saved:
                        anns_saved.add(ann['annotation_id'])
                        real_results.append(ann)
                with open(result_fname + 'pkl3', "wb+") as pfout:
                    pickle.dump(real_results, pfout)

                evaluate(model_name, dataset, BENCHMARK_SPLIT_NAME, train_on_portion, data_dir)
        with open(cls_output_file, 'a+') as fw:
            fw.write('/////////////////experiment ends//////////////////\n\n\n')
