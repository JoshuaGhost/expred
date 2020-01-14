import tensorflow as tf
import tensorflow_hub as hub

import numpy as np

from bert.run_classifier import InputExample
from bert_with_ration import BasicTokenizerWithRation, \
                             convert_examples_to_features, \
                             FullTokenizerWithRations, \
                             InputRationalFeatures
from config import *

def create_tokenizer_from_hub_module():
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default(): # basically useless, but good practice to specify the graph using, even it sets the default graph as the default graph
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = '0'
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1
        with tf.Session(config=config) as sess: # create a new session, with session we can setup even remote computation
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])

    return FullTokenizerWithRations(vocab_file=vocab_file, do_lower_case=do_lower_case)


def load_bert_features(data, label_list, max_seq_length, data_columns, label_column):
    tokenizer = create_tokenizer_from_hub_module()
    if len(data_columns) == 1:
        input_examples = data.apply(lambda x: InputExample(guid=None,
                                                           text_a=x[data_columns[0]], 
                                                           text_b=None,
                                                           label=x[label_column]), axis = 1)
    else:
        input_examples = data.apply(lambda x: InputExample(guid=None,
                                                           text_a=x[data_columns[0]],
                                                           text_b=x[data_columns[1]],
                                                           label=x[label_column]), axis = 1)
    features = convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
    return features


def convert_bert_features(features, with_label_id, with_rations, exp_output='gru'):
    
    feature_names = "input_ids input_mask segment_ids".split()
    
    input_ids, input_masks, segment_ids = \
        list(map(lambda x: [getattr(f, x) for f in features], feature_names))
    
    rets = [input_ids, input_masks, segment_ids]
    
    if with_rations:
        feature_names.append('rations')
        rations = [getattr(f, 'rations') for f in features]
        rations = np.array(rations).reshape([-1, MAX_SEQ_LENGTH, 1])
        if exp_output == 'interval':
            rations = np.concatenate([np.zeros((rations.shape[0], 1, 1)), 
                                      rations, 
                                      np.zeros((rations.shape[0], 1, 1))], axis=-2)
            rations = rations[:,1:,:] - rations[:,:-1,:]
            rations_start = (rations>0)[:, :-1, :].astype(np.int32)
            rations_end = (rations<0)[:, 1:, :].astype(np.int32)
            rations = np.concatenate((rations_start, rations_end), axis=-1)
        rets.append(rations)
    else:
        rets.append(None)
        
    if with_label_id:
        feature_names.append('label_id')
        label_id = [getattr(f, 'label_id') for f in features]
        labels = np.array(label_id).reshape(-1, 1)
        rets.append(labels)
    else:
        rets.append(None)
        
    return rets