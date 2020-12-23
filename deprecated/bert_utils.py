from transformers.tokenization_bert import get_vocab
import tensorflow

if tensorflow.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf
import tensorflow_hub as hub
from bert.tokenization import load_vocab
from config import *


def get_vocab(config):
    with tf.Graph().as_default():
        bert_module = hub.Module(BERT_MODEL_HUB)
        tokenization_info = bert_module(signature="tokenization_info",
                                        as_dict=True)
        with tf.Session(config=config) as sess:
            vocab_file = sess.run(tokenization_info["vocab_file"])
    vocab = load_vocab(vocab_file)
    return vocab
