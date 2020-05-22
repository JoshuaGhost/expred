IRRATIONAL = 0
RATIONAL = 1

import tensorflow

if tensorflow.__version__.startswith('2'):
    import tensorflow.compat.v1 as tf

    tf.disable_v2_behavior()
else:
    import tensorflow as tf

from bert.tokenization import FullTokenizer, BasicTokenizer, \
    convert_to_unicode, whitespace_tokenize, convert_ids_to_tokens
import tensorflow_hub as hub
from config import *
from bert.run_classifier import InputFeatures, PaddingInputExample, _truncate_seq_pair
#from tqdm import tqdm_notebook as tqdm
from tqdm import tqdm


class BasicTokenizerWithRation(BasicTokenizer):  # usability test passed :)
    def __init__(self, do_lower_case=True):
        super(BasicTokenizerWithRation, self).__init__(do_lower_case)

    def _parse_rations(self, s):
        tokens = s
        rationality = IRRATIONAL
        ret_tokens, ret_rations = [], []
        while len(tokens) > 0:
            t = tokens.pop(0)
            if t[:2] == '</':
                rationality = IRRATIONAL
            elif t[0] == '<':
                rationality = RATIONAL
            else:
                ret_tokens.append(t)
                ret_rations.append(rationality)
        return ret_tokens, ret_rations

    def tokenize(self, text):
        text = convert_to_unicode(text)
        text = self._clean_text(text)
        orig_tokens = whitespace_tokenize(text)
        orig_tokens, orig_rations = self._parse_rations(orig_tokens)
        split_tokens = []
        split_rations = []
        for token, ration in zip(orig_tokens, orig_rations):
            if self.do_lower_case:
                token = token.lower()
                token = self._run_strip_accents(token)
            sub_tokens = self._run_split_on_punc(token)
            sub_tokens = ' '.join(sub_tokens).strip().split()
            if len(sub_tokens) > 0:
                split_tokens.extend(sub_tokens)
                split_rations.extend([ration] * len(sub_tokens))
        return zip(split_tokens, split_rations)


# --------------------------------------------------------------------------------------

class FullTokenizerWithRations(FullTokenizer):  # Test passed :)

    def __init__(self, vocab_file, do_lower_case=True):
        self.basic_rational_tokenizer = BasicTokenizerWithRation(do_lower_case=do_lower_case)
        super(FullTokenizerWithRations, self).__init__(vocab_file, do_lower_case)

    def tokenize(self, text):
        split_tokens = []
        split_rations = []
        for token, ration in self.basic_rational_tokenizer.tokenize(text):
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
# --------------------------------------------------------------------------------------

class InputRationalFeatures(InputFeatures):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 rations=None,
                 is_real_example=True):
        self.rations = rations
        super(InputRationalFeatures, self).__init__(input_ids, input_mask, segment_ids, label_id, is_real_example)


def convert_single_rational_example(ex_index, example, label_list, max_seq_length, tokenizer):
    if isinstance(example, PaddingInputExample):
        return InputRationalFeatures(
            input_ids=[0] * max_seq_length,
            input_mask=[0] * max_seq_length,
            segment_ids=[0] * max_seq_length,
            label_id=0,
            rations=[IRRATIONAL] * max_seq_length,
            is_real_example=False)

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None  # no tokens_b in our tasks
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b)

    if tokens_b:
        _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[0:(max_seq_length - 2)]

    tokens = []
    segment_ids = []
    rations = []
    tokens.append("[CLS]")
    segment_ids.append(0)
    rations.append(IRRATIONAL)
    for token, ration in tokens_a:
        tokens.append(token)
        segment_ids.append(0)
        rations.append(ration)
    tokens.append("[SEP]")
    segment_ids.append(0)
    rations.append(IRRATIONAL)

    if tokens_b:
        for token, ration in tokens_b:
            tokens.append(token)
            segment_ids.append(1)
            rations.append(ration)
        tokens.append("[SEP]")
        segment_ids.append(1)
        rations.append(IRRATIONAL)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    input_mask = [1] * len(input_ids)

    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        rations.append(IRRATIONAL)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(rations) == max_seq_length

    label_id = label_map[example.label]
    '''
    if ex_index < 5:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
            [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("rations: %s" % " ".join([str(x) for x in rations]))
        tf.logging.info('')
        tf.logging.info("label: %s (id = %d)" % (example.label, label_id))
    '''
    feature = InputRationalFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        label_id=label_id,
        rations=rations,
        is_real_example=True)
    return feature


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
    features = []
    for (ex_index, example) in enumerate(tqdm(examples, desc="Converting examples to features")):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_rational_example(ex_index, example, label_list,
                                                  max_seq_length, tokenizer)
        features.append(feature)
    return features


def convert_ids_to_token_list(input_ids, vocab):
    iv_vocab = {input_id: wordpiece for wordpiece, input_id in vocab.items()}

    token_list = convert_ids_to_tokens(iv_vocab, input_ids)
    return token_list
