# bert_rational_feature.py
from tqdm import tqdm_notebook
import expred.rationale_tokenization as tokenization
import logging

IRRATIONAL = 0
RATIONAL = 1

logger = logging.getLogger('feature converter')
logger.setLevel(logging.INFO)


class InputRationalExample(object):
    def __init__(self, guid, text_a, text_b=None, label=None, evidences=None):
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
        self.evidences = evidences


class InputRationalFeatures(object):
    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id,
                 rations=None,
                 is_real_example=True):
        self.rations = rations
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def convert_single_rational_example(ex_index, example, label_list, max_seq_length, tokenizer):

    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
              break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()

    label_map = {}
    for (i, label) in enumerate(label_list):
        label_map[label] = i

    tokens_a = tokenizer.tokenize(example.text_a)
    tokens_b = None  # no tokens_b in our tasks
    if example.text_b:
        tokens_b = tokenizer.tokenize(example.text_b, example.evidences)

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

    if ex_index < 5:
        logger.info("*** Example ***")
        logger.info("guid: %s" % (example.guid))
        logger.info("tokens: %s" % " ".join([tokenization.printable_text(x) for x in tokens]))
        logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        logger.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        logger.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        logger.info("rations: %s" % " ".join([str(x) for x in rations]))
        logger.info('')
        logger.info("label: %s (id = %d)" % (example.label, label_id))

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
    for (ex_index, example) in enumerate(tqdm_notebook(examples, desc="Converting examples to features")):
        if ex_index % 10000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))
        feature = convert_single_rational_example(ex_index, example, label_list,
                                                  max_seq_length, tokenizer)
        features.append(feature)
    return features
