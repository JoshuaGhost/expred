# preprocessing.py
from itertools import chain

from copy import deepcopy
from expred.bert_rational_feature import InputRationalExample, convert_examples_to_features
# from config import *
from expred.utils import Evidence
import logging

IRRATIONAL = 0
RATIONAL = 1

logger = logging.getLogger('preprocessing.py')
logger.setLevel(logging.INFO)


def load_bert_features(data, docs, label_list, max_seq_length, merge_evidences, tokenizer):
    input_examples = []
    for ann in data:
        text_a = ann.query
        label = ann.classification
        if not merge_evidences:
            for ev_group in ann.evidences:
                doc_ids = list(set([ev.docid for ev in ev_group]))
                sentences = chain.from_iterable(docs[doc_id] for doc_id in doc_ids)
                flattened_tokens = chain(*sentences)
                text_b = ' '.join(flattened_tokens)
                evidences = ev_group
                input_examples.append(InputRationalExample(guid=None,
                                                           text_a=text_a,
                                                           text_b=text_b,
                                                           label=label,
                                                           evidences=evidences))
        if merge_evidences:
            docids_to_offsets = dict()
            latest_offset = 0
            example_evidences = []
            text_b_tokens = []
            for ev_group in ann.evidences:
                for ev in ev_group:
                    if ev.docid in docids_to_offsets:
                        offset = docids_to_offsets[ev.docid]
                    else:
                        tokens = list(chain.from_iterable(docs[ev.docid]))
                        docids_to_offsets[ev.docid] = latest_offset
                        offset = latest_offset
                        latest_offset += len(tokens)
                        text_b_tokens += tokens
                    example_ev = Evidence(text=ev.text,
                                          docid=ev.docid,
                                          start_token=offset + ev.start_token,
                                          end_token=offset + ev.end_token,
                                          start_sentence=ev.start_sentence,
                                          end_sentence=ev.end_sentence)
                    example_evidences.append(deepcopy(example_ev))
            input_examples.append(InputRationalExample(guid=None,
                                                       text_b=' '.join(text_b_tokens),
                                                       text_a=text_a,
                                                       label=label,
                                                       evidences=example_evidences))
            # print(input_examples[-1].text_b, input_examples[-1].text_a, input_examples[-1].evi)

    features = convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
    return features


# def convert_bert_features(features, with_label_id, with_rations, exp_output='gru'):
#     feature_names = "input_ids input_mask segment_ids".split()
#
#     input_ids, input_masks, segment_ids = \
#         list(map(lambda x: [getattr(f, x) for f in features], feature_names))
#
#     rets = [input_ids, input_masks, segment_ids]
#
#     if with_rations:
#         feature_names.append('rations')
#         rations = [getattr(f, 'rations') for f in features]
#         rations = np.array(rations).reshape([-1, MAX_SEQ_LENGTH, 1])
#         if exp_output == 'interval':
#             rations = np.concatenate([np.zeros((rations.shape[0], 1, 1)),
#                                       rations,
#                                       np.zeros((rations.shape[0], 1, 1))], axis=-2)
#             rations = rations[:, 1:, :] - rations[:, :-1, :]
#             rations_start = (rations > 0)[:, :-1, :].astype(np.int32)
#             rations_end = (rations < 0)[:, 1:, :].astype(np.int32)
#             rations = np.concatenate((rations_start, rations_end), axis=-1)
#         rets.append(rations)
#     else:
#         rets.append(None)
#
#     if with_label_id:
#         feature_names.append('label_id')
#         label_id = [getattr(f, 'label_id') for f in features]
#         labels = np.array(label_id).reshape(-1, 1)
#         rets.append(labels)
#     else:
#         rets.append(None)
#     return rets


# def preprocess(data, docs, label_list, dataset_name, max_seq_length, exp_output, merge_evidences, tokenizer):
#     features = load_bert_features(data, docs, label_list, max_seq_length, merge_evidences, tokenizer)
#
#     with_rations = ('cls' not in dataset_name)
#     with_lable_id = ('seq' not in dataset_name)
#
#     return convert_bert_features(features, with_lable_id, with_rations, exp_output)
