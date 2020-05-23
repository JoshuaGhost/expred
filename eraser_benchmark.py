from itertools import chain

import numpy as np
import re
from copy import deepcopy

from bert_data_preprocessing_rational_eraser import convert_bert_features, create_tokenizer_from_hub_module
from bert_with_ration_eraser import InputRationalExample, convert_examples_to_features
from eraserbenchmark.eraser_utils import extract_doc_ids_from_annotations
from utils import convert_subtoken_ids_to_tokens
from bert_with_ration_eraser import convert_ids_to_token_list
from eraserbenchmark.rationale import Annotation
from eraserbenchmark.rationale_benchmark.utils import Evidence


def flatten_rations(rations, len_sentence):
    rations = [{'end_token': 0, 'start_token': 0}] \
              + sorted(rations, key=lambda x: x['start_token']) \
              + [{'start_token': len_sentence, 'end_token': len_sentence}]
    return rations


def remove_rations(sentence, rations, tokenizer=None, sub='.', combine_subtokens=False):
    sentence = sentence.lower().split()
    if isinstance(rations, list): # a list of Evidence-s
        rations = [e.__dict__ for e in rations]
    else: # an Annotation
        rations = rations['rationales'][0]['hard_rationale_predictions']
    rations = flatten_rations(rations, len(sentence))
    ret = []
    for rat_id, rat in enumerate(rations[:-1]):
        if combine_subtokens or tokenizer is None:
            rep_count = rat['end_token'] - rat['start_token']
        else:
            rep_str = ' '.join(sentence[rat['start_token']: rat['end_token']])
            rep_count = len(tokenizer.tokenize(rep_str))
        ret += [sub] * rep_count + sentence[rat['end_token']:rations[rat_id + 1]['start_token']]
    return ' '.join(ret)


def extract_rations(sentence, rations, tokenizer=None, sub='.', combine_subtokens=False):
    sentence = sentence.lower().split()
    if isinstance(rations, list): # a list of Evidence-s
        rations = [e.__dict__ for e in rations]
    else: # an Annotation
        rations = rations['rationales'][0]['hard_rationale_predictions']
    rations = flatten_rations(rations, len(sentence))
    ret = []
    for rat_id, rat in enumerate(rations[:-1]):
        if combine_subtokens or tokenizer is None:
            rep_count = rations[rat_id + 1]['start_token'] - rations[rat_id]['end_token']
        else:
            rep_str = ' '.join(sentence[rations[rat_id]['end_token']: rations[rat_id + 1]['start_token']])
            rep_count = len(tokenizer.tokenize(rep_str))
        ret += sentence[rat['start_token']: rat['end_token']] + [sub] * (rep_count)
    return ' '.join(ret)


def ce_load_bert_features(rationales, docs, label_list, decorate, max_seq_length, gpu_id, tokenizer):
    #tokenizer = create_tokenizer_from_hub_module(gpu_id)
    input_examples = []
    for r_idx, rational in enumerate(rationales):
        text_a = rational['query']
        docids = rational['docids']
        sentences = chain.from_iterable(docs[docid] for docid in docids)
        flattened_tokens = chain(*sentences)
        text_b = ' '.join(flattened_tokens)
        #print(rational)
        text_b = decorate(text_b, rational, tokenizer=tokenizer)
        label = rational['classification']
        evidences = None
        input_examples.append(InputRationalExample(guid=None,
                                                   text_a=text_a,
                                                   text_b=text_b,
                                                   label=label,
                                                   evidences=evidences))
    features = convert_examples_to_features(input_examples, label_list, max_seq_length, tokenizer)
    return features


def ce_preprocess(rationales, docs, label_list, dataset_name, decorate, max_seq_length, exp_output, gpu_id, tokenizer):
    features = ce_load_bert_features(rationales, docs, label_list, decorate, max_seq_length, gpu_id, tokenizer)

    with_rations = ('cls' not in dataset_name)
    with_lable_id = ('seq' not in dataset_name)

    return convert_bert_features(features, with_lable_id, with_rations, exp_output)


def get_cls_score(model, rationales, docs, label_list, dataset, decorate, max_seq_length, exp_output, gpu_id, tokenizer):
    rets = ce_preprocess(rationales, docs, label_list, dataset, decorate, max_seq_length, exp_output, gpu_id, tokenizer)
    _input_ids, _input_masks, _segment_ids, _rations, _labels = rets

    _inputs = [_input_ids, _input_masks, _segment_ids]
    _pred = model.predict(_inputs)
    if exp_output == 'none':
        _pred = [_pred,]
    return (np.hstack([1 - _pred[0], _pred[0]]))


def add_cls_scores(res, cls, c, s, label_list):
    res['classification_scores'] = {label_list[0]: cls[0], label_list[1]: cls[1]}
    res['comprehensiveness_classification_scores'] = {label_list[0]: c[0], label_list[1]: c[1]}
    res['sufficiency_classification_scores'] = {label_list[0]: s[0], label_list[1]: s[1]}
    return res


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
    docid = list(extract_doc_ids_from_annotations([raw_input]))[0]
    ev = {'docid': docid,
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


def rnr_matrix_to_rational_mask(rnr_matrix):
    start_logits, end_logits = rnr_matrix[:, :1], rnr_matrix[:, 1:]
    starts = np.round(start_logits).reshape((-1, 1))
    ends = np.triu(end_logits)
    ends = starts * ends
    ends_args = np.argmax(ends, axis=1)
    ends = np.zeros_like(ends)
    for i in range(len(ends_args)):
        ends[i, ends_args[i]] = 1
    ends = starts * ends
    ends = np.sum(ends, axis=0, keepdims=True)
    rational_mask = np.cumsum(starts.reshape((1, -1)), axis=1) - np.cumsum(ends, axis=1) + ends
    return rational_mask


def pred_to_results(raw_input, input_ids, pred, hard_selection_count, hard_selection_threshold, vocab, docs, label_list,
                    exp_output):
    cls_pred, exp_pred = pred
    if exp_output == 'rnr':
        exp_pred = rnr_matrix_to_rational_mask(exp_pred)
    exp_pred = exp_pred.reshape((-1,)).tolist()
    try:
        docid = list(raw_input.evidences)[0][0].docid
    except IndexError:
        docid = raw_input.annotation_id  # posR_161 of movies reviews has no evidence
    raw_sentence = ' '.join(list(chain.from_iterable(docs[docid])))
    raw_sentence = re.sub('\x12', '', raw_sentence)
    raw_sentence = raw_sentence.lower().split()
    try:
        token_ids, exp_pred = extract_texts(input_ids, exp_pred, text_a=False, text_b=True)
    except ValueError:
        print(docid)
        print(input_ids)
        print(convert_subtoken_ids_to_tokens(input_ids, vocab))
        raise ValueError
    token_list, exp_pred = convert_subtoken_ids_to_tokens(token_ids, vocab, exp_pred, raw_sentence)
    result = {'annotation_id': raw_input.annotation_id, 'query': raw_input.query}
    ev_groups = []
    result['docids'] = [docid]
    result['rationales'] = [{'docid': docid}]
    for ev in rational_bits_to_ev_generator(token_list, raw_input, exp_pred, hard_selection_count,
                                            hard_selection_threshold):
        ev_groups.append(ev)
    result['rationales'][-1]['hard_rationale_predictions'] = ev_groups
    if exp_output != 'rnr':
        result['rationales'][-1]['soft_rationale_predictions'] = exp_pred + [0] * (len(raw_sentence) - len(token_list))
    result['classification'] = label_list[int(round(cls_pred[0]))]
    return result, (result['annotation_id'], token_list)


def prediction_correct(ann, ref):
    return ann['classification'] == ref[ann['annotation_id']].classification


def ann_to_exp_output(ann, ref, keep_correct_predictions_only=True):
    res = {}
    if not prediction_correct(ann, ref) and keep_correct_predictions_only:
        return res
    res['annotation_id'] = ann['annotation_id']
    res['classification'] = ann['classification']
    res['evidences'] = ann['rationales'][-1]['hard_rationale_predictions']
    res['query'] = ref[ann['annotation_id']].query
    return res


def convert_res_to_csv(results, input_ids, bert_tokens, ref):
    from pandas import DataFrame
    res = {'annotation_id': [],
           'type': [], 'text': [],
           'query': [],
           'classification': []}
    for ids, ann in zip(input_ids, results):
        if not prediction_correct(ann, ref):
            continue
        for i in range(3): # cs, ma, no_ma
            res['annotation_id'].append(ann['annotation_id'])
            res['query'].append(ann['query'])
            res['classification'].append(ann['classification'])
        res['type'].append('cs')
        tokens = ' '.join(bert_tokens[ann['annotation_id']])
        res['text'].append(tokens)
        res['type'].append('ma')
        res['text'].append(extract_rations(tokens, ann))
        res['type'].append('no_ma')
        res['text'].append(remove_rations(tokens, ann))
    return DataFrame(res)