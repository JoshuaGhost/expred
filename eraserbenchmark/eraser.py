#
import argparse
import json
import logging
import os
import pprint

from collections import Counter, defaultdict, namedtuple
from dataclasses import dataclass
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import torch

from eraserbenchmark.rationale_benchmark.utils import (
    Annotation,
    Evidence,
    annotations_from_jsonl,
    load_jsonl,
    load_documents,
    load_flattened_documents
)

from eraserbenchmark.scoring import *
from eraserbenchmark.position_scored_document import PositionScoredDocument
from eraserbenchmark.verify_instances import _has_classifications, \
    _has_hard_predictions, \
    _has_soft_predictions, \
    _has_soft_sentence_predictions, \
    verify_instances
import pickle

logging.basicConfig(level=logging.DEBUG,
                    format='%(relativeCreated)6d %(threadName)s %(message)s')

dataset_name = 'fever'
data_dir = '/home/zzhang/.keras/datasets/{}/'.format(dataset_name)
split_name = 'test'

strict = True
strict = False

iou_thresholds = [0.5]








from typing import Any, Callable, Dict, List, Tuple
from collections import Counter
import logging

def verify_instance(instance: dict, docs: Dict[str, list]):
    error = False
    docids = []
    # verify the internal structure of these instances is correct:
    # * hard predictions are present
    # * start and end tokens are valid
    # * soft rationale predictions, if present, must have the same document length

    for rat in instance['rationales']:
        docid = rat['docid']
        if docid not in docid:
            error = True
            print(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} could not be found as a preprocessed document! Gave up on additional processing.')
            continue
        doc_length = len(docs[docid])
        #print(docs[docid])
        #assert False
        for h1 in rat.get('hard_rationale_predictions', []):
            # verify that each token is valid
            # verify that no annotations overlap
            for h2 in rat.get('hard_rationale_predictions', []):
                if h1 == h2:
                    continue
                try:
                    if len(set(range(h1['start_token'], h1['end_token'])) & set(range(h2['start_token'], h2['end_token']))) > 0:
                        print(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} {h1} and {h2} overlap!')
                        error = True
                except TypeError:
                    print(h1, h2)
                    raise TypeError
            if h1['start_token'] > doc_length:
                print(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} received an impossible tokenspan: {h1} for a document of length {doc_length}')
                error = True
            if h1['end_token'] > doc_length:
                print(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} received an impossible tokenspan: {h1} for a document of length {doc_length}')
                error = True
        # length check for soft rationale
        # note that either flattened_documents or sentence-broken documents must be passed in depending on result
        soft_rationale_predictions = rat.get('soft_rationale_predictions', [])
        if len(soft_rationale_predictions) > 0 and len(soft_rationale_predictions) != doc_length:
            print(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} expected classifications for {doc_length} tokens but have them for {len(soft_rationale_predictions)} tokens instead!')
            error = True

    # count that one appears per-document
    docids = Counter(docids)
    for docid, count in docids.items():
        if count > 1:
            error = True
            print('Error! For instance annotation={instance["annotation_id"]}, docid={docid} appear {count} times, may only appear once!')

    classification = instance.get('classification', '')
    if not isinstance(classification, str):
        print(f'Error! For instance annotation={instance["annotation_id"]}, classification field {classification} is not a string!')
        error = True
    classification_scores = instance.get('classification_scores', dict())
    if not isinstance(classification_scores, dict):
        print(f'Error! For instance annotation={instance["annotation_id"]}, classification_scores field {classification_scores} is not a dict!')
        error = True
    comprehensiveness_classification_scores = instance.get('comprehensiveness_classification_scores', dict())
    if not isinstance(comprehensiveness_classification_scores, dict):
        print(f'Error! For instance annotation={instance["annotation_id"]}, comprehensiveness_classification_scores field {comprehensiveness_classification_scores} is not a dict!')
        error = True
    sufficiency_classification_scores = instance.get('sufficiency_classification_scores', dict())
    if not isinstance(sufficiency_classification_scores, dict):
        print(f'Error! For instance annotation={instance["annotation_id"]}, sufficiency_classification_scores field {sufficiency_classification_scores} is not a dict!')
        error = True
    if ('classification' in instance) != ('classification_scores' in instance):
        print(f'Error! For instance annotation={instance["annotation_id"]}, when providing a classification, you must also provide classification scores!')
        error = True
    if ('comprehensiveness_classification_scores' in instance) and not ('classification' in instance):
        print(f'Error! For instance annotation={instance["annotation_id"]}, when providing a classification, you must also provide a comprehensiveness_classification_score')
        error = True
    if ('sufficiency_classification_scores' in instance) and not ('classification_scores' in instance):
        print(f'Error! For instance annotation={instance["annotation_id"]}, when providing a sufficiency_classification_score, you must also provide a classification score!')
        error = True
    return error

def verify_instances(instances: List[dict], docs: Dict[str, list]):
    annotation_ids = list(x['annotation_id'] for x in instances)
    key_counter = Counter(annotation_ids)
    multi_occurrence_annotation_ids = list(filter(lambda kv: kv[1] > 1, key_counter.items()))
    error = False
    if len(multi_occurrence_annotation_ids) > 0:
        error = True
        print(f'Error in instances: {len(multi_occurrence_annotation_ids)} appear multiple times in the annotations file: {multi_occurrence_annotation_ids}')
    failed_validation = set()
    instances_with_classification = list()
    instances_with_soft_rationale_predictions = list()
    instances_with_soft_sentence_predictions = list()
    instances_with_comprehensiveness_classifications = list()
    instances_with_sufficiency_classifications = list()
    for instance in instances:
        instance_error = verify_instance(instance, docs)
        if instance_error:
            error = True
            failed_validation.add(instance['annotation_id'])
        if instance.get('classification', None) != None:
            instances_with_classification.append(instance)
        if instance.get('comprehensiveness_classification_scores', None) != None:
            instances_with_comprehensiveness_classifications.append(instance)
        if instance.get('sufficiency_classification_scores', None) != None:
            instances_with_sufficiency_classifications.append(instance)
        has_soft_rationales = []
        has_soft_sentences = []
        for rat in instance['rationales']:
            if rat.get('soft_rationale_predictions', None) != None:
                has_soft_rationales.append(rat)
            if rat.get('soft_sentence_predictions', None) != None:
                has_soft_sentences.append(rat)
        if len(has_soft_rationales) > 0:
            instances_with_soft_rationale_predictions.append(instance)
            if len(has_soft_rationales) != len(instance['rationales']):
                error = True
                print(f'Error: instance {instance["annotation"]} has soft rationales for some but not all reported documents!')
        if len(has_soft_sentences) > 0:
            instances_with_soft_sentence_predictions.append(instance)
            if len(has_soft_sentences) != len(instance['rationales']):
                error = True
                print(f'Error: instance {instance["annotation"]} has soft sentences for some but not all reported documents!')
    print(f'Error in instances: {len(failed_validation)} instances fail validation: {failed_validation}')
    if len(instances_with_classification) != 0 and len(instances_with_classification) != len(instances):
        print(f'Either all {len(instances)} must have a classification or none may, instead {len(instances_with_classification)} do!')
        error = True
    if len(instances_with_soft_sentence_predictions) != 0 and len(instances_with_soft_sentence_predictions) != len(instances):
        print(f'Either all {len(instances)} must have a sentence prediction or none may, instead {len(instances_with_soft_sentence_predictions)} do!')
        error = True
    if len(instances_with_soft_rationale_predictions) != 0 and len(instances_with_soft_rationale_predictions) != len(instances):
        print(f'Either all {len(instances)} must have a soft rationale prediction or none may, instead {len(instances_with_soft_rationale_predictions)} do!')
        error = True
    if len(instances_with_comprehensiveness_classifications) != 0 and len(instances_with_comprehensiveness_classifications) != len(instances):
        print(f'Either all {len(instances)} must have a comprehensiveness classification or none may, instead {len(instances_with_comprehensiveness_classifications)} do!')
    if len(instances_with_sufficiency_classifications) != 0 and len(instances_with_sufficiency_classifications) != len(instances):
        print(f'Either all {len(instances)} must have a sufficiency classification or none may, instead {len(instances_with_sufficiency_classifications)} do!')
    if error:
        raise ValueError('Some instances are invalid, please fix your formatting and try again')

def _has_hard_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'hard_rationale_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['hard_rationale_predictions'] is not None

def _has_soft_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'soft_rationale_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['soft_rationale_predictions'] is not None

def _has_soft_sentence_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'soft_sentence_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['soft_sentence_predictions'] is not None

def _has_classifications(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'classification' in results[0] and results[0]['classification'] is not None
                             
                             
                             
                             
                             
                             


def evaluate(model_name):
    annotation_fname = data_dir + split_name + '.jsonl'
    results_fname = 'eraserbenchmark/annotated_by_exp/{}.jsonl'.format(model_name+'_'+split_name)
    score_file = 'eraserbenchmark/outputs/{}.txt'.format(model_name+'_'+split_name)
    results = None
    with open(results_fname+'pkl3', 'rb') as fin:
        results = pickle.load(fin)

    for i in range(len(results)):
        for j in range(len(results[i]['rationales'])):
            results[i]['rationales'][j]['docid'] = results[i]['rationales'][j]['docids'][0]
            results[i]['rationales'][j].pop('docids', None)
        for s in ['classification_scores', 'comprehensiveness_classification_scores', 'sufficiency_classification_scores']:
            results[i][s]['True'] = results[i][s]['POS']
            results[i][s]['False'] = results[i][s]['NEG']
    #print(results[0])
    if dataset_name == 'fever':
        docids = set(chain.from_iterable(
            [rat['docid'] for rat in res['rationales']] for res in results))
    elif dataset_name == 'movie':
        docids = set(chain.from_iterable(
            [rat['docid'] for rat in res['rationales']] for res in results))
    docs = load_flattened_documents(data_dir, docids)
    verify_instances(results, docs)
    # load truth
    annotations = annotations_from_jsonl(annotation_fname)
    docids |= set(
        chain.from_iterable(
            (ev.docid for ev in chain.from_iterable(ann.evidences))
            for ann in annotations
        )
    )
    has_final_predictions = _has_classifications(results)
    scores = dict()
    if strict:
        if not iou_thresholds:
            raise ValueError(
                "iou_thresholds must be provided when running strict scoring")
        if not has_final_predictions:
            raise ValueError(
                "We must have a 'classification', 'classification_score', and 'comprehensiveness_classification_score' field in order to perform scoring!")

    if _has_hard_predictions(results):
        truth = list(chain.from_iterable(Rationale.from_annotation(ann)
                                         for ann in annotations))
        pred = list(chain.from_iterable(Rationale.from_instance(inst)
                                        for inst in results))
        #print(truth, pred)
        if iou_thresholds is not None:
            iou_scores = partial_match_score(truth, pred, iou_thresholds)
            scores['iou_scores'] = iou_scores
        # NER style scoring
        rationale_level_prf = score_hard_rationale_predictions(truth, pred)
        scores['rationale_prf'] = rationale_level_prf
        token_level_truth = list(chain.from_iterable(
            rat.to_token_level() for rat in truth))
        token_level_pred = list(chain.from_iterable(
            rat.to_token_level() for rat in pred))
        token_level_prf = score_hard_rationale_predictions(
            token_level_truth, token_level_pred)
        scores['token_prf'] = token_level_prf
    else:
        print(
            "No hard predictions detected, skipping rationale scoring")

    if _has_soft_predictions(results):
        flattened_documents = load_flattened_documents(data_dir, docids)
        paired_scoring = PositionScoredDocument.from_results(
            results, annotations, flattened_documents, use_tokens=True)
        token_scores = score_soft_tokens(paired_scoring)
        scores['token_soft_metrics'] = token_scores
    else:
        print(
            "No soft predictions detected, skipping rationale scoring")

    if _has_soft_sentence_predictions(results):
        documents = load_documents(data_dir, docids)
        paired_scoring = PositionScoredDocument.from_results(
            results, annotations, documents, use_tokens=False)
        sentence_scores = score_soft_tokens(paired_scoring)
        scores['sentence_soft_metrics'] = sentence_scores
    else:
        print(
            "No sentence level predictions detected, skipping sentence-level diagnostic")

    if has_final_predictions:
        flattened_documents = load_flattened_documents(data_dir, docids)
        class_results = score_classifications(
            results, annotations, flattened_documents)
        scores['classification_scores'] = class_results
    else:
        print(
            "No classification scores detected, skipping classification")

    pprint.pprint(scores)

    if score_file:
        with open(score_file, 'w') as of:
            json.dump(str(scores), of, indent=4, sort_keys=True)
