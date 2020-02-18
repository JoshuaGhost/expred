#
import json
import logging
import pickle
import pprint

from eraserbenchmark.rationale_benchmark.utils import (
    annotations_from_jsonl,
    load_documents,
    load_flattened_documents
)
from eraserbenchmark.scoring import *
from eraserbenchmark.verify_instances import _has_classifications, \
    _has_hard_predictions, \
    _has_soft_predictions, \
    _has_soft_sentence_predictions, \
    verify_instances

logging.basicConfig(level=logging.DEBUG,
                    format='%(relativeCreated)6d %(threadName)s %(message)s')

dataset_name = 'movies'

# split_name = 'test'
# split_name = 'val'

strict = True
strict = False

iou_thresholds = [0.5]


def evaluate(model_name, dataset=dataset_name, split_name='test', train_on_portion=0, data_dir='.'):
    annotation_fname = data_dir + split_name + '.jsonl'
    results_fname = 'eraserbenchmark/annotated_by_exp/{}.jsonl'.format(model_name + '_' + split_name)
    score_file = 'eraserbenchmark/outputs/{}.txt'.format(model_name + '_' + split_name)
    with open(results_fname + 'pkl3', 'rb') as fin:
        results = pickle.load(fin)
    docids = set(chain.from_iterable([rat['docid'] for rat in res['rationales']] for res in results))
    docs = load_flattened_documents(data_dir, docids)
    verify_instances(results, docs)
    # load truth
    annotations = annotations_from_jsonl(annotation_fname)
    if train_on_portion != 0 and split_name == 'train':
        annotations = annotations[:int(len(annotations) * train_on_portion)]
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
        # print(truth, pred)
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

    if 'classification_scores' in scores:
        if 'comprehensiveness' in scores['classification_scores'] and 'sufficiency' in scores['classification_scores']:
            scores['classification_scores']['cs_f1'] = 2 / (
                        1 / (scores['classification_scores']['comprehensiveness'] + 0.000000001) + 1 / (
                            scores['classification_scores']['sufficiency'] + 0.000000001))

    pprint.pprint(scores)

    if score_file:
        with open(score_file, 'w') as of:
            json.dump(str(scores), of, indent=4, sort_keys=True)
