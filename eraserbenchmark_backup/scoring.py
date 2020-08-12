from collections import defaultdict
from itertools import chain
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from scipy.stats import entropy
from sklearn.metrics import accuracy_score, auc, average_precision_score, \
    classification_report, precision_recall_curve, \
    roc_auc_score

from eraserbenchmark.position_scored_document import PositionScoredDocument
from eraserbenchmark.rationale import Rationale
from eraserbenchmark.rationale_benchmark.utils import (
    Annotation
)


def _f1(_p, _r):
    if _p == 0 or _r == 0:
        return 0
    return 2 * _p * _r / (_p + _r)


def _keyed_rationale_from_list(rats: List[Rationale]) -> Dict[Tuple[str, str], Rationale]:
    ret = defaultdict(set)
    for r in rats:
        ret[(r.ann_id, r.docid)].add(r)
    return ret


def partial_match_score(truth: List[Rationale], pred: List[Rationale], thresholds: List[float]) -> List[Dict[str, Any]]:
    """Computes a partial match F1

    Computes an instance-level (annotation) micro- and macro-averaged F1 score.
    True Positives are computed by using intersection-over-union and
    thresholding the resulting intersection-over-union fraction.

    Micro-average results are computed by ignoring instance level distinctions
    in the TP calculation (and recall, and precision, and finally the F1 of
    those numbers). Macro-average results are computed first by measuring
    instance (annotation + document) precisions and recalls, averaging those,
    and finally computing an F1 of the resulting average.
    """

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)
    num_classifications = {k: len(v) for k, v in pred_to_rat.items()}
    # print(num_classifications)
    num_truth = {k: len(v) for k, v in ann_to_rat.items()}
    ious = defaultdict(dict)
    for k in set(ann_to_rat.keys()) | set(pred_to_rat.keys()):
        for p in pred_to_rat.get(k, []):
            best_iou = 0.0
            for t in ann_to_rat.get(k, []):
                num = len(set(range(p.start_token, p.end_token)) & set(range(t.start_token, t.end_token)))
                denom = len(set(range(p.start_token, p.end_token)) | set(range(t.start_token, t.end_token)))
                iou = 0 if denom == 0 else num / denom
                if iou > best_iou:
                    best_iou = iou
            ious[k][p] = best_iou
    scores = []
    for threshold in thresholds:
        threshold_tps = dict()
        for k, vs in ious.items():
            threshold_tps[k] = sum(int(x >= threshold) for x in vs.values())
        micro_r = sum(threshold_tps.values()) / sum(num_truth.values())
        micro_p = sum(threshold_tps.values()) / (sum(num_classifications.values()) + np.finfo(np.float).eps)
        micro_f1 = _f1(micro_r, micro_p)
        macro_rs = list(threshold_tps.get(k, 0.0) / n for k, n in num_truth.items())
        macro_ps = list(threshold_tps.get(k, 0.0) / n for k, n in num_classifications.items())
        macro_r = sum(macro_rs) / len(macro_rs)
        macro_p = sum(macro_ps) / (len(macro_ps) + np.finfo(np.float).eps)
        macro_f1 = _f1(macro_r, macro_p)
        scores.append({'threshold': threshold,
                       'micro': {
                           'p': micro_p,
                           'r': micro_r,
                           'f1': micro_f1
                       },
                       'macro': {
                           'p': macro_p,
                           'r': macro_r,
                           'f1': macro_f1
                       },
                       })
    return scores


def score_hard_rationale_predictions(truth: List[Rationale], pred: List[Rationale]) -> Dict[str, Dict[str, float]]:
    """Computes instance (annotation)-level micro/macro averaged F1s"""
    scores = dict()
    truth = set(truth)
    pred = set(pred)
    micro_prec = len(truth & pred) / (len(pred) + np.finfo(np.float).eps)
    micro_rec = len(truth & pred) / len(truth)
    micro_f1 = _f1(micro_prec, micro_rec)

    scores['instance_micro'] = {
        'p': micro_prec,
        'r': micro_rec,
        'f1': micro_f1,
    }

    ann_to_rat = _keyed_rationale_from_list(truth)
    pred_to_rat = _keyed_rationale_from_list(pred)
    instances_to_scores = dict()
    for k in set(ann_to_rat.keys()) | (pred_to_rat.keys()):
        if len(pred_to_rat.get(k, set())) > 0:
            instance_prec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(pred_to_rat[k])
        else:
            instance_prec = 0
        if len(ann_to_rat.get(k, set())) > 0:
            instance_rec = len(ann_to_rat.get(k, set()) & pred_to_rat.get(k, set())) / len(ann_to_rat[k])
        else:
            instance_rec = 0
        instance_f1 = _f1(instance_prec, instance_rec)
        instances_to_scores[k] = {
            'p': instance_prec,
            'r': instance_rec,
            'f1': instance_f1,
        }
    # these are calculated as sklearn would
    macro_prec = sum(instance['p'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_rec = sum(instance['r'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    macro_f1 = sum(instance['f1'] for instance in instances_to_scores.values()) / len(instances_to_scores)
    scores['instance_macro'] = {
        'p': macro_prec,
        'r': macro_rec,
        'f1': macro_f1,
    }
    return scores


def _auprc(truth: Dict[Any, List[bool]], preds: Dict[Any, List[float]]) -> float:
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    aucs = []
    for k, true in truth.items():
        pred = preds[k]
        true = [int(t) for t in true]
        precision, recall, _ = precision_recall_curve(true, pred)
        aucs.append(auc(recall, precision))
    return np.average(aucs)


def _score_aggregator(truth: Dict[Any, List[bool]], preds: Dict[Any, List[float]],
                      score_function: Callable[[List[float], List[float]], float],
                      discard_single_class_answers: bool) -> float:
    if len(preds) == 0:
        return 0.0
    assert len(truth.keys() and preds.keys()) == len(truth.keys())
    scores = []
    for k, true in truth.items():
        pred = preds[k]
        if (all(true) or all(not x for x in true)) and discard_single_class_answers:
            continue
        true = [int(t) for t in true]
        scores.append(score_function(true, pred))
    return np.average(scores)


def score_soft_tokens(paired_scores: List[PositionScoredDocument]) -> Dict[str, float]:
    truth = {(ps.ann_id, ps.docid): ps.truths for ps in paired_scores}
    pred = {(ps.ann_id, ps.docid): ps.scores for ps in paired_scores}
    auprc_score = _auprc(truth, pred)
    ap = _score_aggregator(truth, pred, average_precision_score, True)
    roc_auc = _score_aggregator(truth, pred, roc_auc_score, True)

    return {
        'auprc': auprc_score,
        'average_precision': ap,
        'roc_auc_score': roc_auc,
    }


def score_classifications(instances: List[dict], annotations: List[Annotation], docs: Dict[str, List[str]]) -> Dict[
    str, float]:
    def compute_kl(cls_scores_, faith_scores_):
        keys = list(cls_scores_.keys())
        cls_scores_ = [cls_scores_[k] for k in keys]
        faith_scores_ = [faith_scores_[k] for k in keys]
        return entropy(faith_scores_, cls_scores_)

    labels = list(set(x.classification for x in annotations))
    label_to_int = {l: i for i, l in enumerate(labels)}
    key_to_instances = {inst['annotation_id']: inst for inst in instances}
    truth = []
    predicted = []
    for ann in annotations:
        truth.append(label_to_int[ann.classification])
        inst = key_to_instances[ann.annotation_id]
        predicted.append(label_to_int[inst['classification']])
    classification_scores = classification_report(truth, predicted, output_dict=True, target_names=labels, digits=3)
    accuracy = accuracy_score(truth, predicted)
    if 'comprehensiveness_classification_scores' in instances[0]:
        comprehensiveness_scores = [
            x['classification_scores'][x['classification']] - x['comprehensiveness_classification_scores'][
                x['classification']] for x in instances]
        comprehensiveness_score = np.average(comprehensiveness_scores)
    else:
        comprehensiveness_score = None
        comprehensiveness_scores = None

    if 'sufficiency_classification_scores' in instances[0]:
        sufficiency_scores = [x['classification_scores'][x['classification']] - x['sufficiency_classification_scores'][
            x['classification']] for x in instances]
        sufficiency_score = np.average(sufficiency_scores)
    else:
        sufficiency_score = None
        sufficiency_scores = None

    if 'comprehensiveness_classification_scores' in instances[0]:
        comprehensiveness_entropies = [entropy(list(x['classification_scores'].values())) - entropy(
            list(x['comprehensiveness_classification_scores'].values())) for x in instances]
        comprehensiveness_entropy = np.average(comprehensiveness_entropies)
        comprehensiveness_kl = np.average(list(
            compute_kl(x['classification_scores'], x['comprehensiveness_classification_scores']) for x in instances))
    else:
        comprehensiveness_entropies = None
        comprehensiveness_kl = None
        comprehensiveness_entropy = None

    if 'sufficiency_classification_scores' in instances[0]:
        sufficiency_entropies = [entropy(list(x['classification_scores'].values())) - entropy(
            list(x['sufficiency_classification_scores'].values())) for x in instances]
        sufficiency_entropy = np.average(sufficiency_entropies)
        sufficiency_kl = np.average(
            list(compute_kl(x['classification_scores'], x['sufficiency_classification_scores']) for x in instances))
    else:
        sufficiency_entropies = None
        sufficiency_kl = None
        sufficiency_entropy = None

    if 'tokens_to_flip' in instances[0]:
        token_percentages = []
        for ann in annotations:
            # in practice, this is of size 1 for everything except e-snli
            docids = set(ev.docid for ev in chain.from_iterable(ann.evidences))
            inst = key_to_instances[ann.annotation_id]
            tokens = inst['tokens_to_flip']
            doc_lengths = sum(len(docs[d]) for d in docids)
            token_percentages.append(tokens / doc_lengths)
        token_percentages = np.average(token_percentages)
    else:
        token_percentages = None

    return {
        'accuracy': accuracy,
        'prf': classification_scores,
        'comprehensiveness': comprehensiveness_score,
        'sufficiency': sufficiency_score,
        'comprehensiveness_entropy': comprehensiveness_entropy,
        'comprehensiveness_kl': comprehensiveness_kl,
        'sufficiency_entropy': sufficiency_entropy,
        'sufficiency_kl': sufficiency_kl,
    }
