from typing import List
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple
from itertools import chain

from eraserbenchmark.rationale_benchmark.utils import (
    Annotation,
    Evidence,
    annotations_from_jsonl,
    load_jsonl,
    load_documents,
    load_flattened_documents
 )


@dataclass(eq=True, frozen=True)
class PositionScoredDocument:
    ann_id: str
    docid: str
    scores: Tuple[float]
    truths: Tuple[bool]

    @classmethod
    def from_results(cls, instances: List[dict], annotations: List[Annotation], docs: Dict[str, List[Any]], use_tokens: bool=True) -> List['PositionScoredDocument']:
        """Creates a paired list of annotation ids/docids/predictions/truth values"""
        key_to_annotation = dict()
        for ann in annotations:
            for ev in chain.from_iterable(ann.evidences):
                key = (ann.annotation_id, ev.docid)
                if key not in key_to_annotation:
                    key_to_annotation[key] = [False for _ in docs[ev.docid]]
                if use_tokens:
                    start, end = ev.start_token, ev.end_token
                else:
                    start, end = ev.start_sentence, ev.end_sentence
                for t in range(start, end):
                    key_to_annotation[key][t] = True
        ret = []
        if use_tokens:
            field = 'soft_rationale_predictions'
        else:
            field = 'soft_sentence_predictions'
        for inst in instances:
            for rat in inst['rationales']:
                docid = rat['docid']
                scores = rat[field]
                key = (inst['annotation_id'], docid)
                assert len(scores) == len(docs[docid])
                if key in key_to_annotation :
                    assert len(scores) == len(key_to_annotation[key])
                else :
                    #In case model makes a prediction on docuemnt(s) for which ground truth evidence is not present
                    key_to_annotation[key] = [False for _ in docs[docid]]
                ret.append(PositionScoredDocument(inst['annotation_id'], docid, tuple(scores), tuple(key_to_annotation[key])))
        return ret
