from dataclasses import dataclass
from typing import List

from eraserbenchmark.rationale_benchmark.utils import (
    Annotation,
    Evidence,
    annotations_from_jsonl,
    load_jsonl,
    load_documents,
    load_flattened_documents
 )


# start_token is inclusive, end_token is exclusive
@dataclass(eq=True, frozen=True)
class Rationale:
    ann_id: str
    docid: str
    start_token: int
    end_token: int

    def to_token_level(self) -> List['Rationale']:
        ret = []
        for t in range(self.start_token, self.end_token):
            ret.append(Rationale(self.ann_id, self.docid, t, t+1))
        return ret

    @classmethod
    def from_annotation(cls, ann: Annotation) -> List['Rationale']:
        ret = []
        for ev_group in ann.evidences:
            for ev in ev_group:
                ret.append(Rationale(ann.annotation_id, ev.docid, ev.start_token, ev.end_token))
        return ret

    @classmethod
    def from_instance(cls, inst: dict) -> List['Rationale']:
        ret = []
        for rat in inst['rationales']:
            for pred in rat.get('hard_rationale_predictions', []):
                ret.append(Rationale(inst['annotation_id'], rat['docid'], pred['start_token'], pred['end_token']))
        return ret