from itertools import chain

from expred.models.pipeline.pipeline_utils import SentenceEvidence

def get_docids(ann):
    ret = []
    for ev_group in ann.evidences:
        for ev in ev_group:
            ret.append(ev.docid)
    return ret

def extract_doc_ids_from_annotations(anns):
    ret = set()
    for ann in anns:
        ret |= set(get_docids(ann))
    return ret

def chain_sentence_evidences(sentences):
    kls = list(chain.from_iterable(s.kls for s in sentences))
    document = list(chain.from_iterable(s.sentence for s in sentences))
    assert len(kls) == len(document)
    return SentenceEvidence(kls=kls,
                            ann_id=sentences[0].ann_id,
                            sentence=document,
                            docid=sentences[0].docid,
                            index=sentences[0].index,
                            query=sentences[0].query,
                            has_evidence=any(map(lambda s: s.has_evidence, sentences)))