def extract_doc_ids_from_annotations(anns):
    ret = set()
    for ann in anns:
        if len(ann.evidences) == 0: # posR_161 in movie reviews has no evidence
            ret.add(ann.annotation_id)
            continue
        for ev_group in ann.evidences:
            for ev in ev_group:
                ret.add(ev.docid)
    return ret
