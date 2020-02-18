def extract_doc_ids_from_annotations(anns):
    ret = set()
    for ann in anns:
        for ev_group in ann.evidences:
            for ev in ev_group:
                ret.add(ev.docid)
    return ret
