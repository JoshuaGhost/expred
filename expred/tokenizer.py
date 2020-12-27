from itertools import chain
from typing import List, Dict, Tuple

import os
import torch
from transformers import BertTokenizer, logger

# from expred.models.pipeline.bert_pipeline import bert_intern_doc
from expred.utils import Evidence, Annotation


class BertTokenizerWithMapping(BertTokenizer):
    def __init__(self, *args, **kwargs):
        super(BertTokenizerWithMapping, self).__init__(*args, **kwargs)

    def tokenize_doc(self, doc: List[List[str]],
                     special_token_map: Dict[str, int]) -> \
            Tuple[List[List[str]], List[List[Tuple[int, int]]]]:
        """ Tokenizes a document and returns [start, end) spans to map the wordpieces back to their source words"""
        sents = []
        sent_token_spans = []
        for sent in doc:
            tokens = []
            spans = []
            start = 0
            for w in sent:
                if w in special_token_map:
                    tokens.append(w)
                else:
                    tokens.extend(super(BertTokenizerWithMapping, self).tokenize(w))
                end = len(tokens)
                spans.append((start, end))
                start = end
            sents.append(tokens)
            sent_token_spans.append(spans)
        return sents, sent_token_spans

    def encode_doc(self,
                   doc: List[List[str]],
                   special_token_map) -> List[List[int]]:
        # return [list(chain.from_iterable(special_token_map.get(w, tokenizer.encode(w, add_special_tokens=False))
        # for w in s)) for s in doc]
        return [[special_token_map.get(w, self.convert_tokens_to_ids(w))
                 for w in s]
                for s in doc]

    def _encode_docs_maybe_load_from_cache(self, documents, cache_fname):
        if os.path.exists(cache_fname):
            logger.info(f'Loading interned documents from {cache_fname}')
            (encoded_docs, encoded_doc_token_slides) = torch.load(cache_fname)
        else:
            tokenizer = self
            logger.info(f'Interning documents')
            special_token_map = {
                'SEP': tokenizer.sep_token_id,
                '[SEP]': tokenizer.sep_token_id,
                '[sep]': tokenizer.sep_token_id,
                'UNK': tokenizer.unk_token_id,
                '[UNK]': tokenizer.unk_token_id,
                '[unk]': tokenizer.unk_token_id,
                'PAD': tokenizer.unk_token_id,
                '[PAD]': tokenizer.unk_token_id,
                '[pad]': tokenizer.unk_token_id,
            }
            encoded_docs = {}
            encoded_doc_token_slides = {}
            for d, doc in documents.items():
                tokenized_doc, w_slices = self.tokenize_doc(doc, special_token_map=special_token_map)
                encoded_docs[d] = self.encode_doc(tokenized_doc, special_token_map=special_token_map)
                encoded_doc_token_slides[d] = w_slices
            torch.save((encoded_docs, encoded_doc_token_slides), cache_fname)
        return encoded_docs, encoded_doc_token_slides

    def encode_docs(self, documents, cache_dir):
        cache_fname = os.path.join(cache_dir, 'preprocessed.pkl')
        encoded_docs, encoded_doc_token_slides = self._encode_docs_maybe_load_from_cache(documents, cache_fname)
        return encoded_docs, encoded_doc_token_slides

    def encode_annotations(self, annotations):
        ret = []
        for ann in annotations:
            ev_groups = []
            for ev_group in ann.evidences:
                evs = []
                for ev in ev_group:
                    text = list(chain.from_iterable(self.tokenize(w)
                                                    for w in ev.text.split()))
                    if len(text) == 0:
                        continue
                    text = self.encode(text, add_special_tokens=False)
                    evs.append(Evidence(text=tuple(text),
                                        docid=ev.docid,
                                        start_token=ev.start_token,
                                        end_token=ev.end_token,
                                        start_sentence=ev.start_sentence,
                                        end_sentence=ev.end_sentence))
                ev_groups.append(tuple(evs))
            query = list(chain.from_iterable(self.tokenize(w)
                                             for w in ann.query.split()))
            if len(query) > 0:
                query = self.encode(query, add_special_tokens=False)
            else:
                query = []
            ret.append(Annotation(annotation_id=ann.annotation_id,
                                  query=tuple(query),
                                  evidences=frozenset(ev_groups),
                                  classification=ann.classification,
                                  query_type=ann.query_type))
        return ret
