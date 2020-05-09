import os
from eraserbenchmark.rationale_benchmark.utils import load_datasets, load_documents
from eraserbenchmark.eraser_utils import extract_doc_ids_from_annotations
from itertools import chain
from preprocessing import preprocess
from tqdm import tqdm_notebook
from utils import cache_decorator


class Dataset():
    def __init__(self, config):
        self.config = config
        if config.dataset_name == 'movies':
            self.label_list = ['POS', 'NEG']
        elif config.dataset == 'multirc':
            self.label_list = ['True', 'False']
        elif config.dataset == 'fever':
            self.label_list = ['SUPPORTS', 'REFUTES']

    def load_dataset(self):
        self.train, self.val, self.test = load_datasets(self.config.data_dir)
        if self.config.train_on_portion != 0:
            self.train = self.train[:int(len(self.train) * self.config.train_on_portion)]
        self.docids = set(chain.from_iterable(extract_doc_ids_from_annotations(d) for d in [self.train, self.val, self.test]))
        self.docs = load_documents(self.config.data_dir, self.docids)

    def preprocess(self, tokenizer):
        cachedir = 'cache'
        if not os.path.isdir(cachedir):
            os.makedirs(cachedir)

        @cache_decorator(os.path.join(cachedir, self.config.DATASET_CACHE_NAME + '_eraser_format'))
        def preprocess_wrapper():
            ret = []
            for split in 'train val test'.split():
                data = getattr(self, split)
                ret.append(
                    preprocess(data, self.docs, self.label_list,
                               self.config.dataset_name, self.config.MAX_SEQ_LENGTH,
                               self.config.EXP_OUTPUT, self.config.merge_evidences,
                               tokenizer))
            return ret

        self.rets_train, self.rets_val, self.rets_test = preprocess_wrapper()

        self.train_input_ids, self.train_input_masks, self.train_segment_ids, self.train_rations, self.train_labels = self.rets_train
        self.val_input_ids, self.val_input_masks, self.val_segment_ids, self.val_rations, self.val_labels = self.rets_val
        self.test_input_ids, self.test_input_masks, self.test_segment_ids, self.test_rations, self.test_labels = self.rets_test

    @classmethod
    def expand_on_evidences(self, data):
        from eraserbenchmark.rationale_benchmark.utils import Annotation
        expanded_data = []
        for ann in tqdm_notebook(data):
            for ev_group in ann.evidences:
                new_ann = Annotation(annotation_id=ann.annotation_id,
                                     query=ann.query,
                                     evidences=frozenset([ev_group]),
                                     classification=ann.classification)
                expanded_data.append(new_ann)
        return expanded_data
