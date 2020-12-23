from transformers import BertTokenizer
import argparse
from itertools import chain
from typing import List, Tuple

from transformers import BertTokenizer, BertConfig
import logging
import torch
import os
import json
from torch import nn
from rationale_benchmark.models.mlp_mtl import BertMTL, BertClassifier
from params import MTLParams
import random

from rationale_benchmark.models.pipeline.bert_pipeline import bert_intern_doc, bert_intern_annotation
from rationale_benchmark.models.pipeline.mtl_pipeline_utils import decode
from rationale_benchmark.utils import load_datasets, load_documents, write_jsonl
from rationale_benchmark.models.pipeline.mtl_pipeline_utils import make_mtl_token_preds_epoch
from rationale_benchmark.models.pipeline.mtl_token_identifier import train_mtl_token_identifier, _get_sampling_method
from rationale_benchmark.models.pipeline.mtl_evidence_classifier import train_mtl_evidence_classifier
from rationale_benchmark.models.pipeline.mtl_pipeline import initialize_models, maybe_load_from_cache
# load dataset
dataset = 'multirc'
#dataset = 'fever'
#dataset = 'movies'
evidence_identifier = None
evidence_classifier = None
for model_num in range(8,9):
    for dataset in ['multirc', 'fever']:
        %env CUDA_VISIBLE_DEVICES=1
        data_dir = f'/home/zzhang/.keras/datasets/{dataset}'
        output_dir = f'output/expred/{dataset}/{model_num}'
        model_params = f'params/{dataset}_expred.json'
        with open(model_params, 'r') as fp:
            model_params = json.load(fp)
        train, val, test = load_datasets(data_dir)
        docids = set(e.docid for e in
                         chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
        documents = load_documents(data_dir, docids)

        # initialize models
        unk_token = '<unk>'
        BATCH_FIRST = True
        mtl_token_identifier, evidence_classifier, \
        word_interner, de_interner, \
        labels_mapping, \
        tokenizer = initialize_models(model_params, batch_first=BATCH_FIRST, unk_token=unk_token)
        interned_documents, token_mapping = maybe_load_from_cache(documents, output_dir, tokenizer)
        #interned_train = bert_intern_annotation(train, tokenizer)
        #interned_val = bert_intern_annotation(val, tokenizer)
        interned_test = bert_intern_annotation(test, tokenizer)
        from rationale_benchmark.models.pipeline.mtl_pipeline_utils import annotations_to_mtl_token_identification
        evidence_test_data = annotations_to_mtl_token_identification(interned_test,
                                                                     source_documents=documents,
                                                                     interned_documents=interned_documents,
                                                                     token_mapping=token_mapping)

        # load identifier
        evidence_identifier_output_dir = os.path.join(output_dir, 'evidence_token_identifier')
        model_save_file = os.path.join(evidence_identifier_output_dir, 'evidence_token_identifier.pt')
        cls_criterion = nn.BCELoss(reduction='none')
        from rationale_benchmark.models.losses import resampling_rebalanced_crossentropy
        exp_criterion = resampling_rebalanced_crossentropy(seq_reduction='none')
        sampling_method = _get_sampling_method(model_params['mtl_token_identifier'])
        batch_size = 80#model_params['mtl_token_identifier']['batch_size']
        max_length = model_params['max_length']
        par_lambda = model_params['mtl_token_identifier']['par_lambda']
        mtl_token_identifier = mtl_token_identifier.cuda()
        device = next(mtl_token_identifier.parameters()).device
        mtl_token_identifier.load_state_dict(torch.load(model_save_file))
        print("loading completed")

        # evaluate identifier
        def _prep_data_for_epoch(evidence_data, sampler):
            output_annotations = []
            ann_ids = sorted(evidence_data.keys())
            # in place shuffle so we get a different per-epoch ordering
            random.shuffle(ann_ids)
            for ann_id in ann_ids:
                for docid, sentences in evidence_data[ann_id][1].items():
                    data = sampler(sentences, None)
                    output_annotations.append((evidence_data[ann_id][0], data))
            return output_annotations

        import numpy as np
        with torch.no_grad():
            mtl_token_identifier = mtl_token_identifier.to(device=device)
            mtl_token_identifier.eval()
            epoch_input_data = _prep_data_for_epoch(evidence_test_data, sampling_method)
            _, _, _, soft_pred_for_cl, hard_pred_for_cl, _, \
            pred_labels_stage_1, labels_stage_1 = \
                make_mtl_token_preds_epoch(mtl_token_identifier, epoch_input_data, labels_mapping,
                                           token_mapping, batch_size, max_length, par_lambda,
                                           device, cls_criterion, exp_criterion, tensorize_model_inputs=True)
            from sklearn.metrics import classification_report
            pred_labels_stage_1 = [np.argmax(p) for p in pred_labels_stage_1]
            labels_stage_1 = [np.argmax(l) for l in labels_stage_1]
            identifier_performance = classification_report(labels_stage_1, pred_labels_stage_1, output_dict=True)

            phase_1_performance_fname = os.path.join(output_dir, 'phase_1_performance.txt')
            with open(phase_1_performance_fname, 'w+') as fout:    
                fout.write(str(identifier_performance))

        # convert dataset
        with torch.no_grad():
            mtl_token_identifier.eval()
            def remove_query_preds(epoch_input_data, soft_pred_for_cl, hard_pred_for_cl):
                hard_pred_for_cl = [h.cpu().tolist() for h in hard_pred_for_cl]
                hard_pred_for_cl = [h[len(d[1].query)+2:] for h, d in zip(hard_pred_for_cl, epoch_input_data)]
                soft_pred_for_cl = [s[len(d[1].query)+2:] for s, d in zip(soft_pred_for_cl, epoch_input_data)]
                return list(zip(epoch_input_data, soft_pred_for_cl, hard_pred_for_cl))
            test_machine_annotated = remove_query_preds(epoch_input_data, soft_pred_for_cl, hard_pred_for_cl)
            def fix_missing_hard_evidence(h_pred, method='doc'):
                if method not in 'ignore,doc,rand'.split(','):
                    assert False
                if sum(h_pred) != 0 or method == 'ignore':
                    return h_pred
                if method == 'doc':
                    return np.ones_like(h_pred)
                if method == 'rand':
                    return np.array([np.random.randint(2) for i in h_pred])

            hard_pred_for_cl = [fix_missing_hard_evidence(h_pred, method='ignore') for h_pred in hard_pred_for_cl]

        # load classifier
        evidence_classifier_output_dir = os.path.join(output_dir, 'evidence_classifier')
        model_save_file = os.path.join(evidence_classifier_output_dir, 'evidence_classifier.pt')
        evidence_classifier = evidence_classifier.cuda()
        evidence_classifier.load_state_dict(torch.load(model_save_file))

        # evaluate classifier
        pipeline_batch_size = 80#min([model_params['evidence_classifier']['batch_size'], model_params['mtl_token_identifier']['batch_size']])
        pipeline_results, train_decoded, val_decoded, test_decoded = decode(evidence_identifier=mtl_token_identifier,
                                                                            evidence_classifier=evidence_classifier,
                                                                            train=None, mrs_train=None,
                                                                            val=None, mrs_eval=None,
                                                                            test=interned_test, mrs_test=test_machine_annotated,
                                                                            source_documents=documents,
                                                                            interned_documents=interned_documents,
                                                                            token_mapping=token_mapping,
                                                                            class_interner=labels_mapping,
                                                                            tensorize_modelinputs=True,
                                                                            vocab=tokenizer.vocab,
                                                                            batch_size=pipeline_batch_size,
                                                                            tokenizer=tokenizer)
        write_jsonl(test_decoded, os.path.join(output_dir, 'test_decoded_ignore_empty.jsonl'))
        del mtl_token_identifier
        del evidence_classifier