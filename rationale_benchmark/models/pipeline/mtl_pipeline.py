import argparse
from itertools import chain
from typing import List, Tuple

from transformers import BertTokenizer, BertConfig
import logging
import torch
import os
import json
from rationale_benchmark.models.mlp_mtl import BertMTL, BertClassifier
from params import MTLParams

from rationale_benchmark.models.pipeline.bert_pipeline import bert_intern_doc, bert_intern_annotation
from rationale_benchmark.models.pipeline.mtl_pipeline_utils import decode
from rationale_benchmark.utils import load_datasets, load_documents, write_jsonl
from rationale_benchmark.models.pipeline.mtl_token_identifier import train_mtl_token_identifier
from rationale_benchmark.models.pipeline.mtl_evidence_classifier import train_mtl_evidence_classifier


def initialize_models(params: dict, batch_first: bool, unk_token='<unk>'):
    assert batch_first
    max_length = params['max_length']
    tokenizer = BertTokenizer.from_pretrained(params['bert_vocab'])
    labels = dict((y, x) for (x, y) in enumerate(params['classes']))
    mtl_params = MTLParams
    mtl_params.num_labels = len(labels)
    mtl_params.dim_exp_gru = params['dim_exp_gru']
    mtl_params.dim_cls_linear = params['dim_cls_linear']
    bert_dir = params['bert_dir']
    use_half_precision = bool(params['mtl_token_identifier'].get('use_half_precision', 1))
    evidence_identifier = BertMTL(bert_dir=bert_dir,
                                  tokenizer=tokenizer,
                                  mtl_params=mtl_params,
                                  max_length=max_length,
                                  use_half_precision=use_half_precision)

    use_half_precision = bool(params['evidence_classifier'].get('use_half_precision', 1))
    evidence_classifier = BertClassifier(bert_dir=bert_dir,
                                         pad_token_id=tokenizer.pad_token_id,
                                         cls_token_id=tokenizer.cls_token_id,
                                         sep_token_id=tokenizer.sep_token_id,
                                         num_labels=mtl_params.num_labels,
                                         max_length=max_length,
                                         mtl_params=mtl_params,
                                         use_half_precision=use_half_precision)
    word_interner = tokenizer.vocab
    de_interner = tokenizer.ids_to_tokens
    return evidence_identifier, evidence_classifier, word_interner, de_interner, labels, tokenizer


logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)
# let's make this more or less deterministic (not resistent to restarts)
#random.seed(12345)
#np.random.seed(67890)
#torch.manual_seed(10111213)
#torch.backends.cudnn.deterministic = True
#torch.backends.cudnn.benchmark = False
import random
rand_seed_1 = ord(os.urandom(1))*ord(os.urandom(1))
rand_seed_2 = ord(os.urandom(1))*ord(os.urandom(1))
rand_seed_3 = ord(os.urandom(1))*ord(os.urandom(1))
logger.info(f'seed 1: {rand_seed_1}, seed 2: {rand_seed_2}, seed 3: {rand_seed_3}')
random.seed(rand_seed_1)
import numpy as np
np.random.seed(rand_seed_2)
torch.manual_seed(rand_seed_3)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

def bert_tokenize_doc(doc: List[List[str]], tokenizer, special_token_map) ->\
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
                tokens.extend(tokenizer.tokenize(w))
            end = len(tokens)
            spans.append((start, end))
            start = end
        sents.append(tokens)
        sent_token_spans.append(spans)
    return sents, sent_token_spans


def bert_intern_doc(doc: List[List[str]], tokenizer, special_token_map) -> List[List[int]]:
    #return [list(chain.from_iterable(special_token_map.get(w, tokenizer.encode(w, add_special_tokens=False)) for w in s)) for s in doc]
    return [[special_token_map.get(w, tokenizer.convert_tokens_to_ids(w)) for w in s] for s in doc]


def maybe_load_from_cache(documents, output_dir, tokenizer):
    cache = os.path.join(output_dir, 'preprocessed.pkl')
    if os.path.exists(cache):
        logger.info(f'Loading interned documents from {cache}')
        (interned_documents, interned_document_token_slices) = torch.load(cache)
    else:
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
        interned_documents = {}
        interned_document_token_slices = {}
        for d, doc in documents.items():
            tokenized, w_slices = bert_tokenize_doc(doc, tokenizer, special_token_map=special_token_map)
            interned_documents[d] = bert_intern_doc(tokenized, tokenizer, special_token_map=special_token_map)
            interned_document_token_slices[d] = w_slices
        torch.save((interned_documents, interned_document_token_slices), cache)
    return interned_documents, interned_document_token_slices


def main():
    parser = argparse.ArgumentParser(description=('Trains a pipeline model.\n'
                                                  '\n'
                                                  'Step 1 is evidence identification, the MTL happens here. It '
                                                  'predicts the label of the current sentence and tags its\n '
                                                  '        sub-tokens in the same time \n'
                                                  '    Step 2 is evidence classification, a BERT classifier takes the output of the evidence identifier and predicts its \n'
                                                  '        sentiment. Unlike in Deyong et al. this classifier takes in the same length as the identifier\'s input but with \n'
                                                  '        irrational sub-tokens masked.\n'
                                                  '\n'
                                                  '    These models should be separated into two separate steps, but at the moment:\n'
                                                  '    * prep data (load, intern documents, load json)\n'
                                                  '    * convert data for evidence identification - in the case of training data we take all the positives and sample some negatives\n'
                                                  '        * side note: this sampling is *somewhat* configurable and is done on a per-batch/epoch basis in order to gain a broader sampling of negative values.\n'
                                                  '    * train evidence identification\n'
                                                  '    * convert data for evidence classification - take all rationales + decisions and use this as input\n'
                                                  '    * train evidence classification\n'
                                                  '    * decode first the evidence, then run classification for each split\n'
                                                  '\n'
                                                  '    '), formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--data_dir', dest='data_dir', required=True,
                        help='Which directory contains a {train,val,test}.jsonl file?')
    parser.add_argument('--output_dir', dest='output_dir', required=True,
                        help='Where shall we write intermediate models + final data to?')
    parser.add_argument('--model_params', dest='model_params', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    BATCH_FIRST = True
    assert BATCH_FIRST

    with open(args.model_params, 'r') as fp:
        logger.info(f'Loading model parameters from {args.model_params}')
        model_params = json.load(fp)
        logger.info(f'Params: {json.dumps(model_params, indent=2, sort_keys=True)}')
    train, val, test = load_datasets(args.data_dir)
    docids = set(e.docid for e in
                 chain.from_iterable(chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test)))))
    documents = load_documents(args.data_dir, docids)
    logger.info(f'Loaded {len(documents)} documents')
    # this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important
    unk_token = '<unk>'

    mtl_token_identifier, evidence_classifier, \
    word_interner, de_interner, \
    labels_mapping, \
    tokenizer = initialize_models(model_params,
                                  batch_first=BATCH_FIRST,
                                  unk_token=unk_token)
    logger.info(f'We have {len(word_interner)} wordpieces')
    interned_documents, interned_document_token_slices = maybe_load_from_cache(documents, args.output_dir, tokenizer)
    interned_train = bert_intern_annotation(train, tokenizer)
    interned_val = bert_intern_annotation(val, tokenizer)
    interned_test = bert_intern_annotation(test, tokenizer)
    #for inst in interned_test:
    #    if inst.annotation_id == 'News_CNN_cnn-3b159c09888b61241afe848844510478546353d4.txt:8:7':
    #        print(inst)
    #print(interned_documents['News_CNN_cnn-3b159c09888b61241afe848844510478546353d4.txt'])
    logger.info('Beginning training of the MTL identifier')
    mtl_token_identifier = mtl_token_identifier.cuda()
    mtl_token_identifier, mtl_token_identifier_results,\
    train_machine_annotated, eval_machine_annotated, test_machine_annotated = \
        train_mtl_token_identifier(mtl_token_identifier,
                                   args.output_dir,
                                   interned_train,
                                   interned_val,
                                   interned_test,
                                   labels_mapping=labels_mapping,
                                   interned_documents=interned_documents,
                                   source_documents=documents,
                                   token_mapping=interned_document_token_slices,
                                   model_pars=model_params,
                                   tensorize_model_inputs=True)
    mtl_token_identifier = mtl_token_identifier.cpu()
    # train the evidence identifier

    logger.info('Beginning training of the evidence classifier')
    evidence_classifier = evidence_classifier.cuda()
    optimizer = None
    scheduler = None
    evidence_classifier, evidence_class_results = train_mtl_evidence_classifier(evidence_classifier,
                                                                                args.output_dir,
                                                                                train_machine_annotated,
                                                                                eval_machine_annotated,
                                                                                interned_documents,
                                                                                model_params,
                                                                                optimizer=optimizer,
                                                                                scheduler=scheduler,
                                                                                class_interner=labels_mapping,
                                                                                tensorize_model_inputs=True)

    # decode
    logger.info('Beginning final decoding')
    mtl_token_identifier = mtl_token_identifier.cuda()
    pipeline_batch_size = min(
        [model_params['evidence_classifier']['batch_size'], model_params['mtl_token_identifier']['batch_size']])
    pipeline_results, train_decoded, val_decoded, test_decoded = decode(evidence_identifier=mtl_token_identifier,
                                                                        evidence_classifier=evidence_classifier,
                                                                        train=interned_train, mrs_train=train_machine_annotated,
                                                                        val=interned_val, mrs_eval=eval_machine_annotated,
                                                                        test=interned_test, mrs_test=test_machine_annotated,
                                                                        source_documents=documents,
                                                                        interned_documents=interned_documents,
                                                                        token_mapping=interned_document_token_slices,
                                                                        class_interner=labels_mapping,
                                                                        tensorize_modelinputs=True,
                                                                        vocab=tokenizer.vocab,
                                                                        batch_size=pipeline_batch_size,
                                                                        tokenizer=tokenizer)
    write_jsonl(train_decoded, os.path.join(args.output_dir, 'train_decoded.jsonl'))
    write_jsonl(val_decoded, os.path.join(args.output_dir, 'val_decoded.jsonl'))
    write_jsonl(test_decoded, os.path.join(args.output_dir, 'test_decoded.jsonl'))
    with open(os.path.join(args.output_dir, 'identifier_results.json'), 'w') as ident_output, \
            open(os.path.join(args.output_dir, 'classifier_results.json'), 'w') as class_output:
        ident_output.write(json.dumps(mtl_token_identifier_results))
        class_output.write(json.dumps(evidence_class_results))
    for k, v in pipeline_results.items():
        if type(v) is dict:
            for k1, v1 in v.items():
                logging.info(f'Pipeline results for {k}, {k1}={v1}')
        else:
            logging.info(f'Pipeline results {k}\t={v}')


if __name__ == '__main__':
    main()
