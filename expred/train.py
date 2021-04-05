import argparse
import logging
import wandb as wandb
from typing import List, Dict, Set, Tuple

import torch
import os
import json
import numpy as np
import random

from itertools import chain

from expred.params import MTLParams
from expred.models.mlp_mtl import BertMTL, BertClassifier
from expred.tokenizer import BertTokenizerWithMapping
from expred.models.pipeline.mtl_pipeline_utils import decode
from expred.utils import load_datasets, load_documents, write_jsonl
from expred.models.pipeline.mtl_token_identifier import train_mtl_token_identifier
from expred.models.pipeline.mtl_evidence_classifier import train_mtl_evidence_classifier

BATCH_FIRST = True


def initialize_models(conf: dict,
                      tokenizer: BertTokenizerWithMapping,
                      batch_first: bool) -> Tuple[BertMTL, BertClassifier, Dict[int, str]]:
    """
    Does several things:
    1. Create a mapping from label names to ids
    2. Configure and create the multi task learner, the first stage of the model (BertMTL)
    3. Configure and create the evidence classifier, second stage of the model (BertClassifier)
    :param conf:
    :param tokenizer:
    :param batch_first:
    :return: BertMTL, BertClassifier, label mapping
    """
    assert batch_first
    max_length = conf['max_length']
    # label mapping
    labels = dict((y, x) for (x, y) in enumerate(conf['classes']))

    # configure multi task learner
    mtl_params = MTLParams
    mtl_params.num_labels = len(labels)
    mtl_params.dim_exp_gru = conf['dim_exp_gru']
    mtl_params.dim_cls_linear = conf['dim_cls_linear']
    bert_dir = conf['bert_dir']
    use_half_precision = bool(conf['mtl_token_identifier'].get('use_half_precision', 1))
    evidence_identifier = BertMTL(bert_dir=bert_dir,
                                  tokenizer=tokenizer,
                                  mtl_params=mtl_params,
                                  max_length=max_length,
                                  use_half_precision=use_half_precision)

    # set up the evidence classifier
    use_half_precision = bool(conf['evidence_classifier'].get('use_half_precision', 1))
    evidence_classifier = BertClassifier(bert_dir=bert_dir,
                                         pad_token_id=tokenizer.pad_token_id,
                                         cls_token_id=tokenizer.cls_token_id,
                                         sep_token_id=tokenizer.sep_token_id,
                                         num_labels=mtl_params.num_labels,
                                         max_length=max_length,
                                         mtl_params=mtl_params,
                                         use_half_precision=use_half_precision)

    return evidence_identifier, evidence_classifier, labels


logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s')
logger = logging.getLogger(__name__)

# let's make this more or less deterministic (not resistent to restarts)
random.seed(12345)
np.random.seed(67890)
torch.manual_seed(10111213)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# or, uncomment the following sentences to make it more than random
# rand_seed_1 = ord(os.urandom(1)) * ord(os.urandom(1))
# rand_seed_2 = ord(os.urandom(1)) * ord(os.urandom(1))
# rand_seed_3 = ord(os.urandom(1)) * ord(os.urandom(1))
# random.seed(rand_seed_1)
# np.random.seed(rand_seed_2)
# torch.manual_seed(rand_seed_3)
# torch.backends.cudnn.deterministic = False
# torch.backends.cudnn.benchmark = True


def main():
    # setup the Argument Parser
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
    parser.add_argument('--conf', dest='conf', required=True,
                        help='JSoN file for loading arbitrary model parameters (e.g. optimizers, pre-saved files, etc.')
    parser.add_argument('--batch_size', type=int, required=False, default=None,
                        help='Overrides the batch_size given in the config file. Helpful for debugging')
    args = parser.parse_args()

    wandb.init(project="expred")
    # Configure
    os.makedirs(args.output_dir, exist_ok=True)

    # loads the config
    with open(args.conf, 'r') as fp:
        logger.info(f'Loading configuration from {args.conf}')
        conf = json.load(fp)
        if args.batch_size is not None:
            logger.info(
                'Overwriting batch_sizes'
                f'(mtl_token_identifier:{conf["mtl_token_identifier"]["batch_size"]}'
                f'evidence_classifier:{conf["evidence_classifier"]["batch_size"]})'
                f'provided in config by command line argument({args.batch_size})'
            )
            conf['mtl_token_identifier']['batch_size'] = args.batch_size
            conf['evidence_classifier']['batch_size'] = args.batch_size
        logger.info(f'Configuration: {json.dumps(conf, indent=2, sort_keys=True)}')

    # todo add seeds
    wandb.config.update(conf)
    wandb.config.update(args)

    # load the annotation data
    train, val, test = load_datasets(args.data_dir)

    # get's all docids needed that are contained in the loaded splits
    docids: Set[str] = set(e.docid for e in
                           chain.from_iterable(
                               chain.from_iterable(map(lambda ann: ann.evidences, chain(train, val, test))))
                        )

    documents: Dict[str, List[List[str]]] = load_documents(args.data_dir, docids)
    logger.info(f'Load {len(documents)} documents')
    # this ignores the case where annotations don't align perfectly with token boundaries, but this isn't that important

    tokenizer = BertTokenizerWithMapping.from_pretrained(conf['bert_vocab'])
    mtl_token_identifier, evidence_classifier, labels_mapping = \
        initialize_models(conf, tokenizer, batch_first=BATCH_FIRST)
    # logger.info(f'We have {len(word_interner)} wordpieces')

    # tokenizes and caches tokenized_docs, same for annotations
    # todo typo here? slides = slices (words?)
    tokenized_docs, tokenized_doc_token_slides = tokenizer.encode_docs(documents, args.output_dir)
    indexed_train, indexed_val, indexed_test = [tokenizer.encode_annotations(data) for data in [train, val, test]]

    logger.info('Beginning training of the MTL identifier')
    mtl_token_identifier = mtl_token_identifier.cuda()
    mtl_token_identifier, mtl_token_identifier_results, \
    train_machine_annotated, eval_machine_annotated, test_machine_annotated = \
        train_mtl_token_identifier(mtl_token_identifier,
                                   args.output_dir,
                                   indexed_train,
                                   indexed_val,
                                   indexed_test,
                                   labels_mapping=labels_mapping,
                                   interned_documents=tokenized_docs,
                                   source_documents=documents,
                                   token_mapping=tokenized_doc_token_slides,
                                   model_pars=conf,
                                   tensorize_model_inputs=True)
    mtl_token_identifier = mtl_token_identifier.cpu()
    # evidence identifier ends

    logger.info('Beginning training of the evidence classifier')
    evidence_classifier = evidence_classifier.cuda()
    optimizer = None
    scheduler = None

    # trains the classifier on the masked (based on rationales) documents
    evidence_classifier, evidence_class_results = train_mtl_evidence_classifier(evidence_classifier,
                                                                                args.output_dir,
                                                                                train_machine_annotated,
                                                                                eval_machine_annotated,
                                                                                tokenized_docs,
                                                                                conf,
                                                                                optimizer=optimizer,
                                                                                scheduler=scheduler,
                                                                                class_interner=labels_mapping,
                                                                                tensorize_model_inputs=True)
    # evidence classifier ends

    logger.info('Beginning final decoding')
    mtl_token_identifier = mtl_token_identifier.cuda()
    pipeline_batch_size = min(
        [conf['evidence_classifier']['batch_size'], conf['mtl_token_identifier']['batch_size']])
    pipeline_results, train_decoded, val_decoded, test_decoded = decode(evidence_identifier=mtl_token_identifier,
                                                                        evidence_classifier=evidence_classifier,
                                                                        train=indexed_train,
                                                                        mrs_train=train_machine_annotated,
                                                                        val=indexed_val,
                                                                        mrs_eval=eval_machine_annotated,
                                                                        test=indexed_test,
                                                                        mrs_test=test_machine_annotated,
                                                                        source_documents=documents,
                                                                        interned_documents=tokenized_docs,
                                                                        token_mapping=tokenized_doc_token_slides,
                                                                        class_interner=labels_mapping,
                                                                        tensorize_modelinputs=True,
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
    # decode ends



    wandb.save(os.path.join(args.output_dir, '*.jsonl'))



if __name__ == '__main__':
    main()
