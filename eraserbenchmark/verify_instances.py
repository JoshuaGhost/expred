from typing import Any, Callable, Dict, List, Tuple
from collections import Counter
import logging

def verify_instance(instance: dict, docs: Dict[str, list]):
    error = False
    docids = []
    # verify the internal structure of these instances is correct:
    # * hard predictions are present
    # * start and end tokens are valid
    # * soft rationale predictions, if present, must have the same document length

    for rat in instance['rationales']:
        docid = rat['docid']
        if docid not in docid:
            error = True
            logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} could not be found as a preprocessed document! Gave up on additional processing.')
            continue
        doc_length = len(docs[docid])
        for h1 in rat.get('hard_rationale_predictions', []):
            # verify that each token is valid
            # verify that no annotations overlap
            for h2 in rat.get('hard_rationale_predictions', []):
                if h1 == h2:
                    continue
                try:
                    if len(set(range(h1['start_token'], h1['end_token'])) & set(range(h2['start_token'], h2['end_token']))) > 0:
                        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} {h1} and {h2} overlap!')
                        error = True
                except TypeError:
                    print(h1, h2)
                    raise TypeError
            if h1['start_token'] > doc_length:
                logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} received an impossible tokenspan: {h1} for a document of length {doc_length}')
                error = True
            if h1['end_token'] > doc_length:
                logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} received an impossible tokenspan: {h1} for a document of length {doc_length}')
                error = True
        # length check for soft rationale
        # note that either flattened_documents or sentence-broken documents must be passed in depending on result
        soft_rationale_predictions = rat.get('soft_rationale_predictions', [])
        if len(soft_rationale_predictions) > 0 and len(soft_rationale_predictions) != doc_length:
            logging.info(f'Error! For instance annotation={instance["annotation_id"]}, docid={docid} expected classifications for {doc_length} tokens but have them for {len(soft_rationale_predictions)} tokens instead!')
            error = True

    # count that one appears per-document
    docids = Counter(docids)
    for docid, count in docids.items():
        if count > 1:
            error = True
            logging.info('Error! For instance annotation={instance["annotation_id"]}, docid={docid} appear {count} times, may only appear once!')

    classification = instance.get('classification', '')
    if not isinstance(classification, str):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, classification field {classification} is not a string!')
        error = True
    classification_scores = instance.get('classification_scores', dict())
    if not isinstance(classification_scores, dict):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, classification_scores field {classification_scores} is not a dict!')
        error = True
    comprehensiveness_classification_scores = instance.get('comprehensiveness_classification_scores', dict())
    if not isinstance(comprehensiveness_classification_scores, dict):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, comprehensiveness_classification_scores field {comprehensiveness_classification_scores} is not a dict!')
        error = True
    sufficiency_classification_scores = instance.get('sufficiency_classification_scores', dict())
    if not isinstance(sufficiency_classification_scores, dict):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, sufficiency_classification_scores field {sufficiency_classification_scores} is not a dict!')
        error = True
    if ('classification' in instance) != ('classification_scores' in instance):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, when providing a classification, you must also provide classification scores!')
        error = True
    if ('comprehensiveness_classification_scores' in instance) and not ('classification' in instance):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, when providing a classification, you must also provide a comprehensiveness_classification_score')
        error = True
    if ('sufficiency_classification_scores' in instance) and not ('classification_scores' in instance):
        logging.info(f'Error! For instance annotation={instance["annotation_id"]}, when providing a sufficiency_classification_score, you must also provide a classification score!')
        error = True
    return error

def verify_instances(instances: List[dict], docs: Dict[str, list]):
    annotation_ids = list(x['annotation_id'] for x in instances)
    key_counter = Counter(annotation_ids)
    multi_occurrence_annotation_ids = list(filter(lambda kv: kv[1] > 1, key_counter.items()))
    error = False
    if len(multi_occurrence_annotation_ids) > 0:
        error = True
        logging.info(f'Error in instances: {len(multi_occurrence_annotation_ids)} appear multiple times in the annotations file: {multi_occurrence_annotation_ids}')
    failed_validation = set()
    instances_with_classification = list()
    instances_with_soft_rationale_predictions = list()
    instances_with_soft_sentence_predictions = list()
    instances_with_comprehensiveness_classifications = list()
    instances_with_sufficiency_classifications = list()
    for instance in instances:
        instance_error = verify_instance(instance, docs)
        if instance_error:
            error = True
            failed_validation.add(instance['annotation_id'])
        if instance.get('classification', None) != None:
            instances_with_classification.append(instance)
        if instance.get('comprehensiveness_classification_scores', None) != None:
            instances_with_comprehensiveness_classifications.append(instance)
        if instance.get('sufficiency_classification_scores', None) != None:
            instances_with_sufficiency_classifications.append(instance)
        has_soft_rationales = []
        has_soft_sentences = []
        for rat in instance['rationales']:
            if rat.get('soft_rationale_predictions', None) != None:
                has_soft_rationales.append(rat)
            if rat.get('soft_sentence_predictions', None) != None:
                has_soft_sentences.append(rat)
        if len(has_soft_rationales) > 0:
            instances_with_soft_rationale_predictions.append(instance)
            if len(has_soft_rationales) != len(instance['rationales']):
                error = True
                logging.info(f'Error: instance {instance["annotation"]} has soft rationales for some but not all reported documents!')
        if len(has_soft_sentences) > 0:
            instances_with_soft_sentence_predictions.append(instance)
            if len(has_soft_sentences) != len(instance['rationales']):
                error = True
                logging.info(f'Error: instance {instance["annotation"]} has soft sentences for some but not all reported documents!')
    logging.info(f'Error in instances: {len(failed_validation)} instances fail validation: {failed_validation}')
    if len(instances_with_classification) != 0 and len(instances_with_classification) != len(instances):
        logging.info(f'Either all {len(instances)} must have a classification or none may, instead {len(instances_with_classification)} do!')
        error = True
    if len(instances_with_soft_sentence_predictions) != 0 and len(instances_with_soft_sentence_predictions) != len(instances):
        logging.info(f'Either all {len(instances)} must have a sentence prediction or none may, instead {len(instances_with_soft_sentence_predictions)} do!')
        error = True
    if len(instances_with_soft_rationale_predictions) != 0 and len(instances_with_soft_rationale_predictions) != len(instances):
        logging.info(f'Either all {len(instances)} must have a soft rationale prediction or none may, instead {len(instances_with_soft_rationale_predictions)} do!')
        error = True
    if len(instances_with_comprehensiveness_classifications) != 0 and len(instances_with_comprehensiveness_classifications) != len(instances):
        logging.info(f'Either all {len(instances)} must have a comprehensiveness classification or none may, instead {len(instances_with_comprehensiveness_classifications)} do!')
    if len(instances_with_sufficiency_classifications) != 0 and len(instances_with_sufficiency_classifications) != len(instances):
        logging.info(f'Either all {len(instances)} must have a sufficiency classification or none may, instead {len(instances_with_sufficiency_classifications)} do!')
    if error:
        raise ValueError('Some instances are invalid, please fix your formatting and try again')

def _has_hard_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'hard_rationale_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['hard_rationale_predictions'] is not None

def _has_soft_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'soft_rationale_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['soft_rationale_predictions'] is not None

def _has_soft_sentence_predictions(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'rationales' in results[0] and len(results[0]['rationales']) > 0 and 'soft_sentence_predictions' in results[0]['rationales'][0] and results[0]['rationales'][0]['soft_sentence_predictions'] is not None

def _has_classifications(results: List[dict]) -> bool:
    # assumes that we have run "verification" over the inputs
    return 'classification' in results[0] and results[0]['classification'] is not None