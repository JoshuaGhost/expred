import logging
from collections import defaultdict, namedtuple
from itertools import chain
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from sklearn.metrics import classification_report, accuracy_score

from expred.models.model_utils import PaddedSequence
from expred.models.pipeline.pipeline_utils import SentenceEvidence, _grouper, \
    score_rationales
from expred.utils import Annotation
from expred.eraser_benchmark import rational_bits_to_ev_generator
from expred.utils import convert_subtoken_ids_to_tokens

from expred.models.pipeline.mtl_token_identifier import chain_sentence_evidences


def mask_annotations_to_evidence_classification(mrs: List[Tuple[Tuple[str, SentenceEvidence], Any]], # mrs for machine rationales
                                                class_interner: dict) -> List[SentenceEvidence]:
    ret = []
    for mr, _, hard_prediction in mrs:
        kls = class_interner[mr[0]]
        evidence = mr[1]
        sentence = evidence.sentence
        query = evidence.query
        try:
            assert len(hard_prediction) == len(sentence)
        except Exception:
            print(mr)
            print(len(hard_prediction))
            print(len(query))
            print(len(sentence))
        masked_sentence = [p * d for p, d in zip(hard_prediction, sentence)]
        ret.append(SentenceEvidence(kls=kls,
                                   query=query,
                                   ann_id=evidence.ann_id,
                                   docid=evidence.docid,
                                   index=-1,
                                   sentence=masked_sentence))
    return ret


def annotations_to_evidence_token_identification(annotations: List[Annotation],
                                                 source_documents: Dict[str, List[List[str]]],
                                                 interned_documents: Dict[str, List[List[int]]],
                                                 token_mapping: Dict[str, List[List[Tuple[int, int]]]]
                                                 ) -> Dict[str, Dict[str, List[SentenceEvidence]]]:
    # TODO document
    # TODO should we simplify to use only source text?
    ret = defaultdict(lambda: defaultdict(list))  # annotation id -> docid -> sentences
    positive_tokens = 0
    negative_tokens = 0
    for ann in annotations:
        annid = ann.annotation_id
        docids = set(ev.docid for ev in chain.from_iterable(ann.evidences))
        sentence_offsets = defaultdict(list)  # docid -> [(start, end)]
        classes = defaultdict(list)  # docid -> [token is yea or nay]
        absolute_word_mapping = defaultdict(list)  # docid -> [(absolute wordpiece start, absolute wordpiece end)]
        for docid in docids:
            start = 0
            assert len(source_documents[docid]) == len(interned_documents[docid])
            for sentence_id, (whole_token_sent, wordpiece_sent) in enumerate(
                    zip(source_documents[docid], interned_documents[docid])):
                classes[docid].extend([0 for _ in wordpiece_sent])
                end = start + len(wordpiece_sent)
                sentence_offsets[docid].append((start, end))
                absolute_word_mapping[docid].extend([(start + relative_wp_start,
                                                      start + relative_wp_end)
                                                     for relative_wp_start,
                                                         relative_wp_end in token_mapping[docid][sentence_id]])
                start = end
        for ev in chain.from_iterable(ann.evidences):
            if len(ev.text) == 0:
                continue
            flat_token_map = list(chain.from_iterable(token_mapping[ev.docid]))
            if ev.start_token != -1 and ev.start_sentence != -1:
                # start, end = token_mapping[ev.docid][ev.start_token][0], token_mapping[ev.docid][ev.end_token][1]
                sentence_offset_start = sentence_offsets[ev.docid][ev.start_sentence][0]
                sentence_offset_end = sentence_offsets[ev.docid][ev.end_sentence - 1][0]
                start = sentence_offset_start + flat_token_map[ev.start_token][0]
                end = sentence_offset_end + flat_token_map[ev.end_token - 1][1]
            elif ev.start_token == -1 and ev.start_sentence != -1:
                start = sentence_offsets[ev.start_sentence][0]
                end = sentence_offsets[ev.end_sentence - 1][1]
            elif ev.start_token != -1 and ev.start_sentence == -1:
                start = absolute_word_mapping[ev.docid][ev.start_token][0]
                end = absolute_word_mapping[ev.docid][ev.end_token][1]
            else:
                continue
            for i in range(start, end):
                try:
                    classes[ev.docid][i] = 1
                except IndexError:
                    print(ev)
                    print(ev.docid)
                    print(classes)
                    print(len(classes[ev.docid]))
                    print(i)
                    raise IndexError
        for docid, offsets in sentence_offsets.items():
            token_assignments = classes[docid]
            positive_tokens += sum(token_assignments)
            negative_tokens += len(token_assignments) - sum(token_assignments)
            for s, (start, end) in enumerate(offsets):
                sent = interned_documents[docid][s]
                ret[annid][docid].append(SentenceEvidence(kls=tuple(token_assignments[start:end]),
                                                          query=ann.query,
                                                          ann_id=ann.annotation_id,
                                                          docid=docid,
                                                          index=s,
                                                          sentence=sent))
    logging.info(f"Have {positive_tokens} positive wordpiece tokens, {negative_tokens} negative wordpiece tokens")
    return ret


def annotations_to_mtl_token_identification(annotations: object,
                                            source_documents: object,
                                            interned_documents: object,
                                            token_mapping: object
                                            ) -> object:
    rets = annotations_to_evidence_token_identification(annotations, source_documents, interned_documents,
                                                        token_mapping)
    for ann in annotations:
        ann_id = ann.annotation_id
        ann_kls = ann.classification
        rets[ann_id] = [ann_kls, rets[ann_id]]
    return rets


# for mtl pipeline
def make_mtl_token_preds_batch(classifier: nn.Module,
                               batch_elements: List[SentenceEvidence],
                               labels_mapping: Dict[str, int],
                               token_mapping: Dict[str, List[List[Tuple[int, int]]]],
                               max_length: int,
                               par_lambda: int,
                               device=None,
                               cls_criterion: nn.Module = None,
                               exp_criterion: nn.Module = None,
                               tensorize_model_inputs: bool = True) -> Tuple[float, float, float, List[float], List[int], List[int]]:
    batch_elements = [s for s in batch_elements if s is not None]
    labels, targets, queries, sentences = zip(*[(s[0], s[1].kls, s[1].query, s[1].sentence)
                                                for s in batch_elements])
    labels = [[i == labels_mapping[label] for i in range(len(labels_mapping))] for label in labels]
    labels = torch.tensor(labels, dtype=torch.float, device=device)
    ids = [(s[1].ann_id, s[1].docid, s[1].index) for s in batch_elements]

    cropped_targets = [[0] * (len(query) + 2)  # length of query and overheads such as [cls] and [sep]
                       + list(target[:(max_length - len(query) - 2)]) for query, target in zip(queries, targets)]
    cropped_targets = PaddedSequence.autopad(
        [torch.tensor(t, dtype=torch.float, device=device) for t in cropped_targets],
        batch_first=True, device=device)

    targets = [[0] * (len(query) + 2)  # length of query and overheads such as [cls] and [sep]
               + list(target) for query, target in zip(queries, targets)]
    targets = PaddedSequence.autopad([torch.tensor(t, dtype=torch.float, device=device) for t in targets],
                                     batch_first=True, device=device)

    if tensorize_model_inputs:
        if all(q is None for q in queries):
            queries = [torch.tensor([], dtype=torch.long) for _ in queries]
        else:
            assert all(q is not None for q in queries)
            queries = [torch.tensor(q, dtype=torch.long) for q in queries]
        sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
    preds = classifier(queries, ids, sentences)
    cls_preds, exp_preds, attention_masks = preds
    cls_loss = cls_criterion(cls_preds, labels).mean(dim=-1).sum()
    cls_preds = [x.cpu().tolist() for x in cls_preds]
    labels = [x.cpu().tolist() for x in labels]

    exp_loss = exp_criterion(exp_preds, cropped_targets.data.squeeze()).mean(dim=-1).sum()
    #print(exp_loss.shape, cls_loss.shape)
    exp_preds = [x.cpu() for x in exp_preds]
    hard_preds = [torch.round(x).to(dtype=torch.int).cpu() for x in targets.unpad(exp_preds)]
    exp_preds = [x.tolist() for x in targets.unpad(exp_preds)]
    token_targets = [[y.item() for y in x] for x in targets.unpad(targets.data.cpu())]
    total_loss = cls_loss + par_lambda * exp_loss

    return total_loss, cls_loss, exp_loss, \
           exp_preds, hard_preds, token_targets, \
           cls_preds, labels


# for mtl pipeline
def make_mtl_token_preds_epoch(classifier: nn.Module,
                               data: List[SentenceEvidence],
                               labels_mapping: Dict[str, int],
                               token_mapping: Dict[str, List[List[Tuple[int, int]]]],
                               batch_size: int,
                               max_length: int,
                               par_lambda: int,
                               device=None,
                               cls_criterion: nn.Module = None,
                               exp_criterion: nn.Module = None,
                               tensorize_model_inputs: bool = True):
    epoch_total_loss = 0
    epoch_cls_loss = 0
    epoch_exp_loss = 0
    epoch_soft_pred = []
    epoch_hard_pred = []
    epoch_token_targets = []
    epoch_pred_labels = []
    epoch_labels = []
    batches = _grouper(data, batch_size)
    classifier.eval()
    #for p in classifier.parameters():
    #    print(str(p.device))
    #    print('cuda:0')
    #    print(str(p.device) == 'cuda:0')
    #    if str(p.device) != 'cuda:0':
    #        print(p)
    #        assert False
    for batch in batches:
        total_loss, cls_loss, exp_loss, \
        soft_preds, hard_preds, token_targets, \
        pred_labels, labels = make_mtl_token_preds_batch(classifier,
                                                         batch,
                                                         labels_mapping,
                                                         token_mapping,
                                                         max_length,
                                                         par_lambda,
                                                         device,
                                                         cls_criterion=cls_criterion,
                                                         exp_criterion=exp_criterion,
                                                         tensorize_model_inputs=tensorize_model_inputs)
        if total_loss is not None:
            epoch_total_loss += total_loss.sum().item()
        if cls_loss is not None:
            epoch_cls_loss += cls_loss.sum().item()
        if exp_loss is not None:
            epoch_exp_loss += exp_loss.sum().item()
        epoch_hard_pred.extend(hard_preds)
        epoch_soft_pred.extend(soft_preds)
        epoch_token_targets.extend(token_targets)
        epoch_pred_labels.extend(pred_labels)
        epoch_labels.extend(labels)
    epoch_total_loss /= len(data)
    epoch_cls_loss /= len(data)
    epoch_exp_loss /= len(data)
    return epoch_total_loss, epoch_cls_loss, epoch_exp_loss, \
           epoch_soft_pred, epoch_hard_pred, epoch_token_targets, \
           epoch_pred_labels, epoch_labels


def make_mtl_classification_preds_batch(classifier: nn.Module,
                     batch_elements: List[SentenceEvidence],
                     class_interner: dict,
                     device=None,
                     criterion: nn.Module = None,
                     tensorize_model_inputs: bool = True) -> Tuple[float, List[float], List[int], List[int]]:
    batch_elements = filter(lambda x: x is not None, batch_elements)
    targets, queries, sentences = zip(*[(s.kls, s.query, s.sentence) for s in batch_elements])
    ids = [(s.ann_id, s.docid, s.index) for s in batch_elements]
    targets = [[i == target for i in range(len(class_interner))] for target in targets]
    targets = torch.tensor(targets, dtype=torch.float, device=device)
    if tensorize_model_inputs:
        queries = [torch.tensor(q, dtype=torch.long) for q in queries]
        sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
    preds = classifier(queries, ids, sentences)
    targets = targets.to(device=preds.device)
    if criterion:
        loss = criterion(preds, targets)
    else:
        loss = None
    # .float() because pytorch 1.3 introduces a bug where argmax is unsupported for float16
    hard_preds = torch.argmax(preds.float(), dim=-1)
    return loss, preds, hard_preds, targets


def make_mtl_classification_preds_epoch(classifier: nn.Module,
                     data: List[SentenceEvidence],
                     class_interner: dict,
                     batch_size: int,
                     device=None,
                     criterion: nn.Module = None,
                     tensorize_model_inputs: bool = True):
    epoch_loss = 0
    epoch_soft_pred = []
    epoch_hard_pred = []
    epoch_truth = []
    batches = _grouper(data, batch_size)
    classifier.eval()
    for batch in batches:
        loss, soft_preds, hard_preds, targets = make_mtl_classification_preds_batch(classifier=classifier,
                                                                                     batch_elements=batch,
                                                                                     class_interner=class_interner,
                                                                                     device=device,
                                                                                     criterion=criterion,
                                                                                     tensorize_model_inputs=tensorize_model_inputs)
        if loss is not None:
            epoch_loss += loss.sum().item()
        epoch_hard_pred.extend(hard_preds)
        epoch_soft_pred.extend(soft_preds.cpu())
        epoch_truth.extend(targets)
    epoch_loss /= len(data)
    epoch_hard_pred = [x.item() for x in epoch_hard_pred]
    epoch_truth = [x.argmax().item() for x in epoch_truth]
    return epoch_loss, epoch_soft_pred, epoch_hard_pred, epoch_truth


def convert_to_global_token_mapping(token_mapping):
    ret = []
    sent_offset = 0
    for sent in token_mapping:
        for mapping in sent:
            ret.append((mapping[0]+sent_offset, mapping[1]+sent_offset))
        sent_offset = ret[-1][-1]
    return ret


def decode(evidence_identifier: nn.Module,
           evidence_classifier: nn.Module,
           train: List[Annotation],
           val: List[Annotation],
           test: List[Annotation],
           source_documents,
           token_mapping,
           mrs_train,
           mrs_eval,
           mrs_test,
           class_interner: Dict[str, int],
           batch_size: int,
           tensorize_modelinputs: bool,
           vocab,
           interned_documents: bool=None,
           tokenizer=None) -> dict:
    device = None
    class_labels = [k for k, v in sorted(class_interner.items(), key=lambda x: x[1])]
    if interned_documents is None:
        interned_documents = source_documents

    def prep(data: List[Annotation]) -> List[Tuple[SentenceEvidence, SentenceEvidence]]:
        identification_data = annotations_to_mtl_token_identification(annotations=data,
                                                                      source_documents=source_documents,
                                                                      interned_documents=interned_documents,
                                                                      token_mapping=token_mapping)

        identification_data = {ann_id: [v[0], {docid: chain_sentence_evidences(sentences)
                                               for docid, sentences in v[1].items()}]
                               for ann_id, v in identification_data.items()}
        classification_data = mask_annotations_to_evidence_classification(mrs=mrs_test,
                                                                          class_interner=class_interner)
        #ann_doc_sents = defaultdict(lambda: defaultdict(dict))  # ann id -> docid -> sent idx -> sent data
        ret = []
        for sent_ev in classification_data:
            id_data = identification_data[sent_ev.ann_id][1][sent_ev.docid]
            ret.append((id_data, sent_ev))
            assert id_data.ann_id == sent_ev.ann_id
            assert id_data.docid == sent_ev.docid
            #assert id_data.index == sent_ev.index
        assert len(ret) == len(classification_data)
        return ret

    def decode_batch(data: List[Tuple[SentenceEvidence, SentenceEvidence]],
                     mrs,
                     name: str,
                     score: bool = False,
                     annotations: List[Annotation] = None,
                     tokenizer=None) -> dict:
        """Identifies evidence statements and then makes classifications based on it.

        Args:
            data: a paired list of SentenceEvidences, differing only in the kls field.
                  The first corresponds to whether or not something is evidence, and the second corresponds to an evidence class
            name: a name for a results dict
        """

        num_uniques = len(set((x.ann_id, x.docid) for x, _ in data))
        logging.info(f'Decoding dataset {name} with {len(data)} sentences, {num_uniques} annotations')
        identifier_data, classifier_data = zip(*data)
        results = dict()
        IdentificationClassificationResult = namedtuple('IdentificationClassificationResult',
                                                        'identification_data classification_data soft_identification hard_identification soft_classification hard_classification')
        with torch.no_grad():
            # make predictions for the evidence_identifier
            evidence_identifier.eval()
            evidence_classifier.eval()
            _, soft_identification_preds, hard_identification_preds = zip(*mrs)
            assert len(soft_identification_preds) == len(data)
            identification_results = defaultdict(list)
            for id_data, cls_data, soft_id_pred, hard_id_pred in zip(identifier_data, classifier_data,
                                                                     soft_identification_preds,
                                                                     hard_identification_preds):
                res = IdentificationClassificationResult(identification_data=id_data,
                                                         classification_data=cls_data,
                                                         # 1 is p(evidence|sent,query)
                                                         soft_identification=soft_id_pred,
                                                         hard_identification=hard_id_pred,
                                                         soft_classification=None,
                                                         hard_classification=False)
                identification_results[(id_data.ann_id, id_data.docid)].append(res)  # in original eraser, each sentence
                # is stored separately, thence for each ann_idxdocid key there is a list of identification results, each
                # corresponds to a sentence. While in our approach a document is chained together from the begining and
                # rationalities are predicted in token-level granularity

            best_identification_results = {key: max(value, key=lambda x: x.soft_identification) for key, value in
                                           identification_results.items()}
            logging.info(
                f'Selected the best sentence for {len(identification_results)} examples from a total of {len(soft_identification_preds)} sentences')
            ids, classification_data = zip(
                *[(k, v.classification_data) for k, v in best_identification_results.items()])
            _, soft_classification_preds, hard_classification_preds, classification_truth = \
                make_mtl_classification_preds_epoch(classifier=evidence_classifier,
                                                    data=classification_data,
                                                    class_interner=class_interner,
                                                    batch_size=batch_size,
                                                    device=device,
                                                    tensorize_model_inputs=tensorize_modelinputs)
            classification_results = dict()
            for eyeD, soft_class, hard_class in zip(ids, soft_classification_preds, hard_classification_preds):
                input_id_result = best_identification_results[eyeD]
                res = IdentificationClassificationResult(identification_data=input_id_result.identification_data,
                                                         classification_data=input_id_result.classification_data,
                                                         soft_identification=input_id_result.soft_identification,
                                                         hard_identification=input_id_result.hard_identification,
                                                         soft_classification=soft_class,
                                                         hard_classification=hard_class)
                classification_results[eyeD] = res

            if score:
                truth = []
                pred = []
                for res in classification_results.values():
                    truth.append(res.classification_data.kls)
                    pred.append(res.hard_classification)
                # results[f'{name}_f1'] = classification_report(classification_truth, pred, target_names=class_labels, output_dict=True)
                results[f'{name}_f1'] = classification_report(classification_truth, hard_classification_preds,
                                                              target_names=class_labels,
                                                              labels=list(range(len(class_labels))), output_dict=True)
                results[f'{name}_acc'] = accuracy_score(classification_truth, hard_classification_preds)
                results[f'{name}_rationale'] = score_rationales(annotations, interned_documents, identifier_data,
                                                                soft_identification_preds)

            # turn the above results into a format suitable for scoring via the rationale scorer
            # n.b. the sentence-level evidence predictions (hard and soft) are
            # broadcast to the token level for scoring. The comprehensiveness class
            # score is also a lie since the pipeline model above is faithful by
            # design.
            decoded = dict()
            decoded_scores = defaultdict(list)
            for (ann_id, docid), pred in classification_results.items():
                #sentence_prediction_scores = [x.soft_identification for x in identification_results[(ann_id, docid)]]
                hard_rationale_predictions = list(chain.from_iterable(x.hard_identification for x in identification_results[(ann_id, docid)]))
                soft_rationale_predictions = list(chain.from_iterable(x.soft_identification for x in identification_results[(ann_id, docid)]))
                subtoken_ids =  list(chain.from_iterable(interned_documents[docid]))
                raw_document = []
                for word in chain.from_iterable(source_documents[docid]):
                    token_ids_origin = tokenizer.encode(word, add_special_tokens=False)
                    if token_ids_origin[0] == tokenizer.unk_token_id:
                        raw_document.append('[UNK]')
                    else:
                        tokenized = ''.join(tokenizer.basic_tokenizer.tokenize(word)) # dumm ass ˈlʊdvɪɡ_væn_ˈbeɪˌtoʊvən
                        raw_document.append(tokenized)
                global_token_mapping = convert_to_global_token_mapping(token_mapping[docid])
                tokens, exp_outputs = convert_subtoken_ids_to_tokens(subtoken_ids,
                                                                     vocab=vocab,
                                                                     token_mapping=global_token_mapping,
                                                                     exps=(hard_rationale_predictions,
                                                                           soft_rationale_predictions),
                                                                     raw_sentence=raw_document)
                hard_rationale_predictions, soft_rationale_predictions = list(zip(*exp_outputs))
                # if docid == 'Ludwig_van_Beethoven':
                #     #print(len(hard_rationale_predictions))
                #     print(len(soft_rationale_predictions))
                ev_generator = rational_bits_to_ev_generator(tokens,
                                                             docid,
                                                             hard_rationale_predictions)
                hard_rationale_predictions = [ev for ev in ev_generator]

                if ann_id not in decoded:
                    decoded[ann_id] = {
                        "annotation_id": ann_id,
                        "rationales": [],
                        "classification": class_labels[pred.hard_classification],
                        "classification_scores": {class_labels[i]: s.item() for i, s in
                                                  enumerate(pred.soft_classification)},
                        # TODO this should turn into the data distribution for the predicted class
                        # "comprehensiveness_classification_scores": 0.0,
                        "truth": pred.classification_data.kls,
                    }
                decoded[ann_id]['rationales'].append({
                    "docid": docid,
                    "hard_rationale_predictions": hard_rationale_predictions,
                    "soft_rationale_predictions": soft_rationale_predictions,
                })
                decoded_scores[ann_id].append(pred.soft_classification)

            return results, list(decoded.values())

    test_results, test_decoded = decode_batch(prep(test), mrs_test, 'test', score=False, tokenizer=tokenizer)
    val_results, val_decoded = dict(), []
    train_results, train_decoded = dict(), []
    # val_results, val_decoded = decode_batch(prep(val), 'val', score=True, annotations=val)
    # train_results, train_decoded = decode_batch(prep(train), 'train', score=True, annotations=train)
    return dict(**train_results, **val_results, **test_results), train_decoded, val_decoded, test_decoded
