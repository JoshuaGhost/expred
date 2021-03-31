import logging
import os
import random

from collections import OrderedDict
from typing import Dict, List, Tuple, Any

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score, classification_report

# from expred.utils import Annotation

from expred.models.pipeline.pipeline_utils import SentenceEvidence
from expred.models.pipeline.mtl_pipeline_utils import (
    mask_annotations_to_evidence_classification,
    make_mtl_classification_preds_epoch
)


def train_mtl_evidence_classifier(evidence_classifier: nn.Module,
                                  save_dir: str,
                                  train: Tuple[List[Tuple[str, SentenceEvidence]], Any],
                                  val: Tuple[List[Tuple[str, SentenceEvidence]], Any],
                                  documents: Dict[str, List[List[int]]],
                                  model_pars: dict,
                                  class_interner: Dict[str, int],
                                  optimizer=None,
                                  scheduler=None,
                                  tensorize_model_inputs: bool = True) -> Tuple[nn.Module, dict]:
    """

    :param evidence_classifier:
    :param save_dir:
    :param train:
    :param val:
    :param documents:
    :param model_pars:
    :param class_interner:
    :param optimizer:
    :param scheduler:
    :param tensorize_model_inputs:
    :return:
    """
    logging.info(
        f'Beginning training evidence classifier with {len(train[0])} annotations, {len(val[0])} for validation')
    # set up output directories
    evidence_classifier_output_dir = os.path.join(save_dir, 'evidence_classifier')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(evidence_classifier_output_dir, exist_ok=True)
    model_save_file = os.path.join(evidence_classifier_output_dir, 'evidence_classifier.pt')
    epoch_save_file = os.path.join(evidence_classifier_output_dir, 'evidence_classifier_epoch_data.pt')

    # set up training (optimizer, loss, patience, ...)
    device = next(evidence_classifier.parameters()).device
    if optimizer is None:
        optimizer = torch.optim.Adam(evidence_classifier.parameters(), lr=model_pars['evidence_classifier']['lr'])
    criterion = nn.BCELoss(reduction='none')
    batch_size = model_pars['evidence_classifier']['batch_size']
    epochs = model_pars['evidence_classifier']['epochs']
    patience = model_pars['evidence_classifier']['patience']
    max_grad_norm = model_pars['evidence_classifier'].get('max_grad_norm', None)

    # mask out the hard prediction (token 0) and convert to [SentenceEvidence...]
    evidence_train_data = mask_annotations_to_evidence_classification(train, class_interner)
    evidence_val_data = mask_annotations_to_evidence_classification(val, class_interner)

    class_labels = [k for k, v in sorted(class_interner.items())]

    results = {
        'train_loss': [],
        'train_f1': [],
        'train_acc': [],
        'val_loss': [],
        'val_f1': [],
        'val_acc': [],
    }
    best_epoch = -1
    best_val_loss = float('inf')
    best_model_state_dict = None
    start_epoch = 0
    epoch_data = {}
    if os.path.exists(epoch_save_file):
        logging.info(f'Restoring model from {model_save_file}')
        evidence_classifier.load_state_dict(torch.load(model_save_file))
        epoch_data = torch.load(epoch_save_file)
        start_epoch = epoch_data['epoch'] + 1
        # handle finishing because patience was exceeded or we didn't get the best final epoch
        if bool(epoch_data.get('done', 0)):
            start_epoch = epochs
        results = epoch_data['results']
        best_epoch = start_epoch
        best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
        logging.info(f'Restoring training from epoch {start_epoch}')
    logging.info(f'Training evidence classifier from epoch {start_epoch} until epoch {epochs}')
    optimizer.zero_grad()
    for epoch in range(start_epoch, epochs):
        epoch_train_data = random.sample(evidence_train_data, k=len(evidence_train_data))
        epoch_val_data = random.sample(evidence_val_data, k=len(evidence_val_data))
        epoch_train_loss = 0
        evidence_classifier.train()
        logging.info(
            f'Training with {len(epoch_train_data) // batch_size} batches with {len(epoch_train_data)} examples')
        for batch_start in range(0, len(epoch_train_data), batch_size):
            batch_elements = epoch_train_data[batch_start:min(batch_start + batch_size, len(epoch_train_data))]
            targets, queries, sentences = zip(*[(s.kls, s.query, s.sentence) for s in batch_elements])
            ids = [(s.ann_id, s.docid, s.index) for s in batch_elements]
            targets = [[i == target for i in range(len(class_interner))] for target in targets]
            targets = torch.tensor(targets, dtype=torch.float, device=device)
            if tensorize_model_inputs:
                queries = [torch.tensor(q, dtype=torch.long) for q in queries]
                sentences = [torch.tensor(s, dtype=torch.long) for s in sentences]
            preds = evidence_classifier(queries, ids, sentences)
            loss = criterion(preds, targets.to(device=preds.device)).sum()
            epoch_train_loss += loss.item()
            loss = loss / len(preds)  # accumulate entire loss above
            loss.backward()
            assert loss == loss  # for nans
            if max_grad_norm:
                torch.nn.utils.clip_grad_norm_(evidence_classifier.parameters(), max_grad_norm)
            optimizer.step()
            if scheduler:
                scheduler.step()
            optimizer.zero_grad()
        epoch_train_loss /= len(epoch_train_data)
        assert epoch_train_loss == epoch_train_loss  # for nans
        results['train_loss'].append(epoch_train_loss)
        logging.info(f'Epoch {epoch} training loss {epoch_train_loss}')

        with torch.no_grad():
            evidence_classifier.eval()
            epoch_train_loss, \
            epoch_train_soft_pred, \
            epoch_train_hard_pred, \
            epoch_train_truth = make_mtl_classification_preds_epoch(
                classifier=evidence_classifier,
                data=epoch_train_data,
                class_interner=class_interner,
                batch_size=batch_size,
                device=device,
                criterion=criterion,
                tensorize_model_inputs=tensorize_model_inputs)
            results['train_f1'].append(
                classification_report(epoch_train_truth, epoch_train_hard_pred, target_names=class_labels,
                                      labels=list(range(len(class_labels))), output_dict=True))
            results['train_acc'].append(accuracy_score(epoch_train_truth, epoch_train_hard_pred))
            epoch_val_loss, \
            epoch_val_soft_pred, \
            epoch_val_hard_pred, \
            epoch_val_truth = make_mtl_classification_preds_epoch(
                classifier=evidence_classifier,
                data=epoch_val_data,
                class_interner=class_interner,
                batch_size=batch_size,
                device=device,
                criterion=criterion,
                tensorize_model_inputs=tensorize_model_inputs)
            results['val_loss'].append(epoch_val_loss)
            results['val_f1'].append(
                classification_report(epoch_val_truth, epoch_val_hard_pred, target_names=class_labels,
                                      labels=list(range(len(class_labels))), output_dict=True))
            results['val_acc'].append(accuracy_score(epoch_val_truth, epoch_val_hard_pred))
            assert epoch_val_loss == epoch_val_loss  # for nans
            logging.info(f'Epoch {epoch} val loss {epoch_val_loss}')
            logging.info(f'Epoch {epoch} val acc {results["val_acc"][-1]}')
            logging.info(f'Epoch {epoch} val f1 {results["val_f1"][-1]}')

            if epoch_val_loss < best_val_loss:
                best_model_state_dict = OrderedDict({k: v.cpu() for k, v in evidence_classifier.state_dict().items()})
                best_epoch = epoch
                best_val_loss = epoch_val_loss
                epoch_data = {
                    'epoch': epoch,
                    'results': results,
                    'best_val_loss': best_val_loss,
                    'done': 0,
                }
                torch.save(evidence_classifier.state_dict(), model_save_file)
                torch.save(epoch_data, epoch_save_file)
                logging.debug(f'Epoch {epoch} new best model with val loss {epoch_val_loss}')
        if epoch - best_epoch > patience:
            logging.info(f'Exiting after epoch {epoch} due to no improvement')
            epoch_data['done'] = 1
            torch.save(epoch_data, epoch_save_file)
            break

    epoch_data['done'] = 1
    epoch_data['results'] = results
    torch.save(epoch_data, epoch_save_file)
    evidence_classifier.load_state_dict(best_model_state_dict)
    evidence_classifier = evidence_classifier.to(device=device)
    evidence_classifier.eval()
    return evidence_classifier, results
