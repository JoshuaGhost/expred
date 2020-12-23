from pytorch_transformers import BertModel, BertForMaskedLM, BertTokenizer, BertConfig
import logging
from typing import Any, List

import torch
import torch.nn as nn

from params import MTLParams
from eraserbenchmark.rationale_benchmark.models.model_utils import PaddedSequence
#from torch import optim
#from torch.optim import lr_scheduler
#import numpy as np
#import torchvision
#from torchvision import datasets, models, transforms
#import matplotlib.pyplot as plt
#import time
#import os
#import copy
#from torch.utils.data import Dataset, DataLoader
#from PIL import Image
#from random import randrange
#import torch.nn.functional as F


class BertLayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        super(BertLayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).power(2).mean(-1, keepdim=True)
        x = (x - u)/torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class BertMTL(nn.Module):
    def __init__(self,
                 bert_dir: str,
                 tokenizer: BertTokenizer,
                 mtl_params: MTLParams,
                 dim_linear_cls: int,
                 num_labels: int,
                 num_exp_gru_units: int,
                 max_length: int=512,
                 use_half_precision=True):
        super(BertMTL, self).__init__()
        self.bare_bert = BertModel.from_pretrained(bert_dir)
        if use_half_precision:
            import apex
            bert = self.bert.half()
        self.bert = bert
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.max_length = max_length

        self.exp_head = nn.Sequential(
            nn.GRU(self.bert.config.hidden_size, mtl_params.num_exp_gru_units),
            nn.Linear(mtl_params.num_exp_gru_units),
            nn.Sigmoid()
        )

        self.cls_head = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, mtl_params.dim_linear_cls, bias=True),
            nn.Tanh(),
            nn.Linear(mtl_params.dim_linear_cls, mtl_params.num_labels, bias=True),
            nn.Softmax()
        )
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        #input_ids, token_type_ids=None, attention_mask=None, labels=None):
        assert len(query) == len(document_batch)
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id].to(device=document_batch[0].device))
        sep_token = torch.tensor([self.sep_token_id].to(device=document_batch[0].device))
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 3 > self.max_length:
                d = d[:(self.max_length - len(q) - 3)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d, sep_token]))
            position_ids.append(torch.tensor(list(range(0, len(q)+1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        outputs = self.bert(bert_input.data, attention_mask=bert_input.mask(on=0.0, off=float('-inf', dtype=torch.float, device=target_device)))

        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        return sequence_output, pooled_output
    '''
    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True
    '''
    '''
    # building models
    from model import BertLayer
    from pytorch_transformers import BertConfig

    from tensorflow.keras.layers import CuDNNGRU, CuDNNLSTM
    from tensorflow.keras.layers import Bidirectional
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Dense, Reshape, Multiply, Concatenate, Dot, Lambda, Softmax

    from metrices import sp_precision_wrapper, sp_recall_wrapper
    from metrices_pytorch import sp_precision_wrapper, sp_recall_wrapper

    from
    '''
DIM_LINEAR_CLS = 256
NUM_GRU_UNITS_BERT_SEQ = 128
NUM_INTERVAL_LSTM_WIDTH = 100

class BERT_cls(nn.Module):
    def __init__(self, dim_input, dim_linear_cls=DIM_LINEAR_CLS,
                 num_cls=2, eps=1e-12):
        super(BERT_cls, self).__init__()
        self.dim_input = dim_input
        self.dim_linear_cls = dim_linear_cls
        self.num_cls = num_cls
        nn.lin1 = nn.Linear(self.dim_input, self.dim_linear_cls, bias=True)
        nn.act1 = nn.Tanh()
        nn.lin2 = nn.Linear(self.dim_linear_cls, self.num_cls, bias=True)
        nn.act2 = nn.Sigmoid()
        self.variance_epsilon = eps
        nn.init.uniform_(self.classifier.weight)

    def forward(self, x):
        x = self.act1(self.lin1(x))
        cls_output = self.act2(self.lin2(x))
        return cls_output


class GRU_exp(nn.Module):
    def __init__(self, dim_input, dim_exp_gru=NUM_GRU_UNITS_BERT_SEQ):
        super(GRU_exp, self).__init__()
        self.dim_exp_gru = dim_exp_gru
        self.gru = nn.GRU(dim_input, hidden_size=self.dim_exp_gru)
        self.exp_lin = nn.Linear(self.dim_exp_gru, 1)
        #gru = CuDNNGRU(
        #        NUM_GRU_UNITS_BERT_SEQ, kernel_initializer='random_uniform', return_sequences=True,
        #        name='exp_gru_gru')(
        #        bert_exp_output)
        ##    exp = Dense(1, activation='sigmoid', name='exp_gru_dense')(gru)
        #    output_mask = Reshape((MAX_SEQ_LENGTH, 1), name='exp_gru_reshape')(in_mask)
        #    exp_outputs = Multiply(name='exp_output')([output_mask, exp])
    def forward(self, x, input_mask):
        x = self.gru(x)
        exp_output = self.exp_lin(x)
        exp_output *= input_mask
        return exp_output


    def build_model():
        in_id = Input(shape=(MAX_SEQ_LENGTH,), name="input_ids")
        in_mask = Input(shape=(MAX_SEQ_LENGTH,), name="input_masks")
        in_segment = Input(shape=(MAX_SEQ_LENGTH,), name="segment_ids")
        bert_inputs = [in_id, in_mask, in_segment]
        bert_config = BertConfig.from_pretrained('bert-base-uncased')
        bert_cls_output, bert_exp_output = BertLayer(
            n_fine_tune_layers=10, name='bert')(bert_inputs)

        outputs = []
        if 'seq' not in dataset:
            # Classifier output
            dense = Dense(DIM_DENSE_CLS, activation='tanh', name='cls_dense')(bert_cls_output)
            cls_output = Dense(1, activation='sigmoid', name='cls_output')(dense)
            outputs.append(cls_output)
        if 'cls' not in dataset:
            # Explainer output
            if EXP_OUTPUT == 'gru':
                gru = CuDNNGRU(
                    NUM_GRU_UNITS_BERT_SEQ, kernel_initializer='random_uniform', return_sequences=True,
                    name='exp_gru_gru')(
                    bert_exp_output)
                exp = Dense(1, activation='sigmoid', name='exp_gru_dense')(gru)
                output_mask = Reshape((MAX_SEQ_LENGTH, 1), name='exp_gru_reshape')(in_mask)
                exp_outputs = Multiply(name='exp_output')([output_mask, exp])
            elif EXP_OUTPUT == 'rnr':
                M1 = Bidirectional(
                    layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True, name='exp_rnr_lstm1'),
                    merge_mode='concat', name='exp_rnr_bidirectional1')(bert_exp_output)
                p_starts = Dense(1, activation='sigmoid', name='exp_rnr_starts')(
                    Concatenate(axis=-1)([bert_exp_output, M1]))
                start_mask = Reshape((MAX_SEQ_LENGTH, 1))(in_mask)
                p_starts = Multiply()([p_starts, start_mask])

                m1_tilde = Dot(axes=-2)([p_starts, M1])
                M1_tilde = Lambda(lambda x: tf.tile(x, (1, MAX_SEQ_LENGTH, 1)))(m1_tilde)
                x = Multiply()([M1, M1_tilde])
                M2 = Bidirectional(
                    layer=CuDNNLSTM(NUM_INTERVAL_LSTM_WIDTH, return_sequences=True, name='exp_rnr_lstm2'),
                    merge_mode='concat', name='exp_rnr_bidirecitonal2')(
                    Concatenate(axis=-1)([bert_exp_output, M1, M1_tilde, x]))
                p_end_given_start = Dense(MAX_SEQ_LENGTH, activation='linear', name='exp_rnr_end')(
                    Concatenate(axis=-1)([bert_exp_output, M2]))
                end_mask = Lambda(lambda x: tf.tile(x, (1, MAX_SEQ_LENGTH, 1)))(Reshape((1, MAX_SEQ_LENGTH))(in_mask))
                p_end_given_start = Multiply()([p_end_given_start, end_mask])
                p_end_given_start = Lambda(lambda x: tf.linalg.band_part(x, 0, -1))(p_end_given_start)
                p_end_given_start = Softmax(axis=-1)(p_end_given_start)

                exp_outputs = Concatenate(axis=-1, name='exp_output')([p_starts, p_end_given_start])
            outputs.append(exp_outputs)

        model = Model(inputs=bert_inputs, outputs=outputs)
        optimizer = Adam(LEARNING_RATE)

        model.compile(loss=loss, loss_weights=loss_weights, optimizer=optimizer, metrics=metrics)

        model_exp = Model(inputs=bert_inputs, outputs=exp_outputs)
        optimizer = Adam(LEARNING_RATE)
        model_exp.compile(loss=loss['exp_output'], optimizer=optimizer, metrics=[metrics['exp_output']])

        model_cls = Model(inputs=bert_inputs, outputs=cls_output)
        optimizer = Adam(LEARNING_RATE)
        model_cls.compile(loss=loss['cls_output'], optimizer=optimizer, metrics=[metrics['cls_output']])

        return model, model_cls, model_exp

if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    text = 'what is a pug'# + ' pug'*1024 # induces error due to the overwhelming length
    zz = tokenizer.tokenize(text)
    print(len(zz))
    print(zz)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    bert_config = BertConfig.from_pretrained('bert-base-uncased')
    bertlayer = BertLayer(bert_config).to(device)

    zz = torch.tensor([tokenizer.convert_tokens_to_ids(zz)]).to(device)

    print(zz)

    print(bertlayer(zz))

