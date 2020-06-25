from pytorch_transformers import BertModel, BertForMaskedLM, BertTokenizer, BertConfig
import logging


import torch
import torch.nn as nn
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


class BertLayer(nn.Module):
    def __init__(self):
        super(BertLayer, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
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
    DIM_DENSE_CLS = 256
    NUM_GRU_UNITS_BERT_SEQ = 128
    NUM_INTERVAL_LSTM_WIDTH = 100

class BERT_cls(nn.Module):
    def __init__(self, dim_input, dim_cls_dense=256, num_cls=2, eps=1e-12):
        super(BERT_cls, self).__init__()
        self.dim_input = dim_input
        self.dim_cls_dense = dim_cls_dense
        self.num_cls = num_cls
        nn.lin1 = nn.Linear(self.dim_input, self.dim_cls_dense, bias=True)
        nn.act1 = nn.Tanh()
        nn.lin2 = nn.Linear(self.dim_cls_dense, self.dim_input, bias=True)
        nn.act2 = nn.Sigmoid()
        self.variance_epsilon = eps
        nn.init.uniform_(self.classifier.weight)

    def forward(self, x):
        x = self.act1(self.lin1(x))
        x = self.act2(self.lin2(x))


    class GRU_exp(nn.Module):
        def __init__(self, dim_input, dim_exp_gru=100):
            super(GRU_exp, self).__init__()
            self.dim_exp_gru = dim_exp_gru
            self.gru = nn.GRU(dim_input, hidden_size=self.dim_exp_gru)
            self.exp_lin = nn.Linear(self.dim_exp_gru, 1)
                       gru = CuDNNGRU(
                    NUM_GRU_UNITS_BERT_SEQ, kernel_initializer='random_uniform', return_sequences=True,
                    name='exp_gru_gru')(
                    bert_exp_output)
                exp = Dense(1, activation='sigmoid', name='exp_gru_dense')(gru)
                output_mask = Reshape((MAX_SEQ_LENGTH, 1), name='exp_gru_reshape')(in_mask)
                exp_outputs = Multiply(name='exp_output')([output_mask, exp])

    class L2I_model(nn.Module):
        def __init__(self, dim_cls_dense=256, dim_bilstm=128, dim_exp_gru=100):
            super(L2I_model, self).__init__()
            self.dim_cls_dense = dim_cls_dense
            self.dim_bilstm = dim_bilstm
            self.dim_exp_gru=


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

