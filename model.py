from pytorch_transformers import BertModel, BertForMaskedLM, BertTokenizer, BertConfig
import logging
logging.basicConfig(level=logging.INFO)
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from random import randrange
import torch.nn.functional as F


text = 'what is a pug'
zz = tokenizer.tokenize(text)

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

    def __init__(self, config, num_labels=2):
        super(BertLayer, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        nn.init.xavier_normal_(self.classifier.weight)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None):
        sequence_output, pooled_output = self.bert(input_ids=input_ids,
                                                   token_type_ids=token_type_ids,
                                                   attention_mask=attention_mask)
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

    def freeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_bert_encoder(self):
        for param in self.bert.parameters():
            param.requires_grad = True


config = BertConfig.from_pretrained('bert-base-uncased')


'''
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
from tensorflow.keras import backend as K
import tensorflow_hub as hub


class BertLayer(tf.keras.layers.Layer):
    def __init__(
            self,
            n_fine_tune_layers=10,
            bert_path="https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1",
            pooling='first',  # or 'mean'
            **kwargs
    ):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        self.bert_path = bert_path
        self.pooling = pooling

        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            self.bert_path, trainable=self.trainable, name="{}_module".format(self.name)
        )

        # Remove unused layers
        trainable_vars = self.bert.variables
        trainable_vars = [
            var
            for var in trainable_vars
            if not "/cls/" in var.name and not "/pooler/" in var.name
        ]
        trainable_layers = []

        # Select how many layers to fine tune
        for i in range(self.n_fine_tune_layers):
            trainable_layers.append("encoder/layer_{}".format(str(11 - i)))

        # Update trainable vars to contain only the specified layers
        trainable_vars = [
            var
            for var in trainable_vars
            if any([l in var.name for l in trainable_layers])
        ]

        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)

        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)

        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        cls = self.bert(inputs=bert_inputs, signature="tokens",
                        as_dict=True)["pooled_output"]
        exp = self.bert(inputs=bert_inputs, signature="tokens",
                        as_dict=True)["sequence_output"]

        def mul_mask(x, m): return x * tf.expand_dims(m, axis=-1)

        input_mask = tf.cast(input_mask, tf.float32)
        exp = mul_mask(exp, input_mask)

        if self.pooling == 'mean':
            def masked_reduce_mean(x, m): return tf.reduce_sum(mul_mask(x, m), axis=1) / (
                    tf.reduce_sum(m, axis=1, keepdims=True) + 1e-10)

            cls = masked_reduce_mean(exp, input_mask)
        return [cls, exp]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)
'''
