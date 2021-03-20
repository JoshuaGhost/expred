import torch
import torch.nn as nn

from transformers import BertModel, BertTokenizer
from expred.params import MTLParams
from typing import Any, List
from expred.models.model_utils import PaddedSequence


class BertMTL(nn.Module):
    def __init__(self,
                 bert_dir: str,
                 tokenizer: BertTokenizer,
                 mtl_params: MTLParams,
                 max_length: int=512,
                 use_half_precision=True):
        super(BertMTL, self).__init__()
        bare_bert = BertModel.from_pretrained(bert_dir)
        if use_half_precision:
            import apex
            bare_bert = bare_bert.half()
        self.bare_bert = bare_bert
        self.pad_token_id = tokenizer.pad_token_id
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        self.max_length = max_length

        class ExpHead(nn.Module):
            def __init__(self, input_size, hidden_size):
                super(ExpHead, self).__init__()
                self.exp_gru = nn.GRU(input_size, hidden_size)
                self.exp_linear = nn.Linear(hidden_size, 1, bias=True)
                self.exp_act = nn.Sigmoid()

            def forward(self, x):
                return self.exp_act(self.exp_linear(self.exp_gru(x)[0]))

        #self.exp_head = lambda x: exp_act(exp_linear(exp_gru(x)[0]))
        self.exp_head = ExpHead(self.bare_bert.config.hidden_size, mtl_params.dim_exp_gru)
        self.cls_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.bare_bert.config.hidden_size, mtl_params.dim_cls_linear, bias=True),
            nn.Tanh(),
            nn.Linear(mtl_params.dim_cls_linear, mtl_params.num_labels, bias=True),
            nn.Softmax(dim=-1)
        )
        for layer in self.cls_head:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
        nn.init.xavier_normal_(self.exp_head.exp_linear.weight)

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        #input_ids, token_type_ids=None, attention_mask=None, labels=None):
        assert len(query) == len(document_batch)
        #print(next(self.cls_head.parameters()).device)
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id,
                                            device=target_device)
        attention_mask = bert_input.mask(on=1., off=0., device=target_device)
        exp_output, cls_output = self.bare_bert(bert_input.data, attention_mask=attention_mask)
        exp_output = self.exp_head(exp_output).squeeze() * attention_mask
        cls_output = self.cls_head(cls_output)
        assert torch.all(cls_output == cls_output)
        assert torch.all(exp_output == exp_output)
        return cls_output, exp_output, attention_mask


class BertClassifier(nn.Module):
    """Thin wrapper around BertForSequenceClassification"""
    def __init__(self,
                 bert_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 num_labels: int,
                 mtl_params: MTLParams,
                 max_length: int=512,
                 use_half_precision=True):
        super(BertClassifier, self).__init__()
        bert = BertModel.from_pretrained(bert_dir, num_labels=num_labels)
        if use_half_precision:
            import apex
            bert = bert.half()
        self.bert = bert
        self.cls_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(bert.config.hidden_size, mtl_params.dim_cls_linear, bias=True),
            nn.Tanh(),
            nn.Linear(mtl_params.dim_cls_linear, mtl_params.num_labels, bias=True),
            nn.Softmax(dim=-1)
        )
        for layer in self.cls_head:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these conf on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        _, classes = self.bert(bert_input.data, attention_mask=bert_input.mask(on=1., off=0., device=target_device), position_ids=positions.data)
        classes = self.cls_head(classes)
        assert torch.all(classes == classes) # for nans
        return classes


class BertClassifier2(nn.Module):
    """Thin wrapper around BertForSequenceClassification"""
    def __init__(self,
                 bert_dir: str,
                 pad_token_id: int,
                 cls_token_id: int,
                 sep_token_id: int,
                 num_labels: int,
                 mtl_params: MTLParams,
                 max_length: int=512,
                 use_half_precision=True):
        super(BertClassifier2, self).__init__()
        bert = BertModel.from_pretrained(bert_dir, num_labels=num_labels)
        if use_half_precision:
            import apex
            bert = bert.half()
        self.bert = bert
        self.cls_head = nn.Sequential(
            #nn.Dropout(0.1),
            nn.Linear(bert.config.hidden_size, mtl_params.dim_cls_linear, bias=True),
            nn.Tanh(),
            nn.Linear(mtl_params.dim_cls_linear, mtl_params.num_labels, bias=True),
            nn.Softmax(dim=-1)
        )
        for layer in self.cls_head:
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
        self.pad_token_id = pad_token_id
        self.cls_token_id = cls_token_id
        self.sep_token_id = sep_token_id
        self.max_length = max_length

    def forward(self,
                query: List[torch.tensor],
                docids: List[Any],
                document_batch: List[torch.tensor]):
        assert len(query) == len(document_batch)
        # note about device management:
        # since distributed training is enabled, the inputs to this module can be on *any* device (preferably cpu, since we wrap and unwrap the module)
        # we want to keep these conf on the input device (assuming CPU) for as long as possible for cheap memory access
        target_device = next(self.parameters()).device
        cls_token = torch.tensor([self.cls_token_id]).to(device=document_batch[0].device)
        sep_token = torch.tensor([self.sep_token_id]).to(device=document_batch[0].device)
        input_tensors = []
        position_ids = []
        for q, d in zip(query, document_batch):
            if len(q) + len(d) + 2 > self.max_length:
                d = d[:(self.max_length - len(q) - 2)]
            input_tensors.append(torch.cat([cls_token, q, sep_token, d]))
            position_ids.append(torch.tensor(list(range(0, len(q) + 1)) + list(range(0, len(d) + 1))))
        bert_input = PaddedSequence.autopad(input_tensors, batch_first=True, padding_value=self.pad_token_id, device=target_device)
        positions = PaddedSequence.autopad(position_ids, batch_first=True, padding_value=0, device=target_device)
        _, classes = self.bert(bert_input.data, attention_mask=bert_input.mask(on=1., off=0., device=target_device), position_ids=positions.data)
        classes = self.cls_head(classes)
        assert torch.all(classes == classes) # for nans
        return classes