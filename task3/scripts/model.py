#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File      : model.py
# Author    : Rui-Zhao
# Date      : 2022/10/13 19:11 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import pickle as pkl
from utils import join

class Config(object):
    def __init__(self, embed_path=None, vocab_path=None):
        # pararms for dataset
        self.train_path = join(r'data/snli_1.0/snli_1.0_train.jsonl')
        self.dev_path = join(r'data/snli_1.0/snli_1.0_dev.jsonl')
        self.test_path = join(r'data/snli_1.0/snli_1.0_test.jsonl')
        self.save_path = './ouput/saved_dict.ckpt'
        if not os.path.exists(os.path.dirname(self.save_path)):
            os.makedirs(os.path.dirname(self.save_path))

        self.vocab_map = pkl.load(open(vocab_path, 'rb')) if vocab_path else pkl.load(
            open(join(r'pretrained_models/glove/standford/vocab_map.pkl'), 'rb'))
        self.n_vocab = len(self.vocab_map)
        self.label_map = pkl.load(open(join(r'data/snli_1.0/label_map.pkl'), 'rb'))
        self.label_list = list(self.label_map.keys())
        self.label_num = len(self.label_map)

        # params for train
        self.pad_size_p = 64
        self.pad_size_h = 32

        self.batch_size = 32
        self.epoch = 20
        self.learning_rate = 1e-4
        self.drop_out = 0.3
        self.log_step = 1000
        self.require_improvement = self.log_step * 10
        self.early_stop = False

        # params for embedding
        self.embedding_pretrained = torch.FloatTensor(
            pkl.load(open(embed_path, 'rb'))) if embed_path is not None else None
        self.embedding_size = self.embedding_pretrained.size(1) if self.embedding_pretrained is not None else 600

        # params for rnn_like models
        self.hidden_size = 600
        self.num_layer = 2
        self.bidirectional = True

        # param for MLP
        self.linear_size = 600

        # params for cnn model
        self.num_channel = 256
        self.kernel_sizes = (3, 4, 5)

        # device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

class ESIM(nn.Module):
    def __init__(self, config):
        super(ESIM, self).__init__()

        # 1. embedding layer
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False,
                                                          padding_idx=config.n_vocab - 1)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embedding_size, padding_idx=config.n_vocab - 1)
        # 2.input encoding layer
        self.encoding = nn.LSTM(config.embedding_size,
                                config.hidden_size,
                                batch_first=True,
                                bidirectional=config.bidirectional)
        # 3.inference Composition layer
        self.composition = nn.LSTM(config.hidden_size * 2 * 4,
                                   config.hidden_size,
                                   batch_first=True,
                                   bidirectional=config.bidirectional)
        # 4.multi-classification layer
        self.mlp = nn.Sequential(
            nn.Dropout(config.drop_out),
            nn.Linear(config.hidden_size * 2 * 4, config.linear_size),
            nn.Tanh(),
            nn.Linear(config.linear_size, config.label_num)
        )

    def soft_align(self, p_bar, h_bar, mask_p, mask_h):
        '''
        Args:
            p_bar:  batch * len_p * (hidden_size*2)
            mask_p: batch * len_p
            h_bar:  batch * len_h * (hidden_size*2)
            mask_h: batch * len_h
        return:
            align_p: batch * len_p * (hidden_size*2)
            align_h: batch * len_h * (hidden_size*2)
        '''
        # Locality of inference: calucate the attention weight, equation(11)
        attention_weigth = torch.matmul(p_bar, h_bar.transpose(1, 2))  # batch * len_p * len_h

        # Local inference collected over sequences, equation(12-13)
        mask_p = mask_p.float().masked_fill_(mask_p, float('-inf'))  # batch * len_p
        mask_h = mask_h.float().masked_fill_(mask_h, float('-inf'))  # batch * len_h

        # broadcast of mask_p and mask_h to mask attention_p and attention_h for softmax
        weight_p = F.softmax(attention_weigth + mask_h.unsqueeze(1), dim=-1)  # batch * len_p * len_h
        weight_h = F.softmax(attention_weigth.transpose(1, 2) + mask_p.unsqueeze(1), dim=-1)  # batch * len_h * len_p

        align_p = torch.matmul(weight_p, h_bar)  # batch * len_p * (hidden_size*2)
        align_h = torch.matmul(weight_h, p_bar)  # batch * len_h * (hidden_size*2)
        return align_p, align_h

    def pooling(self, p, h):
        '''
        MASK?!
        Args:
            p: batch * len_p * (hidden_size*2)
            h: batch * len_h * (hidden_size*2)
        return:
            v: batch * (hidden_size*2*4)
        '''
        avg_p = F.avg_pool1d(p.transpose(1, 2), p.shape[1]).squeeze(-1)  # batch * (hidden_size*2)
        max_p = F.max_pool1d(p.transpose(1, 2), p.shape[1]).squeeze(-1)  # batch * (hidden_size*2)

        avg_h = F.avg_pool1d(h.transpose(1, 2), h.shape[1]).squeeze(-1)  # batch * (hidden_size*2)
        max_h = F.max_pool1d(h.transpose(1, 2), h.shape[1]).squeeze(-1)  # batch * (hidden_size*2)

        return torch.cat([avg_p, max_p, avg_h, max_h], -1)  # batch * (hidden_size*2*4)

    def forward(self, x):
        p, h = x
        # 1.embedding
        embedding_p = self.embedding(p)  # batch * len_p * embedding_size
        embedding_h = self.embedding(h)  # batch * len_h * embedding_size

        # 2.input encoding
        p_bar, _ = self.encoding(embedding_p)  # batch * len_p * (embedding_size*2)
        h_bar, _ = self.encoding(embedding_h)  # batch * len_h * (embedding_size*2)

        # 3.inference Composition - soft align and enhancement
        mask_p = p.eq(self.embedding.padding_idx)  # batch * len_p
        mask_h = h.eq(self.embedding.padding_idx)  # batch * len_h
        align_p, align_h = self.soft_align(p_bar, h_bar, mask_p, mask_h)
        enhanced_p = torch.cat([p_bar, align_p, p_bar - align_p, p_bar * align_p],
                               -1)  # batch * len_p * (hidden_size*2*4)
        enhanced_h = torch.cat([h_bar, align_h, h_bar - align_h, h_bar * align_h],
                               -1)  # batch * len_h * (hidden_size*2*4)

        # 3.inference Composition - lstm
        composition_p, _ = self.composition(enhanced_p)  # batch * len_p * (hidden_size*2)
        composition_h, _ = self.composition(enhanced_h)  # batch * len_h * (hidden_size*2)

        # 4.pooling
        v = self.pooling(composition_p, composition_h)  # batch * (hidden_size*2*4)

        # 5.MLP
        logit = self.mlp(v)
        probablity = F.softmax(logit, dim=1)

        return probablity


def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0.0)
            else:
                pass