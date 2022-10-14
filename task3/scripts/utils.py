#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File      : utils.py
# Author    : Rui-Zhao
# Date      : 2022/10/13 19:14 
import pandas as pd
import os
import time
from datetime import timedelta
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


UNK, PAD, SEP = '<UNK>', '<PAD>', '<SEP>'  # 未知字，padding符号

def join(path):
    '''
    hide the home path
    '''
    home_path = os.path.expanduser('~')
    path = os.path.join(home_path, path)
    return path

def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))

def SNLItokenizer(sentence, pad_size):
    tokenizer = lambda x: x.split()  # word_level tokenzier

    def pad_and_trunc(tokens):
        if len(tokens) < pad_size:
            tokens.extend([PAD] * (pad_size - len(tokens)))
        else:
            tokens = tokens[:pad_size]
        return tokens

    tokens = tokenizer(sentence)
    tokens = pad_and_trunc(tokens)
    return tokens


class SNLIDataset(Dataset):

    def __init__(self, hypothesis, premise, labels, config):
        self.hypothesis = hypothesis
        self.premise = premise
        self.pad_size_p = config.pad_size_p
        self.pad_size_h = config.pad_size_h
        self.labels = labels
        self.device = config.device
        self.vocab_map = config.vocab_map
        self.label_map = config.label_map

    def __getitem__(self, index):
        # padding and truncation
        tokens_p = SNLItokenizer(self.premise[index], self.pad_size_p)
        tokens_h = SNLItokenizer(self.hypothesis[index], self.pad_size_h)
        ind_p = [self.vocab_map.get(token, self.vocab_map.get(UNK)) for token in tokens_p]
        ind_h = [self.vocab_map.get(token, self.vocab_map.get(UNK)) for token in tokens_h]
        p = torch.LongTensor(ind_p).to(self.device)
        h = torch.LongTensor(ind_h).to(self.device)
        y = torch.LongTensor([self.label_map.get(self.labels[index])]).to(self.device)
        return (p, h), y

    def __len__(self):
        return len(self.labels)


def build_dataset(config):
    def load_dataset(path):
        df = pd.read_json(path, lines=True)
        df.drop(df.loc[df['gold_label'] == '-'].index, inplace=True)
        data = SNLIDataset(df['sentence1'].tolist(), df['sentence2'].tolist(), df['gold_label'].tolist(), config)
        print("{0} loaded succcess, {1} in total.".format(os.path.split(path)[-1], len(data)))
        return data

    start_time = time.time()
    print('loading data...')
    train_set = load_dataset(config.train_path)
    dev_set = load_dataset(config.dev_path)
    test_set = load_dataset(config.test_path)
    time_used = get_time_dif(start_time)
    print('Time used:', time_used)
    return train_set, dev_set, test_set

# tensor_dataloader = DataLoader(tensor_dataset,     # 封装的对象
#                                batch_size = 2,     # 输出的batch size
#                                shuffle = True,     # 随机输出
#                                num_workers = 0)    # 只有1个进程