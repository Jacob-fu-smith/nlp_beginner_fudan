#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File      : run.py
# Author    : Rui-Zhao
# Date      : 2022/10/13 19:27
import time
from datetime import timedelta, datetime
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train_eval import train, test
from model import Config, ESIM, init_network
from utils import build_dataset
import argparse

parser = argparse.ArgumentParser(description='ESIM for SNLI')
parser.add_argument('--learning_rate', type=float, required=False, help='learning rate')
parser.add_argument('--batch_size', type=int, required=False, help='batch size')
parser.add_argument('--epoch', type=int, required=False, help='epoch')
parser.add_argument('--output_dir', type=str, required=False, help='output directory')
parser.add_argument('--early_stop', type=bool, required=False, help='early stop')
parser.add_argument('--seed', type=int, required=False, help='random seed')
args = parser.parse_args()

def over_fitting_batch():
    config = Config()
    model = ESIM(config).to(config.device)
    train_set, dev_set, test_set = build_dataset(config)
    train_iter = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    first_batch = next(iter(train_iter))
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    for i, (x, y_true) in enumerate([first_batch] * 50):
        out = model(x)
        model.zero_grad()
        loss = F.cross_entropy(out, y_true.reshape(-1))
        print(loss)
        loss.backward()
        optimizer.step()


def main():
    config = Config()
    # hyper-parameters
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.early_stop = args.early_stop

    train_set, dev_set, test_set = build_dataset(config)
    model = ESIM(config).to(config.device)
    print(model)
    ### customized hyper param
    config.log_step = 1000

    ###
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    init_network(model)
    train(model, config, train_set, dev_set, test_set)


if __name__ == '__main__':
    print("start time:", (datetime.utcnow() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))
    # main()
    config = Config()
    model = ESIM(config).to(config.device)
    train_set, dev_set, test_set = build_dataset(config)
    test_set = DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test(model, config, test_set)
    print("end time:", (datetime.utcnow() + timedelta(hours=8)).strftime('%Y-%m-%d %H:%M:%S'))

