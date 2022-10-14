#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File      : train_eval.py
# Author    : Rui-Zhao
# Date      : 2022/10/13 19:25
import numpy as np
import time
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader
from utils import get_time_dif


def train(model, config, train_set, dev_set, test_set):
    train_iter = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    dev_iter = DataLoader(dev_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    test_iter = DataLoader(test_set, batch_size=config.batch_size, shuffle=True, num_workers=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    model.train()
    start_time = time.time()

    loss_dev_best = float('inf')
    step_total = 0
    step_last_improve = 0
    best_dev_acc = 0
    flag_early_stop = False
    for epoch in range(config.epoch):
        print('Epoch [{}/{}]'.format(epoch + 1, config.epoch))
        for i, (x, y_true) in enumerate(train_iter):
            y_true = y_true.reshape(-1)
            out = model(x)
            model.zero_grad()
            loss = F.cross_entropy(out, y_true)
            loss.backward()
            optimizer.step()
            step_total += 1
            if step_total % config.log_step == 0 or i + 1 == len(train_iter):
                #                 acc_train, loss_train = evalute(model, config, DataLoader(train_set, config.batch_size, True)) # 在该batch还是整个train set?
                predic = torch.max(out.data, 1)[1].cpu()
                acc_train, loss_train = metrics.accuracy_score(y_true.cpu(), predic), loss.item()

                acc_dev, loss_dev = evaluate(model, config, DataLoader(dev_set, config.batch_size, True))
                model.train()
                improve = ""
                if loss_dev < loss_dev_best:
                    loss_dev_best = loss_dev
                    improve = "*"
                    step_last_improve = step_total
                    torch.save(model.state_dict(), config.save_path)  # save the best model on dev set
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5}{6}'
                print(msg.format(step_total, loss_train, acc_train, loss_dev, acc_dev, time_dif, improve))
            if step_total - step_last_improve > config.require_improvement and config.early_stop:
                # 验证集loss超过config.require_improvement个batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag_early_stop = True
                break
        if flag_early_stop and config.early_stop:
            break
    test(model, config, test_iter)


def evaluate(model, config, data_iter, test=False):
    model.eval()
    loss_total = 0
    labels_predict = np.arange(0)
    labels_true = np.arange(0)
    with torch.no_grad():
        for x, y_true in data_iter:
            y_true = y_true.reshape(-1)
            out = model(x)
            loss = F.cross_entropy(out, y_true)
            loss_total += loss
            labels_predict = np.concatenate((labels_predict, out.data.cpu().numpy().argmax(axis=1)), axis=0)
            labels_true = np.concatenate((labels_true, y_true.data.cpu().numpy()), axis=0)
        try:
            assert len(labels_true) == labels_predict.size
        except AssertionError as e:
            print(len(x))
            print(out.shape)
            print("labels_true:", len(labels_true))
            print("labels_predict:", labels_predict.size)
            raise AssertionError
    #     acc = (labels_true == labels_predict).mean()
    acc = metrics.accuracy_score(labels_true, labels_predict)
    if test:
        report = metrics.classification_report(labels_true, labels_predict, target_names=config.label_list, digits=4)
        confusion = metrics.confusion_matrix(labels_true, labels_predict)
        return acc, loss_total.data.cpu().numpy() / len(data_iter), report, confusion
    return acc, loss_total.data.cpu().numpy() / len(data_iter)


def test(model, config, data_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    test_acc, test_loss, test_report, test_confusion = evaluate(model, config, data_iter, test=True)
    print("Confusion Matrix...")
    print(test_confusion)
    print("Precision, Recall and F1-Score...")
    print(test_report)
    msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
    print(msg.format(test_loss, test_acc))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)