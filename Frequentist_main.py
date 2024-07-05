import os
import torch as th
import numpy as np
import torch.nn as nn
from torch.optim import Adam, lr_scheduler

from utils import data, metrics
#import config_bayesian as cfg
from Frequentist.FrequentistCNN import AlexNet

# CUDA settings
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def train_model(net, optimizer, criterion, train_loader):
    train_loss = 0.0
    net.train()
    accs = []
    for data, target in train_loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return train_loss/len(train_loader), np.mean(accs)


def validate_model(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    accs = []

    with th.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            loss = criterion(output, target)
            valid_loss += loss.item()*data.size(0)
            accs.append(metrics.acc(output.detach(), target))
    return valid_loss/len(valid_loader), np.mean(accs)