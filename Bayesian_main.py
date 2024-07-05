import os
import torch as th
import numpy as np
from torch.optim import Adam, lr_scheduler
from torch.nn import functional as F

from utils import data, metrics
from Bayesian.BayesianCNN import BBBAlexNet

# CUDA settings
device = th.device("cuda:0" if th.cuda.is_available() else "cpu")

def train_model(net, optimizer, criterion, trainloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    net.train()
    training_loss = 0.0
    accs = []
    kl_list = []
    for i, (inputs, labels) in enumerate(trainloader, 1):

        optimizer.zero_grad()

        inputs, labels = inputs.to(device), labels.to(device)
        outputs = th.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)

        kl = 0.0
        for j in range(num_ens):
            net_out, _kl = net(inputs)
            kl += _kl
            outputs[:, :, j] = F.log_softmax(net_out, dim=1)
        
        kl = kl / num_ens
        kl_list.append(kl.item())
        log_outputs = metrics.logmeanexp(outputs, dim=2)

        beta = metrics.get_beta(i-1, len(trainloader), beta_type, epoch, num_epochs)
        loss = criterion(log_outputs, labels, kl, beta)
        loss.backward()
        optimizer.step()

        accs.append(metrics.acc(log_outputs.data, labels))
        training_loss += loss.cpu().data.numpy()
    return training_loss/len(trainloader), np.mean(accs), np.mean(kl_list)


def validate_model(net, criterion, validloader, num_ens=1, beta_type=0.1, epoch=None, num_epochs=None):
    """Calculate ensemble accuracy and NLL Loss"""
    net.train()
    valid_loss = 0.0
    accs = []

    with th.no_grad():
        for i, (inputs, labels) in enumerate(validloader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = th.zeros(inputs.shape[0], net.num_classes, num_ens).to(device)
            kl = 0.0
            for j in range(num_ens):
                net_out, _kl = net(inputs)
                kl += _kl
                outputs[:, :, j] = F.log_softmax(net_out, dim=1).data
    
            log_outputs = metrics.logmeanexp(outputs, dim=2)
    
            beta = metrics.get_beta(i-1, len(validloader), beta_type, epoch, num_epochs)
            valid_loss += criterion(log_outputs, labels, kl, beta).item()
            accs.append(metrics.acc(log_outputs, labels))

    return valid_loss/len(validloader), np.mean(accs)