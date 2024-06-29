import os
import argparse

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
    return train_loss/len(trainloader), np.mean(accs)


def validate_model(net, criterion, valid_loader):
    valid_loss = 0.0
    net.eval()
    accs = []
    for data, target in valid_loader:
        data, target = data.to(device), target.to(device)
        output = net(data)
        loss = criterion(output, target)
        valid_loss += loss.item()*data.size(0)
        accs.append(metrics.acc(output.detach(), target))
    return valid_loss/len(validloader), np.mean(accs)


def run(dataset, net_type):

    # Hyper Parameter settings
    n_epochs = cfg.n_epochs
    lr = cfg.lr
    num_workers = cfg.num_workers
    valid_size = cfg.valid_size
    batch_size = cfg.batch_size

    trainset, testset, inputs, outputs = data.getDataset(dataset)
    train_loader, valid_loader, test_loader = data.getDataloader(
        trainset, testset, valid_size, batch_size, num_workers)
    net = AlexNet(outputs, inputs).to(device)

    ckpt_dir = f'checkpoints/{dataset}/frequentist'
    ckpt_name = f'checkpoints/{dataset}/frequentist/model_{net_type}.pt'

    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir, exist_ok=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=lr)
    lr_sched = lr_scheduler.ReduceLROnPlateau(optimizer, patience=6, verbose=True)
    valid_loss_min = np.Inf
    for epoch in range(1, n_epochs+1):

        train_loss, train_acc = train_model(net, optimizer, criterion, train_loader)
        valid_loss, valid_acc = validate_model(net, criterion, valid_loader)
        lr_sched.step(valid_loss)

        train_loss = train_loss/len(train_loader.dataset)
        valid_loss = valid_loss/len(valid_loader.dataset)
            
        print('Epoch: {} \tTraining Loss: {:.4f} \tTraining Accuracy: {:.4f} \tValidation Loss: {:.4f} \tValidation Accuracy: {:.4f}'.format(
            epoch, train_loss, train_acc, valid_loss, valid_acc))
        
        # save model if validation loss has decreased
        if valid_loss <= valid_loss_min:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                valid_loss_min, valid_loss))
            th.save(net.state_dict(), ckpt_name)
            valid_loss_min = valid_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Pyth Frequentist Model Training")
    parser.add_argument('--net_type', default='lenet', type=str, help='model')
    parser.add_argument('--dataset', default='MNIST', type=str, help='dataset = [MNIST/CIFAR10/CIFAR100]')
    args = parser.parse_args()

    run(args.dataset, args.net_type)
