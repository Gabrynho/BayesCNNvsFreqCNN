import numpy as np
from tqdm import tqdm
import os
import csv
import itertools
import torch as th
from torch.optim import Adam, lr_scheduler

from utils import data, metrics
import Frequentist_main as FCNN
import Bayesian_main as BCNN
from Bayesian.BayesianCNN import BBBAlexNet
from Frequentist.FrequentistCNN import AlexNet

# Set the device
device = th.device("cuda" if th.cuda.is_available() else "cpu")
print(device)

# Set the parameters
priors = {
    'prior_mu': 0,
    'prior_sigma': 0.1,
    'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
    'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
}

n_epochs = 100
valid_size = 0.2
beta_type = "Blundell"

# Best hyperparameter
lr_start_c10 = 0.001
#lr_start_c100 = 0.0005
batch_size = 128

# Dataset and Dataloader
c10_trainset, c10_testset, c10_inputs, c10_outputs = data.getDataset('CIFAR10')
c10_train_loader, c10_valid_loader, c10_test_loader = data.getDataloader(
    c10_trainset, c10_testset, valid_size, batch_size)

# FrequentistCNN on CIFAR10
filename = f"results_f10_lr{lr_start_c10}_batch{batch_size}.csv"

with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Train KL Div'])

        fc10_net = AlexNet(c10_outputs, c10_inputs).to(device)
        fc10_criterion = nn.CrossEntropyLoss()
        fc10_optimizer = Adam(fc10_net.parameters(), lr=lr_start_c10)
        fc10_lr_sched = lr_scheduler.ReduceLROnPlateau(fc10_optimizer, patience=6, verbose=True)
        fc10_valid_loss_max = np.Inf
        
        ckpt_name = 'Frequentist/Models/fc10.pth'
        if os.path.isfile(ckpt_name):
            checkpoint = th.load(ckpt_name)
            fc10_net.load_state_dict(checkpoint['model_state_dict'])
            fc10_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            fc10_lr_sched.load_state_dict(checkpoint['scheduler_state_dict'])
            fc10_valid_loss_max = checkpoint['valid_loss_max']
            start_epoch = checkpoint['epoch'] + 1  
            print('Model loaded from {}'.format(ckpt_name))
        else:
            start_epoch = 0
        
        for epoch in tqdm(range(start_epoch, n_epochs)):
        
            fc10_train_loss, fc10_train_acc = FCNN.train_model(fc10_net, fc10_optimizer, fc10_criterion, c10_train_loader)
            fc10_valid_loss, fc10_valid_acc = FCNN.validate_model(fc10_net, fc10_criterion, c10_valid_loader)
            fc10_lr_sched.step(fc10_valid_loss)
        
            # save model if validation accuracy has increased
            if fc10_valid_loss <= fc10_valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    fc10_valid_loss_max, fc10_valid_loss))
                th.save({
                    'model_state_dict': fc10_net.state_dict(),
                    'optimizer_state_dict': fc10_optimizer.state_dict(),
                    'scheduler_state_dict': fc10_lr_sched.state_dict(),
                    'valid_loss_max': fc10_valid_loss,
                    'epoch': epoch
                }, ckpt_name)
                fc10_valid_loss_max = fc10_valid_loss

            writer.writerow([epoch, fc10_train_loss, fc10_train_acc, fc10_valid_loss, fc10_valid_acc])