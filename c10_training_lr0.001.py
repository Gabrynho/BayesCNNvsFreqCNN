import numpy as np
from tqdm import tqdm
import os
import csv
import itertools
import torch as th
from torch.optim import Adam, lr_scheduler

from utils import data, metrics
import Bayesian_main as BCNN
from Bayesian.BayesianCNN import BBBAlexNet

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
lr_start =  0.0005
batch_size = [256, 512]
hp_tuning = list(itertools.product(lr_start, batch_size))
valid_size = 0.2
beta_type = "Blundell"

# Dataset and Dataloader
c10_trainset, c10_testset, c10_inputs, c10_outputs = data.getDataset('CIFAR10')

# BayesianCNN with softplus on CIFAR10 hyperparametrization tuning
for lr, batch in hp_tuning:
    filename = f"results_c10_lr{lr}_batch{batch}.csv"

    c10_train_loader, c10_valid_loader, _ = data.getDataloader(c10_trainset, c10_testset, valid_size, batch)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Train KL Div'])

        bc10_net = BBBAlexNet(c10_outputs, c10_inputs, priors, activation_type='softplus').to(device)
        bc10_criterion = metrics.ELBO(len(c10_trainset)).to(device)
        bc10_optimizer = Adam(bc10_net.parameters(), lr=lr)
        bc10_lr_sched = lr_scheduler.ReduceLROnPlateau(bc10_optimizer, patience=6, verbose=True)
        bc10_valid_loss_max = np.Inf

        ckpt_name = f"Bayesian/Models/bc10_lr{lr}_batch{batch}.pth"
        if os.path.isfile(ckpt_name):
            checkpoint = th.load(ckpt_name)
            bc10_net.load_state_dict(checkpoint['model_state_dict'])
            bc10_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            bc10_lr_sched.load_state_dict(checkpoint['scheduler_state_dict'])
            bc10_valid_loss_max = checkpoint['valid_loss_max']
            start_epoch = checkpoint['epoch'] + 1
            print('Model loaded from {}'.format(ckpt_name))
        else:
            start_epoch = 0

        for epoch in tqdm(range(start_epoch, n_epochs)):  # loop over the dataset multiple times
            bc10_train_loss, bc10_train_acc, bc10_train_kl = BCNN.train_model(
                bc10_net, bc10_optimizer, bc10_criterion, c10_train_loader, num_ens=1,
                beta_type=beta_type, epoch=epoch, num_epochs=n_epochs
            )
            bc10_valid_loss, bc10_valid_acc = BCNN.validate_model(
                bc10_net, bc10_criterion, c10_valid_loader, num_ens=1,
                beta_type=beta_type, epoch=epoch, num_epochs=n_epochs
            )
            bc10_lr_sched.step(bc10_valid_loss)

            # save model if validation accuracy has increased
            if bc10_valid_loss <= bc10_valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    bc10_valid_loss_max, bc10_valid_loss))
                th.save({
                    'model_state_dict': bc10_net.state_dict(),
                    'optimizer_state_dict': bc10_optimizer.state_dict(),
                    'scheduler_state_dict': bc10_lr_sched.state_dict(),
                    'valid_loss_max': bc10_valid_loss,
                    'epoch': epoch
                }, ckpt_name)
                bc10_valid_loss_max = bc10_valid_loss

            writer.writerow([epoch, bc10_train_loss, bc10_train_acc, bc10_valid_loss, bc10_valid_acc, bc10_train_kl])