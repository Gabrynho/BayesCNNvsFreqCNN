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
lr_start =  [0.001]
batch_size = [128 ,256, 512]
hp_tuning = list(itertools.product(lr_start, batch_size))
valid_size = 0.2
beta_type = "Blundell"

# Dataset and Dataloader
c100_trainset, c100_testset, c100_inputs, c100_outputs = data.getDataset('CIFAR100')

# BayesianCNN with softplus on CIFAR100
for lr, batch in hp_tuning:
    filename = f"results_c100_lr{lr}_batch{batch}.csv"

    c100_train_loader, c100_valid_loader, _ = data.getDataloader(c100_trainset, c100_testset, valid_size, batch)

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Training Loss', 'Training Accuracy', 'Validation Loss', 'Validation Accuracy', 'Train KL Div'])

        bc100_net = BBBAlexNet(c100_outputs, c100_inputs, priors, activation_type='softplus').to(device)
        bc100_criterion = metrics.ELBO(len(c100_trainset)).to(device)
        bc100_optimizer = Adam(bc100_net.parameters(), lr=lr)
        bc100_lr_sched = lr_scheduler.ReduceLROnPlateau(bc100_optimizer, patience=6, verbose=True)
        bc100_valid_loss_max = np.Inf

        ckpt_name = f"Bayesian/Models/bc100_lr{lr}_batch{batch}.pth"
        if os.path.isfile(ckpt_name):
            checkpoint = th.load(ckpt_name)
            bc100_net.load_state_dict(checkpoint['model_state_dict'])
            bc100_optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            bc100_lr_sched.load_state_dict(checkpoint['scheduler_state_dict'])
            bc100_valid_loss_max = checkpoint['valid_loss_max']
            start_epoch = checkpoint['epoch'] + 1
            print('Model loaded from {}'.format(ckpt_name))
        else:
            start_epoch = 0

        for epoch in tqdm(range(start_epoch, n_epochs)):  # loop over the dataset multiple times
            bc100_train_loss, bc100_train_acc, bc100_train_kl = BCNN.train_model(
                bc100_net, bc100_optimizer, bc100_criterion, c100_train_loader, num_ens=1,
                beta_type=beta_type, epoch=epoch, num_epochs=n_epochs
            )
            bc100_valid_loss, bc100_valid_acc = BCNN.validate_model(
                bc100_net, bc100_criterion, c100_valid_loader, num_ens=1,
                beta_type=beta_type, epoch=epoch, num_epochs=n_epochs
            )
            bc100_lr_sched.step(bc100_valid_loss)

            # save model if validation accuracy has increased
            if bc100_valid_loss <= bc100_valid_loss_max:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    bc100_valid_loss_max, bc100_valid_loss))
                th.save({
                    'model_state_dict': bc100_net.state_dict(),
                    'optimizer_state_dict': bc100_optimizer.state_dict(),
                    'scheduler_state_dict': bc100_lr_sched.state_dict(),
                    'valid_loss_max': bc100_valid_loss,
                    'epoch': epoch
                }, ckpt_name)
                bc100_valid_loss_max = bc100_valid_loss

            writer.writerow([epoch, bc100_train_loss, bc100_train_acc, bc100_valid_loss, bc100_valid_acc, bc100_train_kl])