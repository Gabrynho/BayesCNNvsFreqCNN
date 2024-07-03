import numpy as np
import copy
import torch as th
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def getDataset(dataset):

    transform_cifar = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        ])

    if(dataset == 'CIFAR10'):
        trainset = torchvision.datasets.CIFAR10(root='./utils', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR10(root='./utils', train=False, download=True, transform=transform_cifar)
        num_classes = 10
        inputs = 3

    elif(dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./utils', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR100(root='./utils', train=False, download=True, transform=transform_cifar)
        num_classes = 100
        inputs = 3

    return trainset, testset, inputs, num_classes

def getDataloader(trainset, testset, valid_size, batch_size):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, pin_memory=True)
    valid_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, pin_memory=True)
    test_loader = th.utils.data.DataLoader(testset, batch_size=batch_size, 
        pin_memory=True)

    return train_loader, valid_loader, test_loader


def getDataloader_mislabel(trainset, testset, valid_size, batch_size, mislabel_percentage):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    # Create a deep copy of the trainset for mislabeling
    mislabeled_trainset = copy.deepcopy(trainset)

    # Mislabel a custom percentage of labels in the trainset
    #mislabel_count = int(mislabel_percentage * len(train_idx))
    mislabel_count = int(mislabel_percentage * num_train)
    mislabel_indices = np.random.choice(train_idx, size=mislabel_count, replace=False)
    for idx in mislabel_indices:
        original_label = mislabeled_trainset[idx][1]
        possible_new_labels = [i for i in range(len(set(mislabeled_trainset.targets))) if i != original_label]
        new_label = np.random.choice(possible_new_labels)
        mislabeled_trainset.targets[idx] = new_label

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader_mislabel = th.utils.data.DataLoader(mislabeled_trainset, batch_size=batch_size,
        sampler=train_sampler, pin_memory=True)
    valid_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, pin_memory=True)
    test_loader = th.utils.data.DataLoader(testset, batch_size=batch_size, 
        pin_memory=True)
    train_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, pin_memory=True)

    return train_loader_mislabel, valid_loader, test_loader, train_loader


def mislabel_data(trainset, mislabel_ratio=0.1):
    # Get the total number of samples in the dataset
    num_samples = len(trainset)
    
    # Calculate the number of samples to mislabel
    num_mislabel = int(num_samples * mislabel_ratio)
    
    # Randomly select samples to mislabel
    mislabel_indices = np.random.choice(num_samples, num_mislabel, replace=False)
    
    for idx in mislabel_indices:
        # Get the current label
        current_label = trainset.targets[idx]
        
        # Choose a new label that is different from the current one
        new_label = np.random.choice([i for i in range(trainset.num_classes) if i != current_label])
        
        # Mislabel the data
        trainset.targets[idx] = new_label
    
    return trainset