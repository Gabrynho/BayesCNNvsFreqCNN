import numpy as np
import torch as th
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

class CustomDataset(Dataset):
    def __init__(self, data, labels, transform=None):
        self.data = data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        sample = self.data[idx]
        label = self.labels[idx]
        if self.transform:
            sample = self.transform(sample)

        return sample, label


def extract_classes(dataset, classes):
    idx = th.zeros_like(dataset.targets, dtype=th.bool)
    for target in classes:
        idx = idx | (dataset.targets==target)

    data, targets = dataset.data[idx], dataset.targets[idx]
    return data, targets


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
        inputs=3

    elif(dataset == 'CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./utils', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR100(root='./utils', train=False, download=True, transform=transform_cifar)
        num_classes = 100
        inputs = 3
# entrambi da modifiare per il flipper
    elif(dataset == "MIXED_CIFAR10"):
        trainset = torchvision.datasets.CIFAR10(root='./utils', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR10(root='./utils', train=False, download=True, transform=transform_cifar)
        num_classes = 10
        inputs=3
        
    elif(dataset == 'MIXED_CIFAR100'):
        trainset = torchvision.datasets.CIFAR100(root='./utils', train=True, download=True, transform=transform_cifar)
        testset = torchvision.datasets.CIFAR100(root='./utils', train=False, download=True, transform=transform_cifar)
        num_classes = 100
        inputs = 3
        # Qui vanno messe cose che servono a fare il mix delle classi su una porzione del dataset di training


def getDataloader(trainset, testset, valid_size, batch_size, num_workers):
    num_train = len(trainset)
    indices = list(range(num_train))
    np.random.shuffle(indices)
    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = indices[split:], indices[:split]

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size,
        sampler=train_sampler, num_workers=num_workers)
    valid_loader = th.utils.data.DataLoader(trainset, batch_size=batch_size, 
        sampler=valid_sampler, num_workers=num_workers)
    test_loader = th.utils.data.DataLoader(testset, batch_size=batch_size, 
        num_workers=num_workers)

    return train_loader, valid_loader, test_loader