import torch
from torchvision import transforms
from torch.utils import data
from torchvision.datasets import MNIST
from PIL import Image
import os
import numpy as np
import pickle
import sys


class CustomMNIST(data.Dataset):
    def __init__(self, root, split='train+test', transform=None, target_transform=None, download=True, target_list=range(10)):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform

        # Load MNIST dataset
        if split == 'train':
            mnist_dataset = MNIST(root=self.root, train=True, transform=None, download=download)
        elif split == 'test':
            mnist_dataset = MNIST(root=self.root, train=False, transform=None, download=download)
        elif split == 'train+test':
            mnist_train = MNIST(root=self.root, train=True, transform=None, download=download)
            mnist_test = MNIST(root=self.root, train=False, transform=None, download=download)
            self.data = np.vstack([mnist_train.data.numpy(), mnist_test.data.numpy()])
            self.targets = np.concatenate([mnist_train.targets.numpy(), mnist_test.targets.numpy()])
        else:
            raise ValueError("split must be either 'train', 'test' or 'train+test'")

        # Load only selected classes
        if split != 'train+test':
            self.data = mnist_dataset.data.numpy()
            self.targets = mnist_dataset.targets.numpy()

        ind = [i for i in range(len(self.targets)) if self.targets[i] in target_list]
        self.data = self.data[ind]
        self.targets = np.array(self.targets)[ind].tolist()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        # img = Image.fromarray(np.uint8(np.transpose(img, (1, 2, 0))))  # Convert to PIL Image (H, W, C)
        img = Image.fromarray(np.uint8(img), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)


def CustomMNISTData(root, split='train', aug=None, target_list=range(10)):
    if aug == None:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    elif aug == 'twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(28, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
        ]))

    dataset = CustomMNIST(root=root, split=split, transform=transform, target_list=target_list)
    return dataset


def CustomMNISTLoader(root, batch_size, split='train', num_workers=2, aug=None, shuffle=True, target_list=range(10), drop_last=True):
    dataset = CustomMNISTData(root, split, aug, target_list)

    if drop_last:
        num_samples = len(dataset)
        num_batches = num_samples // batch_size
        indices = list(range(num_batches * batch_size))  # Adjust the indices to ensure full batches
        dataset = data.Subset(dataset, indices)

    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)
    return loader
