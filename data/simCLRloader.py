from __future__ import print_function
import torch
import torch.utils.data as data
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np


class ContrastiveTransformations(object):
    def __init__(self, base_transforms, n_views=2):
        self.base_transforms = base_transforms
        self.n_views = n_views

    def __call__(self, x):
        return [self.base_transforms(x) for i in range(self.n_views)]


class SimCLRDataset(data.Dataset):
    def __init__(self, dataset_name, split, dataset_root=None):
        self.split = split.lower()
        self.dataset_name = dataset_name.lower()
        self.name = self.dataset_name + '_' + self.split

        if self.dataset_name == 'cifar10':
            self.mean_pix = [x / 255.0 for x in [125.3, 123.0, 113.9]]
            self.std_pix = [x / 255.0 for x in [63.0, 62.1, 66.7]]

            if split != 'test':
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.1)
                    ], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomRotation(5),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)
                ])

            self.transform = ContrastiveTransformations(self.transform, n_views=2)

            self.data = datasets.__dict__[self.dataset_name.upper()](
                root=dataset_root, train=self.split == 'train',
                download=True, transform=self.transform)

        elif self.dataset_name == 'mnist' or self.dataset_name == 'svhn':
            self.mean_pix = (0.5,)  # Grayscale mean
            self.std_pix = (0.5,)   # Grayscale std

            if split != 'test':
                self.transform = transforms.Compose([
                    transforms.Resize(32),  # Resize to 32x32 for compatibility with CIFAR
                    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
                    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
                    transforms.RandomApply([
                        transforms.ColorJitter(brightness=0.5, contrast=0.5)
                    ], p=0.8),
                    transforms.RandomRotation(10),
                    transforms.GaussianBlur(kernel_size=9),
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)  # Apply grayscale normalization
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize(32),  # Resize to 32x32 for compatibility with CIFAR
                    transforms.Grayscale(num_output_channels=1),    # Convert to grayscale
                    transforms.ToTensor(),
                    transforms.Normalize(self.mean_pix, self.std_pix)  # Apply grayscale normalization
                ])

            self.transform = ContrastiveTransformations(self.transform, n_views=2)

            if self.dataset_name == 'mnist':
                self.data = datasets.MNIST(
                    root=dataset_root, train=self.split == 'train',
                    download=True, transform=self.transform)
            elif self.dataset_name == 'svhn':
                self.data = datasets.SVHN(
                    root=dataset_root, split=self.split,
                    download=True, transform=self.transform)
        else:
            raise ValueError(f'Not recognized dataset {dataset_name}')

    def __getitem__(self, index):
        imgs, label = self.data[index]  # n views of images and label
        return imgs, int(label)

    def __len__(self):
        return len(self.data)
