from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
from .utils import download_url, check_integrity
from .utils import TransformTwice, TransformKtimes, RandomTranslateWithReflect, TwoStreamBatchSampler 
from .concat import ConcatDataset 
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

class SVHN(data.Dataset):
    """`SVHN <http://ufldl.stanford.edu/housenumbers/>`_ Dataset.
    Note: The SVHN dataset assigns the label `10` to the digit `0`. However, in this Dataset,
    we assign the label `0` to the digit `0` to be compatible with PyTorch loss functions which
    expect the class labels to be in the range `[0, C-1]`

    Args:
        root (string): Root directory of dataset where directory
            ``SVHN`` exists.
        split (string): One of {'train', 'test', 'extra'}.
            Accordingly dataset is selected. 'extra' is Extra training set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.

    """
    url = ""
    filename = ""
    file_md5 = ""

    split_list = {
        'train': ["http://ufldl.stanford.edu/housenumbers/train_32x32.mat",
                  "train_32x32.mat", "e26dedcc434d2e4c54c9b2d4a06d8373"],
        'test': ["http://ufldl.stanford.edu/housenumbers/test_32x32.mat",
                 "test_32x32.mat", "eb5a983be6a315427106f1b164d9cef3"],
        'extra': ["http://ufldl.stanford.edu/housenumbers/extra_32x32.mat",
                  "extra_32x32.mat", "a93ce644f1a588dc4d68dda5feec44a7"]}

    def __init__(self, root, split='train',
                 transform=None, target_transform=None, download=True, target_list=range(5)):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.split = split  # training set or test set or extra set

        if self.split not in self.split_list:
            raise ValueError('Wrong split entered! Please use split="train" '
                             'or split="extra" or split="test"')

        self.url = self.split_list[split][0]
        self.filename = self.split_list[split][1]
        self.file_md5 = self.split_list[split][2]

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You can use download=True to download it')

        # import here rather than at top of file because this is
        # an optional dependency for torchvision
        import scipy.io as sio

        # reading(loading) mat file as array
        loaded_mat = sio.loadmat(os.path.join(self.root, self.filename))

        self.data = loaded_mat['X']
        # loading from the .mat file gives an np array of type np.uint8
        # converting to np.int64, so that we have a LongTensor after
        # the conversion from the numpy array
        # the squeeze is needed to obtain a 1D tensor
        self.labels = loaded_mat['y'].astype(np.int64).squeeze()

        # the svhn dataset assigns the class label "10" to the digit 0
        # this makes it inconsistent with several loss functions
        # which expect the class labels to be in the range [0, C-1]
        np.place(self.labels, self.labels == 10, 0)
        self.data = np.transpose(self.data, (3, 2, 0, 1))

        ind = [i for i in range(len(self.labels)) if int(self.labels[i]) in target_list]
        
        self.data = self.data[ind]
        self.labels= self.labels[ind]

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], int(self.labels[index])

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index

    def __len__(self):
        return len(self.data)

    def _check_integrity(self):
        root = self.root
        md5 = self.split_list[self.split][2]
        fpath = os.path.join(root, self.filename)
        return check_integrity(fpath, md5)

    def download(self):
        md5 = self.split_list[self.split][2]
        download_url(self.url, self.root, self.filename, md5)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Split: {}\n'.format(self.split)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

def SVHNData(root, split='train', aug=None, target_list=range(5)):
    if aug == None:
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize for 1 channel
        ])
    elif aug == 'once':
        transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize for 1 channel
        ])
    elif aug == 'twice':
        transform = TransformTwice(transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # Normalize for 1 channel
        ]))

    dataset = SVHN(root=root, split=split, transform=transform, target_list=target_list)
    return dataset


def SVHNLoader(root, batch_size, split='train',  num_workers=2, aug=None, shuffle=True, target_list=range(5)):
    dataset = SVHNData(root, split, aug,target_list)
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def SVHNLoaderMix(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10)):
    dataset_labeled = SVHNData(root, split, aug, labeled_list)
    dataset_unlabeled = SVHNData(root, split, aug, unlabeled_list)
    dataset_labeled.labels = np.concatenate((dataset_labeled.labels,dataset_unlabeled.labels))
    dataset_labeled.data = np.concatenate((dataset_labeled.data,dataset_unlabeled.data),0)
    loader = data.DataLoader(dataset_labeled, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return loader

def SVHNLoaderTwoStream(root, batch_size, split='train',num_workers=2, aug=None, shuffle=True, labeled_list=range(5), unlabeled_list=range(5, 10),  unlabeled_batch_size=64):
    dataset_labeled = SVHNData(root, split, aug, labeled_list)
    dataset_unlabeled = SVHNData(root, split, aug, unlabeled_list)
    dataset = ConcatDataset((dataset_labeled, dataset_unlabeled)) 
    labeled_idxs = range(len(dataset_labeled)) 
    unlabeled_idxs = range(len(dataset_labeled), len(dataset_labeled)+len(dataset_unlabeled))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, unlabeled_batch_size) 
    loader = data.DataLoader(dataset, batch_sampler=batch_sampler, num_workers=num_workers)
    loader.labeled_length = len(dataset_labeled)
    loader.unlabeled_length = len(dataset_unlabeled)
    return loader


