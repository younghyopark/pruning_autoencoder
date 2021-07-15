'''Dataset Class'''
import os
import torch
from torchvision.datasets.mnist import MNIST


class MNISTLeaveOut(MNIST):
    """
    MNIST Dataset with some digits excluded.
    
    targets will be 1 for excluded digits (outlier) and 0 for included digits.
    
    See also the original MNIST class: 
        https://pytorch.org/docs/stable/_modules/torchvision/datasets/mnist.html#MNIST
    """
    img_size = (28, 28)

    def __init__(self, root, l_out_class, split='training', transform=None, target_transform=None,
                 download=False):
        """
        l_out_class : a list of ints. these clases are excluded in training
        """
        super(MNISTLeaveOut, self).__init__(root, transform=transform,
                                            target_transform=target_transform, download=download)
        if split == 'training' or split == 'validation':
            self.train = True  # training set or test set
        else:
            self.train = False
        self.split = split
        self.l_out_class = list(l_out_class)
        for c in l_out_class:
            assert c in set(list(range(10)))
        set_out_class = set(l_out_class)

        if download:
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if self.train:
            data_file = self.training_file
        else:
            data_file = self.test_file

        data, targets = torch.load(os.path.join(self.processed_folder, data_file))

        if split == 'training':
            data = data[:54000]
            targets = targets[:54000]
        elif split == 'validation':
            data = data[54000:]
            targets = targets[54000:]

        out_idx = torch.zeros(len(data), dtype=torch.bool)  # pytorch 1.2
        for c in l_out_class:
            out_idx = out_idx | (targets == c)

        # if self.train:
        # if split == 'training':
        self.data = data[~out_idx]
        self.digits = targets[~out_idx]
        # self.targets = torch.ones_like(self.digits)
        self.targets = self.digits
        # else:
        #     self.data = data
        #     self.digits = targets
        #     self.targets = out_idx.long()

    @property
    def raw_folder(self):
        return os.path.join(self.root, 'MNIST', 'raw')

    @property
    def processed_folder(self):
        return os.path.join(self.root, 'MNIST', 'processed')


