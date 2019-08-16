import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from PIL import Image
from ..utils import *

label_clss = ['NoF', 'DOL', 'LAG', 'BET', 'OTHER', 'SHARK', 'YFT', 'ALB']
clss_to_label = {c:i for i,c in enumerate(label_clss)}
n_classes = len(label_clss)

rgb_mean = [0.485, 0.456, 0.406]
rgb_std = [0.229, 0.224, 0.225]

tsfm_trn = T.Compose([
                    T.Resize((224, 224)),
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                    T.RandomHorizontalFlip(),
                    T.RandomAffine(degrees=10, shear=10),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std),
                ])

tsfm_trn_nof = T.Compose([
                    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.15, hue=0.1),
                    T.RandomResizedCrop(224, scale=(0.1, 1.0)),
                    T.RandomHorizontalFlip(),
                    T.RandomAffine(degrees=10, shear=10),
                    T.ToTensor(),
                    T.Normalize(rgb_mean, rgb_std),
                ])

tsfm_test = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(rgb_mean, rgb_std),
        ])

class NCFMclass(Dataset):

    def __init__(self, path, split, seed=135):
        """
        :param path: path where data files are stored
        :param split: split, one of 'TRAIN', 'VALID', or 'TEST'
        """
        self.path = path
        self.split = split.upper()
        assert self.split in {'TRAIN', 'VALID', 'TEST'}

        # Train/val split
        if self.split in {'TRAIN', 'VALID'}:
            train_files = []
            valid_files = []
            np.random.seed(seed)
            for c in label_clss:
                im_files = os.listdir(os.path.join(path, 'train_cls', c))
                mask = np.random.rand(len(im_files)) > 0.1
                train_files += [c+'/'+f for f in list(np.array(im_files)[mask])]
                valid_files += [c+'/'+f for f in list(np.array(im_files)[~mask])]
            np.random.shuffle(train_files)

            # Set files
            self.files = train_files if self.split == 'TRAIN' else valid_files

        else:
            self.files = os.listdir(os.path.join(path, 'test_cls'))

    def __getitem__(self, i, transform=None):
        # Read image
        if self.split == 'TEST':
            image = Image.open(os.path.join(self.path, 'test_cls', self.files[i]), mode='r')
            image = image.convert('RGB')
            if transform is None: transform = tsfm_test
            return transform(image), self.files[i]

        image = Image.open(os.path.join(self.path, 'train_cls', self.files[i]), mode='r')
        image = image.convert('RGB')
        if transform is None:
            if self.split == 'TRAIN':
                if self.files[i].split('/')[0] == 'NoF':
                    transform = tsfm_trn_nof
                else:
                    transform = tsfm_trn
            else:
                transform = tsfm_test

        # Read label
        label = clss_to_label[self.files[i].split('/')[0]]

        return transform(image), torch.LongTensor([label])

    def __len__(self):
        return len(self.files)


