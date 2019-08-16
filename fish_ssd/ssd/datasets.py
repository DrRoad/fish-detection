import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
from ..utils import *

label_clss = ['DOL', 'LAG', 'BET', 'OTHER', 'SHARK', 'YFT', 'ALB']
no_bbs = ['img_00568.jpg', 'img_01958.jpg', 'img_07008.jpg', 'img_00425.jpg',
          'img_04798.jpg', 'img_06460.jpg', 'img_02292.jpg', 'img_00576.jpg',
          'img_00379.jpg', 'img_06773.jpg', 'img_05444.jpg', 'img_06082.jpg',
          'img_03183.jpg', 'img_04558.jpg', 'img_02785.jpg']


class NCFMdataset(Dataset):

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
                im_files = os.listdir(os.path.join(path, 'train', c))
                mask = np.random.rand(len(im_files)) > 0.1
                train_files += [c+'/'+f for f in list(np.array(im_files)[mask]) if f not in no_bbs]
                valid_files += [c+'/'+f for f in list(np.array(im_files)[~mask]) if f not in no_bbs]
            np.random.shuffle(train_files)

            # Read data files
            self.files = train_files if self.split == 'TRAIN' else valid_files
            self.gt = {}
            for c in label_clss:
                with open(os.path.join(path, 'fish_bbox', c.lower()+'_labels.json')) as f:
                    self.gt[c] = json.load(f)

        else:
            self.files = ['test_stg2/'+f for f in os.listdir(os.path.join(path, 'test_stg2'))]

    def __getitem__(self, i):
        # Read image
        if self.split == 'TEST':
            image = Image.open(os.path.join(self.path, self.files[i]), mode='r')
            image = image.convert('RGB')
            return image

        image = Image.open(os.path.join(self.path, 'train', self.files[i]), mode='r')
        image = image.convert('RGB')

        # Read objects in this image (bounding boxes, labels)
        cls, fname = self.files[i].split('/')
        mask = list(map(lambda x: x['filename'].split('/')[-1]==fname, self.gt[cls]))
        bbs = np.array(self.gt[cls])[mask][0]['annotations']

        boxes = torch.FloatTensor([[b['x'], b['y'], (b['x']+b['width']), (b['y']+b['height'])] for b in bbs])  # (n_objects, 4)
        labels = torch.LongTensor([1]*len(bbs))  # (n_objects)

        # Apply transformations
        image, boxes, labels = transform(image, boxes, labels, split=self.split)

        return image, boxes, labels

    def __len__(self):
        return len(self.files)

    def collate_fn(self, batch):
        """
        Since each image may have a different number of objects, we need a collate function (to be passed to the DataLoader).

        This describes how to combine these tensors of different sizes. We use lists.

        Note: this need not be defined in this Class, can be standalone.

        :param batch: an iterable of N sets from __getitem__()
        :return: a tensor of images, lists of varying-size tensors of bounding boxes, and labels
        """

        images = list()
        boxes = list()
        labels = list()

        for b in batch:
            images.append(b[0])
            boxes.append(b[1])
            labels.append(b[2])

        images = torch.stack(images, dim=0)

        return images, boxes, labels  # tensor (N, ...), 3 lists of N tensors each


