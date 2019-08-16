import os, re, json, random
import os.path as osp
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = './'
label_cls = ['DOL', 'LAG', 'BET', 'OTHER', 'SHARK', 'YFT', 'ALB']
no_bbs = ['img_00568.jpg', 'img_01958.jpg', 'img_07008.jpg', 'img_00425.jpg',
          'img_04798.jpg', 'img_06460.jpg', 'img_02292.jpg', 'img_00576.jpg',
          'img_00379.jpg', 'img_06773.jpg', 'img_05444.jpg', 'img_06082.jpg',
          'img_03183.jpg', 'img_04558.jpg', 'img_02785.jpg']


def create_files_split_gt(path, seed=135):
    train_files = []
    valid_files = []
    np.random.seed(seed)
    for c in label_cls:
        fs = os.listdir(os.path.join(path, 'train', c))
        mask = np.random.rand(len(fs)) > 0.1
        train_files += [c+'/'+f for f in np.array(fs)[mask] if f not in no_bbs]
        valid_files += [c+'/'+f for f in np.array(fs)[~mask] if f not in no_bbs]

    gt = []
    for c in label_cls:
        with open(osp.join(path, 'fish_bbox', c.lower()+'_labels.json')) as f:
            gt.extend(json.load(f))
    gt = {imgt['filename'].split('/')[-1]: imgt for imgt in gt}

    return train_files, valid_files, gt


def prepare_data(path, img_files, gt, split):
    # create required dirs
    if not osp.exists(osp.join(path, 'train_frcnn')):
        os.makedirs(osp.join(path, 'train_frcnn', 'images'))
        os.makedirs(osp.join(path, 'train_frcnn', 'annotations'))

    # annotations to be prepared
    anno = {}
    anno['type'] = 'instances'
    anno['categories'] = [{'supercategory': 'none', 'id': 1, 'name': 'fish'}]

    images_anno, annos_anno = [], []
    obj_id = 1
    for imfile in img_files:
        # image information
        fname = imfile.split('/')[-1]
        width, height = Image.open(osp.join(path, 'train', imfile)).size
        im_id = int(fname.split('.')[0].split('_')[-1])
        images_anno.append(
            {'file_name':fname, 'width':width, 'height':height, 'id':im_id})

        # object annotation
        im_annos = gt[fname]['annotations']
        for o in im_annos:
            bb = [o['x'], o['y'], o['width'], o['height']]
            annos_anno.append(
                {'bbox':bb, 'area':bb[-2]*bb[-1], 'image_id':im_id, 'id':obj_id,
                 'category_id':1, 'iscrowd':0, 'ignore':0})
            obj_id += 1

        # copy images to 'train_frcnn/images'
        copyfile(osp.join(path, 'train', imfile), osp.join(path, 'train_frcnn/images', fname))

    anno['images'] = images_anno
    anno['annotations'] = annos_anno

    # to json file
    with open(osp.join(path, 'train_frcnn/annotations', f'fish_{split}.json'), 'w') as f:
        json.dump(anno, f)


if __name__ == '__main__':
    train_files, valid_files, gt = create_files_split_gt(path)
    prepare_data(path, train_files, gt, 'train')
    prepare_data(path, valid_files, gt, 'val')





