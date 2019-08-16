# Fish monitoring trained with yolov3, from darknet-53
# Environment: cuda 10.0, cuDNN 10.0
# The script is for data preparation for training and testing

import os, re, json, random
import wget
from shutil import copyfile

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

path = './'
ncfm_labels = ('fish',)
label_clss = ['DOL', 'LAG', 'BET', 'OTHER', 'SHARK', 'YFT', 'ALB']
no_bbs = ['img_00568.jpg', 'img_01958.jpg', 'img_07008.jpg', 'img_00425.jpg',
          'img_04798.jpg', 'img_06460.jpg', 'img_02292.jpg', 'img_00576.jpg',
          'img_00379.jpg', 'img_06773.jpg', 'img_05444.jpg', 'img_06082.jpg',
          'img_03183.jpg', 'img_04558.jpg', 'img_02785.jpg']


def create_files_split_gt(path, seed=135):
    img_files = []
    train_files = []
    valid_files = []
    np.random.seed(seed)
    for c in label_clss:
        fs = os.listdir(os.path.join(path, 'train', c))
        img_files += [c+'/'+f for f in fs if f not in no_bbs]

        mask = np.random.rand(len(fs)) > 0.1
        train_files += [f for f in list(np.array(fs)[mask]) if f not in no_bbs]
        valid_files += [f for f in list(np.array(fs)[~mask]) if f not in no_bbs]
    np.random.shuffle(train_files)

    gt = {}
    for c in label_clss:
        with open(os.path.join(path, 'fish_bbox', c.lower()+'_labels.json')) as f:
            gt[c] = json.load(f)

    return img_files, train_files, valid_files, gt


def prepare_data(path, img_files, train_files, valid_files, gt):
    # save required files in 'train_yolo'
    if not os.path.exists(os.path.join(path, 'train_yolo')):
        os.mkdir(os.path.join(path, 'train_yolo'))

    for i in range(len(img_files)):
        # load image size
        im = Image.open(os.path.join(path, 'train', img_files[i]))
        width, height = im.size
        width, height = float(width), float(height)

        # correct bbox to yolo format and save to 'train_yolo'
        # yolo format: normalized center-size coordinates (c_x, c_y, w, h)
        cls, fname = img_files[i].split('/')
        mask = list(map(lambda x: x['filename'].split('/')[-1]==fname, gt[cls]))
        bbs = np.array(gt[cls])[mask][0]['annotations']
        bboxes = [[(b['x']+0.5*b['width'])/width, (b['y']+0.5*b['height'])/height, b['width']/width, b['height']/height] for b in bbs]
        with open(os.path.join(path, 'train_yolo', fname.split('.')[0]+'.txt'), 'w') as f:
            for box in bboxes:
                f.write(f'0 {box[0]} {box[1]} {box[2]} {box[3]}\n')

        # copy images to 'train_yolo'
        copyfile(os.path.join(path, 'train', img_files[i]), os.path.join(path, 'train_yolo', fname))

    # save required files to 'cfg_fish'
    if not os.path.exists(os.path.join(path, 'cfg_fish')):
        os.mkdir(os.path.join(path, 'cfg_fish'))

    with open(os.path.join(path, 'cfg_fish', 'train.txt'), 'w') as f:
        for fname in train_files:
            f.write('train_yolo/'+fname+'\n')

    with open(os.path.join(path, 'cfg_fish', 'valid.txt'), 'w') as f:
        for fname in valid_files:
            f.write('train_yolo/'+fname+'\n')

    with open(os.path.join(path, 'cfg_fish', 'test.txt'), 'w') as f:
        for fname in os.listdir('test_stg1'): f.write('test_stg1/'+fname+'\n')
        for fname in os.listdir('test_stg2'): f.write('test_stg2/'+fname+'\n')

    with open(os.path.join(path, 'cfg_fish', 'obj.data'), 'w') as f:
        f.write("classes= 1\n")
        f.write("train  = cfg_fish/train.txt\n")
        f.write("valid  = cfg_fish/valid.txt\n")
        f.write("names = cfg_fish/obj.names\n")
        f.write("backup = weights/")

    with open(os.path.join(path, 'cfg_fish', 'obj.names'), 'w') as the_file:
        for c in ncfm_labels:
            f.write(f"{c}\n")

    # create dir 'weights'
    if not os.path.exists(os.path.join(path, 'weights')):
        os.mkdir(os.path.join(path, 'weights'))


def set_cfg(cfg_file, batch_size=64):
    with open(cfg_file) as f:
        content = f.readlines()

    num_class = 1
    num_filter = (num_class+5) * 3
    num_subdivision = 8

    updated = []
    for line in content:
        if line.startswith('batch='):
            line = f'batch={batch_size}\n'
        if line.startswith('subdivisions='):
            line = f'subdivisions={num_subdivision}\n'
        if line.startswith('filters=255'):
            line = f'filters={num_filter}\n'
        if line.startswith('classes=80'):
            line = f'classes={num_class}\n'
        updated.append(line)

    with open(os.path.join(path, 'cfg_fish', 'fish_yolov3.cfg'), 'w') as f:
        for line in updated:
            f.write(line)


def download_weights(url):
    print("Downloading the pretrained model...")
    wget.download(url)


def crop_test_img(path):
    with open(os.path.join(path, 'cfg_fish', 'test.txt')) as f:
        test_files = f.readlines()

    with open(os.path.join(path, 'result.txt')) as f:
        results = f.readlines()

    line_id, file_id = 0, 0
    while True:

        def save_no_box_image():
            fname = test_files[file_id][:-1]
            copyfile(os.path.join(path, fname), os.path.join(path, 'test_cls', fname.split('/')[-1]))

        def crop_and_save_boxed_image():
            detects = sorted(detects)
            fname = test_files[file_id][:-1]
            im = Image.open(os.path.join(path, fname))
            bbox = detects[-1][1:]  # yolo pred box: (left_x, top_y, width, height) in original size
            bbox = [bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]]
            im = im.crop(bbox)
            im.save(os.path.join(path, 'test_cls', fname.split('/')[-1]))

        if line_id+1 == len(results):
            if results[line_id].startswith('Enter'):
                save_no_box_image()
            else:
                detects.append( [int(s) for s in re.findall(r'\d+', results[line_id])] )
                crop_and_save_boxed_image()
            break

        if results[line_id].startswith('Enter'):
            if results[line_id+1].startswith('Enter'):
                save_no_box_image()
                file_id += 1
            else:
                detects = []

        else:
            detects.append( [int(s) for s in re.findall(r'\d+', results[line_id])] )
            if results[line_id+1].startswith('Enter'):
                crop_and_save_boxed_image()
                file_id += 1

        line_id += 1


if __name__ == '__main__':
    img_files, train_files, valid_files, gt = create_files_split_gt(path)
    prepare_data(path, img_files, train_files, valid_files, gt)
    set_cfg('darknet/cfg/yolov3.cfg')
    download_weights('https://pjreddie.com/media/files/darknet53.conv.74')

