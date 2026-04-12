import os

if os.path.exists("/kaggle/input"):
    # Kaggle
    IMG_DIR     = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/JPEGImages'
    ANN_DIR     = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/Annotations'
    TRAIN_TXT   = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/ImageSets/Main/train.txt'
    VAL_TXT     = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/ImageSets/Main/val.txt'
    CKPT_DIR    = '/kaggle/working'
    NUM_WORKERS = 2
else:
    # Local
    IMG_DIR   = 'data/VOC2012/JPEGImages'
    ANN_DIR   = 'data/VOC2012/Annotations'
    TRAIN_TXT = 'data/VOC2012/ImageSets/Main/train.txt'
    VAL_TXT   = 'data/VOC2012/ImageSets/Main/val.txt'
    CKPT_DIR  = 'checkpoints'
    NUM_WORKERS = 0



# classes
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS2IDX = {cls: idx for idx, cls in enumerate(CLASSES)}