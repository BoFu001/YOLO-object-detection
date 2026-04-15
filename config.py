import os
import torch

# paths
if os.path.exists("/kaggle/input") and not os.path.exists("/content/drive"):
    # Kaggle
    IMG_DIR     = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/JPEGImages'
    ANN_DIR     = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/Annotations'
    TRAIN_TXT   = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/ImageSets/Main/train.txt'
    VAL_TXT     = '/kaggle/input/datasets/huanghanchina/pascal-voc-2012/VOC2012/ImageSets/Main/val.txt'
    CKPT_DIR    = '/kaggle/working'
    NUM_WORKERS = 2
elif os.path.exists("/content/VOC2012"):
    # Colab local
    IMG_DIR     = '/content/VOC2012/VOC2012/JPEGImages'
    ANN_DIR     = '/content/VOC2012/VOC2012/Annotations'
    TRAIN_TXT   = '/content/VOC2012/VOC2012/ImageSets/Main/train.txt'
    VAL_TXT     = '/content/VOC2012/VOC2012/ImageSets/Main/val.txt'
    CKPT_DIR    = '/content/drive/MyDrive/checkpoints'
    NUM_WORKERS = 4
elif os.path.exists("/content/drive"):
    # Colab Drive
    IMG_DIR     = '/content/drive/MyDrive/VOC2012/VOC2012/JPEGImages'
    ANN_DIR     = '/content/drive/MyDrive/VOC2012/VOC2012/Annotations'
    TRAIN_TXT   = '/content/drive/MyDrive/VOC2012/VOC2012/ImageSets/Main/train.txt'
    VAL_TXT     = '/content/drive/MyDrive/VOC2012/VOC2012/ImageSets/Main/val.txt'
    CKPT_DIR    = '/content/drive/MyDrive/checkpoints'
    NUM_WORKERS = 2
else:
    # Local
    IMG_DIR   = 'data/VOC2012/JPEGImages'
    ANN_DIR   = 'data/VOC2012/Annotations'
    TRAIN_TXT = 'data/VOC2012/ImageSets/Main/train.txt'
    VAL_TXT   = 'data/VOC2012/ImageSets/Main/val.txt'
    CKPT_DIR  = 'checkpoints'
    NUM_WORKERS = 0

# device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# reproducibility
SEED = 42

# classes
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS2IDX = {cls: idx for idx, cls in enumerate(CLASSES)}
