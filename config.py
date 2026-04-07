# paths
IMG_DIR   = 'data/VOC2012/JPEGImages'
ANN_DIR   = 'data/VOC2012/Annotations'
TRAIN_TXT = 'data/VOC2012/ImageSets/Main/train.txt'
VAL_TXT   = 'data/VOC2012/ImageSets/Main/val.txt'

# classes
CLASSES = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
CLASS2IDX = {cls: idx for idx, cls in enumerate(CLASSES)}