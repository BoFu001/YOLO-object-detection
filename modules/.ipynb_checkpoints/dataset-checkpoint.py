from torchvision import transforms
from PIL import Image

# define image transform pipeline
transform = transforms.Compose([
    transforms.Resize((448, 448)),        # resize image to 448x448
    transforms.ToTensor(),                # convert PIL image to tensor, pixel values 0~1
    transforms.Normalize(                 # normalize using ImageNet mean and std
        mean=[0.485, 0.456, 0.406],       # mean for R, G, B channels
        std=[0.229, 0.224, 0.225]         # std for R, G, B channels
    )
])


def load_image(img_path):
    # open image file
    img = Image.open(img_path)

    # convert to RGB (some images might be grayscale or RGBA)
    img = img.convert('RGB')

    # apply transform: resize + to tensor + normalize
    img = transform(img)

    # return tensor of shape (3, 448, 448)
    return img






# test
if __name__ == '__main__':
    
    img_path = 'data/VOC2012/JPEGImages/2008_000008.jpg'
    
    img = load_image(img_path)
    
    print("Image shape:", img.shape)   # (3, 448, 448)
    print("Min value:",  img.min())    # around -2.0 after normalize
    print("Max value:",  img.max())    # around 2.0 after normalize







import xml.etree.ElementTree as ET
import torch

def parse_xml(xml_path):
    # read and parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # get original image width and height
    # used to normalize coordinates to 0~1
    w = float(root.find('size/width').text)
    h = float(root.find('size/height').text)

    boxes  = []  # store bounding boxes
    labels = []  # store class labels

    # loop through each object in the XML
    for obj in root.findall('object'):

        # get class name, e.g. 'dog', 'car'
        name = obj.find('name').text.strip()

        # only process if class is in our 20 classes
        if name in CLASS2IDX:

            # get bounding box coordinates (pixel values)
            b  = obj.find('bndbox')
            x1 = float(b.find('xmin').text)
            y1 = float(b.find('ymin').text)
            x2 = float(b.find('xmax').text)
            y2 = float(b.find('ymax').text)
    
            # normalize coordinates to 0~1
            x1 = x1 / w
            y1 = y1 / h
            x2 = x2 / w
            y2 = y2 / h
    
            # add to lists
            boxes.append([x1, y1, x2, y2])
            labels.append(CLASS2IDX[name])

    # if no objects found, return empty tensors
    if len(boxes) == 0:
        return torch.zeros((0, 4)), torch.zeros((0,), dtype=torch.long)

    # convert lists to tensors
    boxes  = torch.tensor(boxes,  dtype=torch.float32)  # shape (N, 4)
    labels = torch.tensor(labels, dtype=torch.long)      # shape (N,)

    return boxes, labels






# test
if __name__ == '__main__':
    xml_path = 'data/VOC2012/Annotations/2008_000008.xml'
    boxes, labels = parse_xml(xml_path)
    
    print("boxes:",  boxes)    # tensor of coordinates
    print("labels:", labels)   # tensor of class indices
    print("number of objects:", len(boxes))






    




def encode_yolo(boxes, labels):
    # input:
    #   boxes  : (N, 4)  normalized x1,y1,x2,y2
    #   labels : (N,)    class index
    # output:
    #   target : (7, 7, 30)

    # initialize all zeros
    # shape: (S, S, B*5 + C) = (7, 7, 30)
    target = torch.zeros((S, S, B * 5 + C))

    # loop through each object
    for i in range(len(boxes)):

        # get coordinates
        x1, y1, x2, y2 = boxes[i]

        # get class index
        cls_idx = labels[i].item()

        # calculate center point
        cx = (x1 + x2) / 2.0
        cy = (y1 + y2) / 2.0

        # calculate width and height
        w = x2 - x1
        h = y2 - y1

        # find which grid cell the center falls into
        gx = int(cx * S)       # column index 0~6
        gy = int(cy * S)       # row index 0~6

        # prevent out of bounds
        gx = min(gx, S - 1)
        gy = min(gy, S - 1)

        # if cell is empty, fill in the object
        if target[gy, gx, 4] == 0.0:

            # calculate offset inside the cell (0~1)
            x_cell = cx * S - gx
            y_cell = cy * S - gy

            # fill box 1 values
            target[gy, gx, 0] = x_cell   # x offset in cell
            target[gy, gx, 1] = y_cell   # y offset in cell
            target[gy, gx, 2] = w        # width
            target[gy, gx, 3] = h        # height
            target[gy, gx, 4] = 1.0      # confidence = 1

            # fill class label (one-hot encoding)
            target[gy, gx, B * 5 + cls_idx] = 1.0

    return target




# test
if __name__ == '__main__':
    xml_path = 'data/VOC2012/Annotations/2008_000008.xml'
    boxes, labels = parse_xml(xml_path)
    
    target = encode_yolo(boxes, labels)
    
    print("target shape:", target.shape)          # (7, 7, 30)
    print("cells with object:", (target[..., 4] > 0).sum().item())











from torch.utils.data import Dataset


class VOCDataset(Dataset):

    def __init__(self, txt_file):
        # read image ID list from txt file
        with open(txt_file, 'r') as f:
            self.img_ids = [line.strip() for line in f.readlines()]

    def __len__(self):
        # return total number of images
        return len(self.img_ids)

    def __getitem__(self, idx):
        # get image ID e.g. '2008_000008'
        img_id = self.img_ids[idx]

        # load image
        img = load_image(f'{IMG_DIR}/{img_id}.jpg')

        # parse XML and encode
        boxes, labels = parse_xml(f'{ANN_DIR}/{img_id}.xml')
        target = encode_yolo(boxes, labels)

        return img, target







# test
if __name__ == '__main__':
    dataset = VOCDataset(TRAIN_TXT)
    
    print("Total images:", len(dataset))
    
    img, target = dataset[0]
    print("img shape:   ", img.shape)
    print("target shape:", target.shape)
