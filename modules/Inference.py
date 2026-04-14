import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.ops import nms
from torchvision import transforms
from PIL import Image
from config import DEVICE, CLASSES, ANN_DIR
import xml.etree.ElementTree as ET


# image transform for inference (no augmentation)
transform = transforms.Compose([
    transforms.Resize((448, 448)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_gt_boxes(xml_path):
    """
    Get ground truth boxes from XML file.

    Args:
        xml_path (str): path to XML annotation file
    Returns:
        boxes  (list): list of [x1, y1, x2, y2] normalized 0~1
        labels (list): list of class names
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    w = float(root.find('size/width').text)
    h = float(root.find('size/height').text)

    boxes  = []
    labels = []

    for obj in root.findall('object'):
        name = obj.find('name').text.strip()
        b    = obj.find('bndbox')
        x1   = float(b.find('xmin').text) / w
        y1   = float(b.find('ymin').text) / h
        x2   = float(b.find('xmax').text) / w
        y2   = float(b.find('ymax').text) / h
        boxes.append([x1, y1, x2, y2])
        labels.append(name)

    return boxes, labels


def decode_predictions(pred, S, B, C, conf_thresh):
    """
    Decode model output into boxes, scores and labels.

    Args:
        pred        (tensor): model output, shape (S, S, B*5+C)
        S           (int):    grid size
        B           (int):    number of boxes per cell
        C           (int):    number of classes
        conf_thresh (float):  score threshold (conf x cls_score)
    Returns:
        boxes  (tensor): shape (N, 4) normalized 0~1
        scores (tensor): shape (N,)
        labels (tensor): shape (N,)
    """
    boxes  = []
    scores = []
    labels = []

    cell_size = 1.0 / S

    for gy in range(S):
        for gx in range(S):
            for b in range(B):
                conf = torch.sigmoid(pred[gy, gx, b * 5 + 4])

                cx = (gx + torch.sigmoid(pred[gy, gx, b * 5 + 0])) * cell_size
                cy = (gy + torch.sigmoid(pred[gy, gx, b * 5 + 1])) * cell_size
                w  = torch.sigmoid(pred[gy, gx, b * 5 + 2])
                h  = torch.sigmoid(pred[gy, gx, b * 5 + 3])

                x1 = (cx - w / 2).clamp(0, 1)
                y1 = (cy - h / 2).clamp(0, 1)
                x2 = (cx + w / 2).clamp(0, 1)
                y2 = (cy + h / 2).clamp(0, 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                cls_scores          = torch.sigmoid(pred[gy, gx, B * 5:])
                cls_score, cls_idx  = cls_scores.max(dim=0)

                # filter by final score = conf x cls_score
                score = conf * cls_score
                if score < conf_thresh:
                    continue

                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                scores.append(score.item())
                labels.append(cls_idx.item())

    if len(boxes) == 0:
        return (
            torch.zeros((0, 4)),
            torch.zeros((0,)),
            torch.zeros((0,), dtype=torch.long)
        )

    return (
        torch.tensor(boxes,  dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long)
    )


def inference(model, img_path, S, B, C, conf_thresh, iou_thresh):
    """
    Run inference on a single image and visualise predictions vs ground truth.

    Args:
        model       : trained YOLO model
        img_path    (str): path to image file
        S, B, C     : YOLO parameters
        conf_thresh (float): score threshold
        iou_thresh  (float): IoU threshold for NMS
    """
    # load image
    img_pil    = Image.open(img_path).convert('RGB')
    img_tensor = transform(img_pil).unsqueeze(0).to(DEVICE)
    W, H       = img_pil.size

    # forward pass
    model.eval()
    with torch.no_grad():
        pred = model(img_tensor)[0].cpu()

    # decode predictions
    boxes, scores, labels = decode_predictions(pred, S, B, C, conf_thresh)

    # apply NMS
    if boxes.shape[0] > 0:
        keep   = nms(boxes, scores, iou_thresh)
        boxes  = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    # get ground truth
    xml_path        = img_path.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml')
    gt_boxes, gt_labels = get_gt_boxes(xml_path)

    # draw
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_pil)

    # draw ground truth boxes (green)
    for (x1, y1, x2, y2), name in zip(gt_boxes, gt_labels):
        x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
        rect = patches.Rectangle(
            (x1, y1), x2-x1, y2-y1,
            linewidth=2, edgecolor='green', facecolor='none'
        )
        ax.add_patch(rect)
        ax.text(
            x1, min(H, y2 + 15),
            name,
            color='green', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5, pad=2)
        )

    # draw predicted boxes (red)
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        x1, y1, x2, y2 = x1*W, y1*H, x2*W, y2*H
        rect = patches.Rectangle(
            (x1.item(), y1.item()), (x2-x1).item(), (y2-y1).item(),
            linewidth=2, edgecolor='red', facecolor='none'
        )
        ax.add_patch(rect)
        cls_name = CLASSES[labels[i].item()]
        score    = scores[i].item()
        ax.text(
            x1.item(), max(0, y1.item() - 5),
            f"{cls_name} {score:.2f}",
            color='red', fontsize=10,
            bbox=dict(facecolor='white', alpha=0.5, pad=2)
        )

    plt.axis('off')
    plt.title(f"Green: Ground Truth | Red: Predictions ({boxes.shape[0]} objects)")
    plt.tight_layout()
    plt.show()