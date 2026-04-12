import torch
from torchvision.ops import nms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from config import DEVICE


def decode_predictions(pred, S, B, C, conf_thresh):
    """
    Decode model output into boxes, scores and labels.

    Args:
        pred        (tensor): model output, shape (S, S, B*5+C)
        S           (int):    grid size
        B           (int):    number of boxes per cell
        C           (int):    number of classes
        conf_thresh (float):  confidence threshold
    Returns:
        boxes  (tensor): shape (N, 4) in x1y1x2y2 format (0~1)
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
                # confidence score
                conf = torch.sigmoid(pred[gy, gx, b * 5 + 4])

                if conf < conf_thresh:
                    continue

                # box coordinates
                cx = (gx + torch.sigmoid(pred[gy, gx, b * 5 + 0])) * cell_size
                cy = (gy + torch.sigmoid(pred[gy, gx, b * 5 + 1])) * cell_size
                w  = torch.sigmoid(pred[gy, gx, b * 5 + 2])
                h  = torch.sigmoid(pred[gy, gx, b * 5 + 3])

                # convert to x1y1x2y2
                x1 = (cx - w / 2).clamp(0, 1)
                y1 = (cy - h / 2).clamp(0, 1)
                x2 = (cx + w / 2).clamp(0, 1)
                y2 = (cy + h / 2).clamp(0, 1)

                if x2 <= x1 or y2 <= y1:
                    continue

                # class prediction
                cls_scores = torch.sigmoid(pred[gy, gx, B * 5:])
                cls_score, cls_idx = cls_scores.max(dim=0)

                # final score = confidence × class score
                score = conf * cls_score

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


def evaluate(model, test_loader, S, B, C, conf_thresh, iou_thresh):
    """
    Evaluate model on test set using mAP.

    Args:
        model       : trained YOLO model
        test_loader : DataLoader for test set
        S, B, C     : YOLO parameters
        conf_thresh : confidence threshold
        iou_thresh  : IoU threshold for NMS
    Returns:
        results (dict): mAP and other metrics
    """
    model.eval()
    metric = MeanAveragePrecision(iou_type="bbox")

    with torch.no_grad():
        for imgs, targets in test_loader:
            imgs    = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds   = model(imgs)

            preds_list   = []
            targets_list = []

            for i in range(imgs.shape[0]):
                # decode predictions
                boxes, scores, labels = decode_predictions(
                    preds[i].cpu(), S, B, C, conf_thresh
                )

                # apply NMS
                if boxes.shape[0] > 0:
                    keep   = nms(boxes, scores, iou_thresh)
                    boxes  = boxes[keep]
                    scores = scores[keep]
                    labels = labels[keep]

                preds_list.append({
                    "boxes":  boxes,
                    "scores": scores,
                    "labels": labels
                })

                # decode targets
                tgt        = targets[i].cpu()
                tgt_boxes  = []
                tgt_labels = []

                for gy in range(S):
                    for gx in range(S):
                        if tgt[gy, gx, 4] > 0.5:
                            cx = (gx + tgt[gy, gx, 0]) / S
                            cy = (gy + tgt[gy, gx, 1]) / S
                            w  = tgt[gy, gx, 2]
                            h  = tgt[gy, gx, 3]
                            x1 = (cx - w / 2).clamp(0, 1)
                            y1 = (cy - h / 2).clamp(0, 1)
                            x2 = (cx + w / 2).clamp(0, 1)
                            y2 = (cy + h / 2).clamp(0, 1)
                            tgt_boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                            cls = tgt[gy, gx, B * 5:].argmax().item()
                            tgt_labels.append(cls)

                if len(tgt_boxes) == 0:
                    targets_list.append({
                        "boxes":  torch.zeros((0, 4)),
                        "labels": torch.zeros((0,), dtype=torch.long)
                    })
                else:
                    targets_list.append({
                        "boxes":  torch.tensor(tgt_boxes,  dtype=torch.float32),
                        "labels": torch.tensor(tgt_labels, dtype=torch.long)
                    })

            metric.update(preds_list, targets_list)

    results = metric.compute()
    print(f"mAP@0.50:      {results['map_50'].item():.4f}")
    print(f"mAP@0.50:0.95: {results['map'].item():.4f}")

    return results