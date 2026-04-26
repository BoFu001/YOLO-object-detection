import copy
import io
import re
from contextlib import redirect_stdout

import pandas as pd
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from modules.Evaluation import evaluate
from modules.Loss import YOLOLoss
from my_config import DEVICE, CLASSES


def _parse_map_metrics(eval_text: str):
    """Extract mAP@0.50 and mAP@0.50:0.95 from evaluate() stdout."""
    map50_match = re.search(r"mAP@0\.50:\s*([0-9]*\.?[0-9]+)", eval_text)
    map5095_match = re.search(r"mAP@0\.50:0\.95:\s*([0-9]*\.?[0-9]+)", eval_text)

    map50 = float(map50_match.group(1)) if map50_match else None
    map5095 = float(map5095_match.group(1)) if map5095_match else None
    return map50, map5095


def _evaluate_map(model, loader, S, B, C, conf_thresh, iou_thresh):
    """Run evaluate() on a loader and capture printed mAP values."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _ = evaluate(
            model=model,
            test_loader=loader,
            S=S,
            B=B,
            C=C,
            conf_thresh=float(conf_thresh),
            iou_thresh=float(iou_thresh),
        )
    eval_text = buffer.getvalue()
    val_map, val_map_50_95 = _parse_map_metrics(eval_text)
    return val_map, val_map_50_95, eval_text


def _clip01(x):
    return max(0.0, min(1.0, float(x)))


def box_iou_xyxy(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h

    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    if union <= 0:
        return 0.0
    return inter / union


def nms_boxes(boxes, iou_thresh=0.5):
    boxes = sorted(boxes, key=lambda x: x["score"], reverse=True)
    kept = []

    while boxes:
        current = boxes.pop(0)
        kept.append(current)

        remaining = []
        for b in boxes:
            same_class = b["class_id"] == current["class_id"]
            if same_class and box_iou_xyxy(b["box"], current["box"]) > iou_thresh:
                continue
            remaining.append(b)
        boxes = remaining

    return kept


def decode_pred_tensor(pred_tensor, S, B, C, conf_thresh=0.25, nms_iou=0.5):
    pred_tensor = pred_tensor.detach().cpu()
    boxes = []

    for gy in range(S):
        for gx in range(S):
            class_probs = pred_tensor[gy, gx, B * 5 : B * 5 + C]
            class_prob, class_id = torch.max(class_probs, dim=0)
            class_prob = float(class_prob.item())
            class_id = int(class_id.item())

            for b in range(B):
                base = b * 5
                x_cell = float(pred_tensor[gy, gx, base + 0].item())
                y_cell = float(pred_tensor[gy, gx, base + 1].item())
                w = float(pred_tensor[gy, gx, base + 2].item())
                h = float(pred_tensor[gy, gx, base + 3].item())
                conf = float(pred_tensor[gy, gx, base + 4].item())

                score = conf * class_prob
                if score < conf_thresh:
                    continue

                cx = (gx + x_cell) / S
                cy = (gy + y_cell) / S

                x1 = _clip01(cx - w / 2.0)
                y1 = _clip01(cy - h / 2.0)
                x2 = _clip01(cx + w / 2.0)
                y2 = _clip01(cy + h / 2.0)

                if x2 <= x1 or y2 <= y1:
                    continue

                boxes.append({
                    "class_id": class_id,
                    "score": score,
                    "box": [x1, y1, x2, y2],
                })

    return nms_boxes(boxes, iou_thresh=nms_iou)


def decode_target_tensor(target_tensor, S, B, C):
    target_tensor = target_tensor.detach().cpu()
    boxes = []

    for gy in range(S):
        for gx in range(S):
            if float(target_tensor[gy, gx, 4].item()) <= 0.0:
                continue

            x_cell = float(target_tensor[gy, gx, 0].item())
            y_cell = float(target_tensor[gy, gx, 1].item())
            w = float(target_tensor[gy, gx, 2].item())
            h = float(target_tensor[gy, gx, 3].item())

            class_vec = target_tensor[gy, gx, B * 5 : B * 5 + C]
            class_id = int(torch.argmax(class_vec).item())

            cx = (gx + x_cell) / S
            cy = (gy + y_cell) / S

            x1 = _clip01(cx - w / 2.0)
            y1 = _clip01(cy - h / 2.0)
            x2 = _clip01(cx + w / 2.0)
            y2 = _clip01(cy + h / 2.0)

            if x2 <= x1 or y2 <= y1:
                continue

            boxes.append({
                "class_id": class_id,
                "score": 1.0,
                "box": [x1, y1, x2, y2],
            })

    return boxes


def match_image(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched_gt = set()
    tp = 0
    fp = 0

    pred_boxes = sorted(pred_boxes, key=lambda x: x["score"], reverse=True)

    for pred in pred_boxes:
        best_iou = 0.0
        best_gt_idx = None

        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            if pred["class_id"] != gt["class_id"]:
                continue

            iou = box_iou_xyxy(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gi

        if best_gt_idx is not None and best_iou >= iou_thresh:
            tp += 1
            matched_gt.add(best_gt_idx)
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched_gt)
    return tp, fp, fn


def evaluate_detection_f1(model, loader, S, B, C, conf_thresh=0.25, iou_thresh=0.5, match_iou_thresh=0.5):
    model.eval()

    total_tp = 0
    total_fp = 0
    total_fn = 0

    per_class = {cls_idx: {"tp": 0, "fp": 0, "fn": 0} for cls_idx in range(C)}

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            preds = model(imgs).detach().cpu()
            targets = targets.detach().cpu()

            for i in range(preds.shape[0]):
                pred_boxes = decode_pred_tensor(
                    preds[i], S=S, B=B, C=C,
                    conf_thresh=conf_thresh,
                    nms_iou=iou_thresh
                )
                gt_boxes = decode_target_tensor(targets[i], S=S, B=B, C=C)

                tp, fp, fn = match_image(
                    pred_boxes,
                    gt_boxes,
                    iou_thresh=match_iou_thresh
                )

                total_tp += tp
                total_fp += fp
                total_fn += fn

                matched_gt = set()
                pred_sorted = sorted(pred_boxes, key=lambda x: x["score"], reverse=True)

                for pred in pred_sorted:
                    best_iou = 0.0
                    best_gt_idx = None
                    for gi, gt in enumerate(gt_boxes):
                        if gi in matched_gt:
                            continue
                        if pred["class_id"] != gt["class_id"]:
                            continue
                        iou = box_iou_xyxy(pred["box"], gt["box"])
                        if iou > best_iou:
                            best_iou = iou
                            best_gt_idx = gi

                    cls = pred["class_id"]
                    if best_gt_idx is not None and best_iou >= match_iou_thresh:
                        per_class[cls]["tp"] += 1
                        matched_gt.add(best_gt_idx)
                    else:
                        per_class[cls]["fp"] += 1

                for gi, gt in enumerate(gt_boxes):
                    if gi not in matched_gt:
                        per_class[gt["class_id"]]["fn"] += 1

    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    per_class_rows = []
    idx2class = {i: name for i, name in enumerate(CLASSES)} if CLASSES is not None else {}

    for cls_idx, stats in per_class.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        cls_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        cls_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cls_f1 = (2 * cls_precision * cls_recall / (cls_precision + cls_recall)) if (cls_precision + cls_recall) > 0 else 0.0

        per_class_rows.append({
            "class_id": cls_idx,
            "class_name": idx2class.get(cls_idx, str(cls_idx)),
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": cls_precision,
            "recall": cls_recall,
            "f1": cls_f1,
        })

    per_class_df = pd.DataFrame(per_class_rows).sort_values("f1", ascending=False).reset_index(drop=True)

    return {
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "per_class_df": per_class_df,
    }


def train_finetune_layerwise_f1(
    model,
    raw_model,
    train_loader,
    val_loader,
    S, B, C,
    BATCH_SIZE,
    EPOCHS,
    LR_HEAD,
    LR_BACKBONE,
    WEIGHT_DECAY,
    LAMBDA_BOX,
    LAMBDA_NOOBJ,
    RUN_NAME,
    CONF_THRESH=0.5,
    NMS_IOU_THRESH=0.45,
    F1_IOU_MATCH_THRESH=0.5,
):
    print("Using device:", DEVICE)

    criterion = YOLOLoss(
        S=S, B=B, C=C,
        lambda_box=LAMBDA_BOX,
        lambda_noobj=LAMBDA_NOOBJ
    )

    optimizer = optim.Adam([
        {"params": raw_model.backbone[-4:].parameters(), "lr": LR_BACKBONE},
        {"params": raw_model.head.parameters(), "lr": LR_HEAD},
    ], weight_decay=WEIGHT_DECAY)

    best_val_f1 = float("-inf")
    best_val_map = float("-inf")
    best_val_map_50_95 = None
    best_val_loss_at_best = float("inf")
    best_state = None
    best_epoch = 0

    patience = 5
    no_improve = 0
    eps = 1e-12

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        train_box = 0.0
        train_obj = 0.0
        train_noobj = 0.0
        train_cls = 0.0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train"):
            imgs = imgs.to(DEVICE)
            targets = targets.to(DEVICE)

            optimizer.zero_grad()
            preds = model(imgs)
            total, box_l, obj_l, noobj_l, cls_l = criterion(preds, targets)
            total.backward()
            optimizer.step()

            train_loss += total.item()
            train_box += box_l.item()
            train_obj += obj_l.item()
            train_noobj += noobj_l.item()
            train_cls += cls_l.item()

        n_train = len(train_loader)
        train_loss /= n_train
        train_box /= n_train
        train_obj /= n_train
        train_noobj /= n_train
        train_cls /= n_train

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} ValLoss"):
                imgs = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds = model(imgs)
                total, _, _, _, _ = criterion(preds, targets)
                val_loss += total.item()

        val_loss /= len(val_loader)

        val_map, val_map_50_95, _ = _evaluate_map(
            model=model,
            loader=val_loader,
            S=S,
            B=B,
            C=C,
            conf_thresh=CONF_THRESH,
            iou_thresh=NMS_IOU_THRESH,
        )

        f1_metrics = evaluate_detection_f1(
            model=model,
            loader=val_loader,
            S=S,
            B=B,
            C=C,
            conf_thresh=CONF_THRESH,
            iou_thresh=NMS_IOU_THRESH,
            match_iou_thresh=F1_IOU_MATCH_THRESH,
        )
        val_f1 = f1_metrics.get("f1")
        val_precision = f1_metrics.get("precision")
        val_recall = f1_metrics.get("recall")

        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val/mAP": val_map,
                "val/mAP_50_95": val_map_50_95,
                "val/F1": val_f1,
                "val/precision": val_precision,
                "val/recall": val_recall,
                "box_loss": train_box,
                "obj_loss": train_obj,
                "noobj_loss": train_noobj,
                "cls_loss": train_cls,
            })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_mAP={val_map} | "
            f"val_F1={val_f1} | "
            f"val_precision={val_precision} | "
            f"val_recall={val_recall}"
        )

        improved = False
        if (
            best_state is None
            or val_f1 > best_val_f1 + eps
            or (
                abs(val_f1 - best_val_f1) <= eps
                and val_map is not None
                and val_map > best_val_map + eps
            )
            or (
                abs(val_f1 - best_val_f1) <= eps
                and (val_map is None or abs(val_map - best_val_map) <= eps)
                and val_loss < best_val_loss_at_best
            )
        ):
            improved = True

        if improved:
            best_val_f1 = val_f1
            best_val_map = val_map if val_map is not None else best_val_map
            best_val_map_50_95 = val_map_50_95
            best_val_loss_at_best = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(raw_model.state_dict())

            print(
                f"Best model at epoch {epoch} | "
                f"val/F1={val_f1} | val/mAP={val_map} | val_loss={val_loss:.4f}"
            )
        else:
            no_improve += 1
            print(f"No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("best_state was never set.")

    raw_model.load_state_dict(best_state)

    print(
        f"Restored best model from epoch {best_epoch} | "
        f"best val/F1={best_val_f1} | "
        f"best val/mAP={best_val_map} | "
        f"best val/mAP_50_95={best_val_map_50_95} | "
        f"val_loss_at_best={best_val_loss_at_best:.4f}"
    )

    if wandb.run is not None:
        wandb.summary["selection_metric"] = "val/F1"
        wandb.summary["best_epoch_by_val_F1"] = best_epoch
        wandb.summary["best_val_F1"] = float(best_val_f1)
        wandb.summary["best_val_mAP_at_best_F1"] = None if best_val_map == float("-inf") else float(best_val_map)
        wandb.summary["best_val_mAP_50_95_at_best_F1"] = None if best_val_map_50_95 is None else float(best_val_map_50_95)
        wandb.summary["val_loss_at_best_epoch"] = float(best_val_loss_at_best)

    return raw_model