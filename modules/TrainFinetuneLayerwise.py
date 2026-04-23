import copy
import io
import re
from contextlib import redirect_stdout

import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from modules.Evaluation import evaluate
from modules.Loss import YOLOLoss
from my_config import DEVICE


def _parse_map_metrics(eval_text: str):
    """Extract mAP@0.50 and mAP@0.50:0.95 from evaluate() stdout."""
    map50_match = re.search(r"mAP@0\.50:\s*([0-9]*\.?[0-9]+)", eval_text)
    map5095_match = re.search(r"mAP@0\.50:0\.95:\s*([0-9]*\.?[0-9]+)", eval_text)

    map50 = float(map50_match.group(1)) if map50_match else None
    map5095 = float(map5095_match.group(1)) if map5095_match else None
    return map50, map5095


def _evaluate_map(model, loader, S, B, C, conf_thresh, iou_thresh):
    """Run evaluate() on the validation loader and capture printed mAP values."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        _ = evaluate(
            model=model,
            test_loader=loader,
            S=S, B=B, C=C,
            conf_thresh=float(conf_thresh),
            iou_thresh=float(iou_thresh),
        )
    eval_text = buffer.getvalue()
    val_map, val_map_50_95 = _parse_map_metrics(eval_text)
    return val_map, val_map_50_95, eval_text


def train_finetune_layerwise(
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
):
    """
    Train a fine-tuning run and select the best epoch by validation mAP@0.50,
    not by validation loss.

    Notes:
    - val_loss is still logged for monitoring.
    - Early stopping is driven by val/mAP.
    - If val/mAP cannot be parsed for an epoch, it falls back to val_loss for
      that epoch only, so training does not crash.
    """
    print("Using device:", DEVICE)

    criterion = YOLOLoss(
        S=S, B=B, C=C,
        lambda_box=LAMBDA_BOX,
        lambda_noobj=LAMBDA_NOOBJ
    )

    optimizer = optim.Adam([
        {'params': raw_model.backbone[-4:].parameters(), 'lr': LR_BACKBONE},
        {'params': raw_model.head.parameters(),           'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    best_val_map = float("-inf")
    best_val_map_50_95 = None
    best_val_loss_at_best = float("inf")
    best_state = None
    best_epoch = 0
    patience = 10
    no_improve = 0
    eps = 1e-12

    for epoch in range(1, EPOCHS + 1):
        # -----------------
        # Train
        # -----------------
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

        # -----------------
        # Validation loss
        # -----------------
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

        # -----------------
        # Validation mAP
        # -----------------
        val_map, val_map_50_95, eval_text = _evaluate_map(
            model=model,
            loader=val_loader,
            S=S, B=B, C=C,
            conf_thresh=CONF_THRESH,
            iou_thresh=NMS_IOU_THRESH,
        )

        # -----------------
        # W&B log
        # -----------------
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val/mAP": val_map,
                "val/mAP_50_95": val_map_50_95,
                "box_loss": train_box,
                "obj_loss": train_obj,
                "noobj_loss": train_noobj,
                "cls_loss": train_cls,
            })

        print(
            f"Epoch {epoch:03d}/{EPOCHS} | "
            f"train: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"val_mAP: {val_map} | "
            f"val_mAP_50_95: {val_map_50_95}"
        )

        # -----------------
        # Best epoch / early stopping
        # Primary criterion: highest val/mAP
        # Tiebreaker: lower val_loss
        # Fallback: if mAP parse failed, use val_loss only
        # -----------------
        improved = False

        if val_map is not None:
            if (
                best_state is None
                or val_map > best_val_map + eps
                or (abs(val_map - best_val_map) <= eps and val_loss < best_val_loss_at_best)
            ):
                improved = True
        else:
            if best_state is None or val_loss < best_val_loss_at_best:
                improved = True

        if improved:
            best_val_map = val_map if val_map is not None else best_val_map
            best_val_map_50_95 = val_map_50_95
            best_val_loss_at_best = val_loss
            best_epoch = epoch
            no_improve = 0
            best_state = copy.deepcopy(raw_model.state_dict())

            if val_map is not None:
                print(f"Best model at epoch {epoch} | val/mAP: {val_map} | val_loss: {val_loss:.4f}")
            else:
                print(f"Best model at epoch {epoch} by fallback val_loss: {val_loss:.4f}")
        else:
            no_improve += 1
            print(f"No improvement {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    if best_state is None:
        raise RuntimeError("best_state was never set. Training may have failed before first validation pass.")

    raw_model.load_state_dict(best_state)

    print(
        f"Restored best model from epoch {best_epoch} | "
        f"best val/mAP: {best_val_map} | "
        f"best val/mAP_50_95: {best_val_map_50_95} | "
        f"val_loss at best epoch: {best_val_loss_at_best:.4f}"
    )

    if wandb.run is not None:
        wandb.summary["selection_metric"] = "val/mAP"
        wandb.summary["best_epoch_by_val_mAP"] = best_epoch
        wandb.summary["best_val_mAP"] = None if best_val_map == float("-inf") else float(best_val_map)
        wandb.summary["best_val_mAP_50_95"] = (
            None if best_val_map_50_95 is None else float(best_val_map_50_95)
        )
        wandb.summary["val_loss_at_best_epoch"] = float(best_val_loss_at_best)

    return raw_model
