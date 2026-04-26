import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    """
    YOLOv1-style loss function.

    Computes three parts of loss:
    1. Box regression loss   (only for cells with objects)
    2. Objectness loss       (confidence = 1 where object exists)
    3. No-object loss        (confidence = 0 where no object)
    4. Classification loss   (only for cells with objects)

    Args:
        S (int): grid size
        B (int): number of boxes per cell
        C (int): number of classes
        lambda_box   (float): weight for box loss
        lambda_noobj (float): weight for no-object loss
    """

    def __init__(self, S, B, C, lambda_box, lambda_noobj):
        super().__init__()
        self.S = S
        self.B = B
        self.C = C
        self.lambda_box   = lambda_box    # box loss weight
        self.lambda_noobj = lambda_noobj  # no-object loss weight

        # BCEWithLogitsLoss: used for confidence and class prediction
        # reduction='none' means return loss for each element, not average
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

        # SmoothL1Loss: used for box regression (x, y, w, h)
        # more stable than MSE for bounding box regression
        self.smoothl1 = nn.SmoothL1Loss(reduction='none')

    def forward(self, pred, target):
        """
        Args:
            pred   (tensor): model output,  shape (N, S, S, B*5+C)
            target (tensor): ground truth,  shape (N, S, S, B*5+C)
        Returns:
            total_loss (tensor): scalar loss value
        """
        N = pred.size(0)

        # ── split predictions into boxes and classes ──
        # pred_boxes: (N, S, S, B, 5)  → x, y, w, h, conf
        pred_boxes = pred[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)
        # pred_cls: (N, S, S, C)
        pred_cls = pred[..., self.B * 5:]

        # ── apply sigmoid to x, y, w, h ──
        px = torch.sigmoid(pred_boxes[..., 0])  # x offset in cell
        py = torch.sigmoid(pred_boxes[..., 1])  # y offset in cell
        pw = torch.sigmoid(pred_boxes[..., 2])  # width
        ph = torch.sigmoid(pred_boxes[..., 3])  # height
        pconf = pred_boxes[..., 4]              # confidence (keep as logits for BCE)

        # ── split targets into boxes and classes ──
        tgt_boxes = target[..., :self.B * 5].view(N, self.S, self.S, self.B, 5)
        tx   = tgt_boxes[..., 0]  # target x
        ty   = tgt_boxes[..., 1]  # target y
        tw   = tgt_boxes[..., 2]  # target width
        th   = tgt_boxes[..., 3]  # target height
        tconf = tgt_boxes[..., 4] # target confidence (1 or 0)

        # ── create masks ──
        # obj_mask:   1 where object exists,    0 where no object
        # noobj_mask: 0 where object exists,    1 where no object
        obj_mask   = (tconf > 0.5).float()  # shape (N, S, S, B)
        noobj_mask = 1.0 - obj_mask

        # ── 1. box regression loss ──
        # only compute where object exists (multiply by obj_mask)
        box_pred = torch.stack([px, py, pw, ph], dim=-1)  # (N, S, S, B, 4)
        box_tgt  = torch.stack([tx, ty, tw, th], dim=-1)  # (N, S, S, B, 4)

        box_loss = self.smoothl1(box_pred, box_tgt).sum(dim=-1)  # (N, S, S, B)
        box_loss = (box_loss * obj_mask).sum() / (obj_mask.sum() + 1e-6)

        # ── 2. objectness loss (where object exists) ──
        bce_conf = self.bce(pconf, obj_mask)  # (N, S, S, B)
        obj_loss = (bce_conf * obj_mask).sum() / (obj_mask.sum() + 1e-6)

        # ── 3. no-object loss (where no object exists) ──
        noobj_loss = (bce_conf * noobj_mask).sum() / (noobj_mask.sum() + 1e-6)

        # ── 4. classification loss ──
        # only compute for cells that contain an object
        # use target[..., 4] (slot 0 confidence) to detect object presence per cell
        cell_obj = (target[..., 4] > 0.5).float()  # (N, S, S)
        cls_tgt  = target[..., self.B * 5:]         # (N, S, S, C)

        cls_loss = self.bce(pred_cls, cls_tgt)       # (N, S, S, C)
        cls_loss = cls_loss.sum(dim=-1)              # (N, S, S)
        cls_loss = (cls_loss * cell_obj).sum() / (cell_obj.sum() + 1e-6)

        # ── combine all losses ──
        total_loss = (
            self.lambda_box   * box_loss    # box regression
            + obj_loss                      # objectness
            + self.lambda_noobj * noobj_loss # no-object
            + cls_loss                      # classification
        )

        return total_loss, box_loss.detach(), obj_loss.detach(), noobj_loss.detach(), cls_loss.detach()