import os
import random
import numpy as np
import torch
import torch.optim as optim
import wandb
from tqdm import tqdm

from modules.Loss import YOLOLoss
# IMPORTANT: no DEVICE import anymore

# -------------------------
# DEVICE (FIXED HERE)
# -------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("USING DEVICE:", DEVICE)

# -------------------------
# CONFIG (self-contained)
# -------------------------
FINAL_RETRAIN_SEEDS = [42, 43, 44]

FINAL_SWEEP_CONFIG = {
    "LR_HEAD": 2.73e-2,
    "LR_BACKBONE": 8.1e-5,
    "WEIGHT_DECAY": 1e-4,
    "LAMBDA_BOX": 0.05,
    "LAMBDA_NOOBJ": 0.5,
    "DROPOUT_P": 0.3,
    "CONF_THRESH": 0.5,
    "BATCH_SIZE": 16,
    "AUGMENT": True,
    "EPOCHS": 20,
}

# -------------------------
# Seed control
# -------------------------
def set_all_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

# -------------------------
# TRAIN FUNCTION
# -------------------------
def train_finetune_layerwise(model, raw_model,
                              train_loader, val_loader,
                              S, B, C,
                              BATCH_SIZE, EPOCHS,
                              LR_HEAD, LR_BACKBONE,
                              WEIGHT_DECAY,
                              LAMBDA_BOX, LAMBDA_NOOBJ,
                              RUN_NAME):

    print("RUN DEVICE:", DEVICE)
    print("MODEL DEVICE:", next(model.parameters()).device)

    run = wandb.init(
        project="YOLO-VOC",
        name=RUN_NAME,
        config=FINAL_SWEEP_CONFIG
    )

    criterion = YOLOLoss(S=S, B=B, C=C,
                         lambda_box=LAMBDA_BOX,
                         lambda_noobj=LAMBDA_NOOBJ)

    optimizer = optim.Adam([
        {'params': raw_model.backbone[-4:].parameters(), 'lr': LR_BACKBONE},
        {'params': raw_model.head.parameters(), 'lr': LR_HEAD}
    ], weight_decay=WEIGHT_DECAY)

    best_val_loss = float("inf")
    best_state = None
    patience = 5
    no_improve = 0

    for epoch in range(EPOCHS):

        model.train()
        train_loss = 0

        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}"):

            # -------------------------
            # FORCE GPU USAGE HERE
            # -------------------------
            imgs = imgs.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            preds = model(imgs)
            loss, box, obj, noobj, cls = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # validation
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs = imgs.to(DEVICE, non_blocking=True)
                targets = targets.to(DEVICE, non_blocking=True)

                preds = model(imgs)
                loss, *_ = criterion(preds, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")

        # early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = raw_model.state_dict().copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    raw_model.load_state_dict(best_state)
    wandb.finish()

    return raw_model