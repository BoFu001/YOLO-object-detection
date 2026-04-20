import torch
import torch.optim as optim
import wandb
import random
from tqdm import tqdm
from modules.Loss import YOLOLoss
from my_config import DEVICE

def train(model, raw_model, train_loader, val_loader, S, B, C, BATCH_SIZE, EPOCHS, LR, WEIGHT_DECAY, LAMBDA_BOX, LAMBDA_NOOBJ, RUN_NAME):


    # Print device info
    print("Using device:", DEVICE)

    
    # Start a new W&B run for this training script
    run = wandb.init(
        entity="y-benjamin_pc-city-st-george-s-university-of-london",
        project="YOLO-VOC",
        name=RUN_NAME,
        config={
            "learning_rate": LR,
            "architecture": "CNN",
            "dataset": "CIFAR-100",
            "epochs": EPOCHS,
            "S": S,
            "B": B,
            "C": C,
            "batch_size": BATCH_SIZE,
            "weight_decay": WEIGHT_DECAY,
            "lambda_box": LAMBDA_BOX,
            "lambda_noobj": LAMBDA_NOOBJ,
        },
    )


    criterion = YOLOLoss(S=S, B=B, C=C, lambda_box=LAMBDA_BOX, lambda_noobj=LAMBDA_NOOBJ)
    
    # use raw_model.head to access head parameters
    optimizer = optim.Adam(raw_model.head.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    # ── training loop ──
    for epoch in range(1, EPOCHS + 1):
        # ── train ──
        model.train()
        train_loss  = 0.0
        train_box   = 0.0
        train_obj   = 0.0
        train_noobj = 0.0
        train_cls   = 0.0
        
        for imgs, targets in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} Train"):
            imgs    = imgs.to(DEVICE)
            targets = targets.to(DEVICE)
            preds = model(imgs)
            total, box_l, obj_l, noobj_l, cls_l = criterion(preds, targets)
            optimizer.zero_grad()
            total.backward()
            optimizer.step()
            train_loss  += total.item()
            train_box   += box_l.item()
            train_obj   += obj_l.item()
            train_noobj += noobj_l.item()
            train_cls   += cls_l.item()
        
        n = len(train_loader)
        train_loss  /= n
        train_box   /= n
        train_obj   /= n
        train_noobj /= n
        train_cls   /= n
        
        # ── validation ──
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} Val"):
                imgs    = imgs.to(DEVICE)
                targets = targets.to(DEVICE)
                preds   = model(imgs)
                total, _, _, _, _ = criterion(preds, targets)
                val_loss += total.item()
        
        val_loss /= len(val_loader)
        
        # ── log to wandb ──
        wandb.log({
            "epoch":      epoch,
            "train_loss": train_loss,
            "val_loss":   val_loss,
            "box_loss":   train_box,
            "obj_loss":   train_obj,
            "noobj_loss": train_noobj,
            "cls_loss":   train_cls,
        })
        
        # ── print progress ──
        print(f"Epoch {epoch:03d}/{EPOCHS} | train: {train_loss:.4f} | val: {val_loss:.4f}")
    
    wandb.finish()
    
    return raw_model

    