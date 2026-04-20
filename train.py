# train.py
# Root-level training entry script
# Run: python train.py

import os
import random
import argparse
import numpy as np
import torch

from config import *

# reproducibility
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# dataloader
from modules.Dataset import get_dataloaders

# model
from modules.Models.YOLOv1 import YOLOv1

# training function
from modules.Train import train




# default config
S = 7
B = 2
C = 20


WEIGHT_DECAY = 1e-4
LAMBDA_BOX = 5.0
LAMBDA_NOOBJ = 0.5



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=5,
                        help="number of training epochs")

    parser.add_argument("--lr", type=float, default=1e-3,
                        help="learning rate")

    parser.add_argument("--batch_size", type=int, default=16,
                        help="batch size")

    return parser.parse_args()
    
def main():
    print("=" * 60)
    print("YOLO Object Detection Training")
    print("=" * 60)

    args = parse_args()
    
    EPOCHS = args.epochs
    LR = args.lr
    BATCH_SIZE = args.batch_size

    RUN_NAME = f"YOLOv1_e{EPOCHS}_lr{LR}_bs{BATCH_SIZE}"
    CKPT_PATH = os.path.join(CKPT_DIR, RUN_NAME + ".pth")

    
    print(f"Device      : {DEVICE}")
    print(f"Checkpoint  : {CKPT_PATH}")
    print(f"Epochs      : {EPOCHS}")
    print(f"LR          : {LR}")
    print(f"Batch Size  : {BATCH_SIZE}")
    print("=" * 60)

    # ensure checkpoint directory exists before saving
    assert os.path.exists(CKPT_DIR), f"Checkpoint directory not found: {CKPT_DIR}"

    # dataloaders
    train_loader, val_loader, _ = get_dataloaders(BATCH_SIZE, S, B, C)

    # model
    model = YOLOv1(S=S, B=B, C=C).to(DEVICE)

    # multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    raw_model = model.module if isinstance(model, torch.nn.DataParallel) else model

    # train
    raw_model = train(
        model=model,
        raw_model=raw_model,
        train_loader=train_loader,
        val_loader=val_loader,
        S=S,
        B=B,
        C=C,
        BATCH_SIZE=BATCH_SIZE,
        EPOCHS=EPOCHS,
        LR=LR,
        WEIGHT_DECAY=WEIGHT_DECAY,
        LAMBDA_BOX=LAMBDA_BOX,
        LAMBDA_NOOBJ=LAMBDA_NOOBJ,
        RUN_NAME=RUN_NAME
    )

    # save checkpoint
    torch.save(raw_model.state_dict(), CKPT_PATH)
    print(f"Saved checkpoint: {CKPT_PATH}")


if __name__ == "__main__":
    main()