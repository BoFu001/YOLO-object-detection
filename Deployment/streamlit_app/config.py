from pathlib import Path
import torch

DEPLOYMENT_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DEPLOYMENT_DIR.parent

# YOLOv1 defaults used in this coursework
S = 7
B = 2
C = 20
INPUT_SIZE = 448

# Select architecture variant that matches your checkpoint
MODEL_VARIANT = "yolov1_finetune"  # one of: yolov1, yolov1_dropout, yolov1_finetune
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "best_model.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PASCAL_VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor",
]
