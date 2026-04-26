from pathlib import Path
import sys

import torch

from streamlit_app.config import PROJECT_ROOT, MODEL_VARIANT, S, B, C, DEVICE


def _import_model_classes():
    models_dir = PROJECT_ROOT / "modules" / "modules" / "Models"
    if not models_dir.exists():
        raise FileNotFoundError(
            f"Could not find coursework model directory: {models_dir}"
        )

    sys.path.insert(0, str(models_dir))
    from YOLOv1 import YOLOv1
    from YOLOv1Dropout import YOLOv1Dropout
    from YOLOv1Finetune import YOLOv1Finetune

    return {
        "yolov1": YOLOv1,
        "yolov1_dropout": YOLOv1Dropout,
        "yolov1_finetune": YOLOv1Finetune,
    }


def build_model(model_variant: str = MODEL_VARIANT):
    model_classes = _import_model_classes()
    if model_variant not in model_classes:
        raise ValueError(
            f"Unsupported MODEL_VARIANT='{model_variant}'. "
            f"Choose one of: {list(model_classes.keys())}"
        )
    return model_classes[model_variant](S=S, B=B, C=C).to(DEVICE)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: str | Path):
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location=DEVICE)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Checkpoint/model mismatch. Check MODEL_VARIANT and checkpoint compatibility. "
            f"Details: {exc}"
        ) from exc

    model.eval()
    return model
