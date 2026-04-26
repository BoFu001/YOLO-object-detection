from dataclasses import dataclass

from PIL import Image, ImageDraw
import torch
from torchvision import transforms
from torchvision.ops import nms

from streamlit_app.config import INPUT_SIZE, S, B, C, DEVICE, PASCAL_VOC_CLASSES


transform = transforms.Compose([
    transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


@dataclass
class Detection:
    class_idx: int
    class_name: str
    score: float
    x1: int
    y1: int
    x2: int
    y2: int


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0).to(DEVICE)


def decode_predictions(pred: torch.Tensor, conf_thresh: float):
    boxes, scores, labels = [], [], []
    cell_size = 1.0 / S

    for gy in range(S):
        for gx in range(S):
            for b_idx in range(B):
                conf = torch.sigmoid(pred[gy, gx, b_idx * 5 + 4])

                cx = (gx + torch.sigmoid(pred[gy, gx, b_idx * 5 + 0])) * cell_size
                cy = (gy + torch.sigmoid(pred[gy, gx, b_idx * 5 + 1])) * cell_size
                w = torch.sigmoid(pred[gy, gx, b_idx * 5 + 2])
                h = torch.sigmoid(pred[gy, gx, b_idx * 5 + 3])

                x1 = (cx - w / 2).clamp(0, 1)
                y1 = (cy - h / 2).clamp(0, 1)
                x2 = (cx + w / 2).clamp(0, 1)
                y2 = (cy + h / 2).clamp(0, 1)
                if x2 <= x1 or y2 <= y1:
                    continue

                cls_scores = torch.sigmoid(pred[gy, gx, B * 5 : B * 5 + C])
                cls_score, cls_idx = cls_scores.max(dim=0)
                score = conf * cls_score
                if score < conf_thresh:
                    continue

                boxes.append([x1.item(), y1.item(), x2.item(), y2.item()])
                scores.append(score.item())
                labels.append(cls_idx.item())

    if len(boxes) == 0:
        return (
            torch.zeros((0, 4), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.float32),
            torch.zeros((0,), dtype=torch.long),
        )

    return (
        torch.tensor(boxes, dtype=torch.float32),
        torch.tensor(scores, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.long),
    )


def run_inference(model: torch.nn.Module, image: Image.Image, conf_thresh: float, iou_thresh: float):
    image_tensor = preprocess_image(image)
    width, height = image.size

    with torch.no_grad():
        pred = model(image_tensor)[0].cpu()

    boxes, scores, labels = decode_predictions(pred, conf_thresh=conf_thresh)
    if boxes.shape[0] > 0:
        keep = nms(boxes, scores, iou_thresh)
        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    detections: list[Detection] = []
    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i]
        detections.append(
            Detection(
                class_idx=int(labels[i].item()),
                class_name=PASCAL_VOC_CLASSES[int(labels[i].item())],
                score=float(scores[i].item()),
                x1=int(x1.item() * width),
                y1=int(y1.item() * height),
                x2=int(x2.item() * width),
                y2=int(y2.item() * height),
            )
        )

    return detections


def draw_detections(image: Image.Image, detections: list[Detection]) -> Image.Image:
    drawn = image.convert("RGB").copy()
    canvas = ImageDraw.Draw(drawn)

    for det in detections:
        canvas.rectangle([(det.x1, det.y1), (det.x2, det.y2)], outline=(255, 0, 0), width=3)
        label = f"{det.class_name} {det.score:.2f}"
        text_anchor = (det.x1, max(0, det.y1 - 14))
        canvas.text(text_anchor, label, fill=(255, 0, 0))

    return drawn


def detections_to_rows(detections: list[Detection]) -> list[dict]:
    return [
        {
            "class": d.class_name,
            "confidence": round(d.score, 4),
            "x1": d.x1,
            "y1": d.y1,
            "x2": d.x2,
            "y2": d.y2,
        }
        for d in detections
    ]
