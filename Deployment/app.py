from pathlib import Path

import streamlit as st
from PIL import Image, UnidentifiedImageError

from streamlit_app.config import CHECKPOINT_PATH, MODEL_VARIANT
from streamlit_app.model_loader import build_model, load_checkpoint
from streamlit_app.inference_utils import run_inference, draw_detections, detections_to_rows


st.set_page_config(page_title="Local YOLO Coursework Inference", layout="wide")
st.title("YOLO Coursework: Local Object Detection")
st.caption("Runs fully locally from a local .pth checkpoint (no external APIs).")


@st.cache_resource
def get_loaded_model(checkpoint_path: str, model_variant: str):
    model = build_model(model_variant=model_variant)
    return load_checkpoint(model, checkpoint_path)


st.sidebar.header("Inference Settings")
checkpoint_input = st.sidebar.text_input("Checkpoint path", value=str(CHECKPOINT_PATH))
model_variant = st.sidebar.selectbox(
    "Model variant",
    options=["yolov1", "yolov1_dropout", "yolov1_finetune"],
    index=["yolov1", "yolov1_dropout", "yolov1_finetune"].index(MODEL_VARIANT),
)
conf_thresh = st.sidebar.slider("Confidence threshold", 0.01, 0.90, 0.25, 0.01)
iou_thresh = st.sidebar.slider("NMS IoU threshold", 0.05, 0.90, 0.50, 0.01)

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

model = None
try:
    checkpoint_abs = str(Path(checkpoint_input).expanduser().resolve())
    model = get_loaded_model(checkpoint_abs, model_variant)
    st.success(f"Model loaded: {model_variant} | checkpoint: {checkpoint_abs}")
except FileNotFoundError as exc:
    st.error(f"Checkpoint missing: {exc}")
except RuntimeError as exc:
    st.error(str(exc))
except Exception as exc:
    st.error(f"Failed to load model: {exc}")

if uploaded is not None and model is not None:
    try:
        image = Image.open(uploaded).convert("RGB")
    except UnidentifiedImageError:
        st.error("Invalid image file. Please upload a valid image.")
        st.stop()

    st.image(image, caption="Input image", use_container_width=True)

    if st.button("Run inference", type="primary"):
        detections = run_inference(
            model=model,
            image=image,
            conf_thresh=conf_thresh,
            iou_thresh=iou_thresh,
        )

        if len(detections) == 0:
            st.warning("No detections found for current thresholds.")
        else:
            rendered = draw_detections(image, detections)
            st.image(rendered, caption=f"Detections ({len(detections)})", use_container_width=True)
            st.subheader("Detection Table")
            st.dataframe(detections_to_rows(detections), use_container_width=True)
