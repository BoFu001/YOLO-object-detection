# Deployment (Local Streamlit App)

This folder contains all Streamlit website/inference files so the core coursework structure stays separate.

## Files
- `app.py` - Streamlit entrypoint
- `streamlit_app/` - config, model loading, inference helpers
- `requirements.txt` - minimal dependencies for this app

## Run
From repo root:

```bash
pip install -r Deployment/requirements.txt
streamlit run Deployment/app.py
```

## Notes
- Uses local checkpoint files only.
- No cloud APIs or external model services.
- Default checkpoint path points to `checkpoints/best_model.pth` in the project root.
