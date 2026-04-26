# YOLO Object Detection on Pascal VOC 2012

## Project Overview
Implementation of a YOLOv1-style object detection model using a pretrained VGG16 backbone, trained and evaluated on the Pascal VOC 2012 dataset with 20 object categories.

**Course:** INM705 Deep Learning for Image Analysis  
**Institution:** City St George's, University of London  
**Authors:** Bo Fu, Yehoshua Perez Condori  

---

## Important Note About This Repository
This repository is focused primarily on the **later-stage experiments** of the coursework pipeline, especially:
- **Experiment 8:** Bayesian hyperparameter tuning with Weights & Biases Sweeps
- **Experiment 9:** final model selection and seeded retraining
- **Experiment 10:** post-inference threshold tuning and final qualitative evaluation

As a result, **Experiments 1-7 are documented in the report and reflected in the project history/results, but they are not the main runnable focus of the current repository state**. The notebook and supporting files were cleaned and organised mainly around the later experiments, where most of the final selection, tuning, checkpoint comparison, and inference work took place.

In practice, this means:
- the repo is **not intended to replay the full coursework chronologically from Experiment 1 onward**
- the most relevant runnable sections are the ones related to **Experiments 8-10**
- earlier experiments are included mainly for **context, reporting, and comparison**, rather than as the primary execution path

This was done intentionally so the repository better reflects the **final model-selection workflow** used for submission, rather than keeping every earlier exploratory stage as the main entry point.

---

## Links
- **Wandb:** https://wandb.ai/bofu001-/YOLO-VOC2012
- **Colab Notebook:** https://drive.google.com/file/d/182m9Fadqzu_SJAwi9HJrPFqUUiMgEdhU/view?usp=sharing
- **Kaggle Notebook:** https://www.kaggle.com/code/bofu001/yolo-object-detection
- **Dataset:** https://www.kaggle.com/datasets/huanghanchina/pascal-voc-2012

---

## Project Structure
```bash
coursework/
├── config.py                        # paths, device, seed, classes
├── requirements.txt                 # dependencies
├── YOLO-object-detection.ipynb      # main notebook (focused mainly on later experiments)
├── checkpoints/                     # saved model weights
│   ├── exp1_YOLOv1_lr1e-3.pth
│   ├── exp2_YOLOv1_lr1e-4.pth
│   ├── exp3_Dropout_lr1e-4.pth
│   ├── exp4_Finetune_lr1e-4.pth
│   ├── exp5_Finetune_lrH1e-4_lrB1e-5.pth
│   ├── exp6_Finetune_lrH1e-4_lrB5e-5.pth
│   └── exp7_Finetune_Aug_lr1e-4.pth
└── modules/
    ├── Dataset.py                   # VOCDataset, data augmentation, dataloaders
    ├── Loss.py                      # YOLOv1 loss function
    ├── Train.py                     # training loop (no early stopping)
    ├── TrainFinetune.py             # training loop with early stopping
    ├── TrainFinetuneLayerwise.py    # training loop with layer-wise LR + early stopping
    ├── Evaluation.py                # mAP evaluation
    ├── Inference.py                 # inference and visualisation
    └── Models/
        ├── YOLOv1.py                # frozen VGG16 backbone + head
        ├── YOLOv1Dropout.py         # frozen VGG16 + dropout in head
        └── YOLOv1Finetune.py        # unfrozen last 2 VGG16 layers + dropout
```

---

## Dataset

**Pascal VOC 2012** - 20 object categories, 11,540 images.

Download from Kaggle:
```bash
kaggle datasets download -d huanghanchina/pascal-voc-2012
```

Place the dataset in:
```bash
coursework/data/VOC2012/
├── JPEGImages/
├── Annotations/
└── ImageSets/Main/
    ├── train.txt
    └── val.txt
```

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Training / Notebook Usage
Open and run `YOLO-object-detection.ipynb` locally or on Colab/Kaggle.

### Current execution focus
The notebook is primarily organised around the later coursework stages:
- checkpoint comparison
- sweep-based model selection
- seeded retraining
- threshold tuning
- final inference and qualitative evaluation

### Earlier experiments
Experiments 1-7 are still part of the overall project and are summarised below for completeness, but they are **not the main execution path of the current repository version**.

| Experiment | Model | LR | Notes | Best Val Loss | mAP@0.50 | mAP@0.50:0.95 |
|-----------|-------|----|-------|---------------|----------|---------------|
| exp1 | YOLOv1 | 1e-3 | Baseline, 5 epochs | 2.4655 | 0.1048 | 0.0221 |
| exp2 | YOLOv1 | 1e-4 | Lower LR, 20 epochs | 2.4579 | 0.0961 | 0.0204 |
| exp3 | YOLOv1Dropout | 1e-4 | Dropout p=0.5 | 2.3808 | 0.0935 | 0.0200 |
| exp4 | YOLOv1Finetune | 1e-4 | Unfreeze last 2 VGG layers + ES | 2.2294 | 0.1208 | 0.0284 |
| exp5 | YOLOv1Finetune | lrH=1e-4, lrB=1e-5 | Layer-wise LR + ES | 2.2649 | 0.1116 | 0.0245 |
| exp6 | YOLOv1Finetune | lrH=1e-4, lrB=5e-5 | Layer-wise LR tuning + ES | 2.2452 | **0.1304** | **0.0284** |
| exp7 | YOLOv1Finetune | 1e-4 | ColorJitter augmentation + ES | 2.1851 | 0.1274 | 0.0272 |

**Best early-stage model: exp6** (mAP@0.50 = 0.1304)

---

## Later Experiments (Main Repository Focus)

### Experiment 8
Bayesian hyperparameter tuning with W&B Sweeps over:
- learning rates
- weight decay
- loss weights
- dropout
- confidence threshold

### Experiment 9
Final model selection using the chosen sweep configuration, followed by seeded retraining to check reproducibility and stability.

### Experiment 10
Post-inference threshold tuning over confidence and NMS IoU thresholds to improve final validation/test-time inference behaviour.

---

## Evaluation
Evaluation uses `torchmetrics.detection.MeanAveragePrecision`:

- **mAP@0.50**: IoU threshold = 0.50
- **mAP@0.50:0.95**: IoU thresholds from 0.50 to 0.95

---

## Inference
The final notebook sections load the selected later-stage checkpoint and run inference on held-out test images, displaying:
- ground truth boxes in green
- predicted boxes in red
- confidence scores for detections

---

## Reproducibility
- Random seed fixed at `SEED = 42`
- Val/Test split uses fixed `Subset` indexing (no `random_split`)
- `torch.backends.cudnn.deterministic = True`

---

## Platform Support
`config.py` automatically detects the environment:

| Platform | Data Path | Checkpoint Path |
|----------|-----------|-----------------|
| Local | data/VOC2012/ | checkpoints/ |
| Kaggle | /kaggle/input/ | /kaggle/working/ |
| Colab  | Google Drive | Google Drive |
