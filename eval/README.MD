# Evaluation Folder

This folder contains notebooks for evaluating different components of the wildlife detection/classification workflow — from individual stages (detector or classifier) to the full two-stage pipeline — as well as single-stage models.  
All evaluations use COCO-style ground truth annotations and generate detailed metrics, per-class results, and confusion matrices.

---

## Notebooks

### 1. `evaluation_detector.ipynb`
- Evaluates **MegaDetector v6** object detection performance.
- Metrics: mAP@50, mAP@50–95, precision, recall, F1-score, per-class AP.
- Includes visual comparisons of predicted vs. ground truth boxes.

### 2. `evaluation_classifier.ipynb`
- Evaluates the **classification stage** using crops from MegaDetector outputs.
- Runs both **ConvNeXt** and **EfficientNet** classifiers.
- Metrics: per-class precision, recall, F1-score, accuracy, weighted/macro averages.
- Outputs confusion matrices and per-class reports.

### 3. `evaluation_pipeline.ipynb`
- Evaluates the **full two-stage pipeline**: MegaDetector → ConvNeXt classifier.
- Measures end-to-end detection + classification accuracy.
- Computes per-class accuracies, weighted/macro F1, and matching rates.
- Saves confusion matrices, classification reports, and unmatched predictions.

### 4. `evaluation_singleStage.ipynb`
- Evaluates **single-stage** YOLOv8 and MegaDetectorV6 models (direct detection + classification).
- Metrics: mAP, per-class accuracy, confusion matrices.

---

## Key Output Folders
- `classifier_stage/` — Classifier evaluation results.
- `detector_stage/` — MegaDetector evaluation results.
- `pipeline_results/` — Full pipeline evaluation outputs.
- `single_stage/` — Single-stage model evaluation outputs.

---

## CSV References
- `val_cls_merged.csv` — Ground truth + classifier predictions (validation set).
- `val_crop_index.csv` — Mapping between original images and classification crops.
- `val_gt_metadata.csv` — Simplified ground truth metadata.
- `val_preds_with_conf.csv` — Classifier predictions with confidence scores.
