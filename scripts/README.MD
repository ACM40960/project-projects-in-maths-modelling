# Scripts

This folder holds the code I actually used to build datasets and train models for camera‑trap images. I wrote this README so you can **understand each script without opening it**: what it does, why I did it that way, what it expects in/out, and how to run it.

Subfolders:

* **augmentation/** – data augmentation and offline balancing
* **dataset/** – preparing/patching/reshaping datasets
* **train/** – training scripts for detectors and classifiers

---

## Augmentation

### `augmentation/augment_cct.py`

**What it is:** My Albumentations pipeline tailored for camera traps. It tries to recreate real failure modes: IR/night shots, glare, motion blur (running animals), branches/leaves blocking the lens, fog/rain/dust, slight perspective changes, and different camera sensors.

**What it does (inside):**

* Groups transforms by **problem type** (illumination, geometric, weather, motion blur, occlusion, sensor variation, seasonal look, perspective, and a few advanced ops like sharpening/downscale).
* For detection it sets `A.BboxParams(format='yolo', label_fields=['cls'], min_visibility=0.2, filter_invalid_bboxes=True)` so labels stay valid.
* Lets me **schedule intensity** across epochs (light early → stronger later) to avoid over‑regularising at the start.

**Inputs → Outputs:**

* In: `image` (HWC, RGB), `bboxes` (YOLO format), `cls` (per‑box class list)
* Out: same fields, transformed; invalid boxes dropped safely

**How I use it (from training code):**

```python
from scripts.augmentation.augment_cct import get_detection_augmentation
pipe = get_detection_augmentation(mode='train', img_size=640, intensity=0.8)
aug = pipe(image=img, bboxes=bboxes, cls=labels)
```

---

### `augmentation/balance_class.py`

**What it is:** A simple **offline oversampler** to help long‑tail species. It copy‑pastes good animal crops onto (often empty) backgrounds with a **soft feathered mask**.

**What it does:**

* Picks a donor annotation (checks sharpness & minimum size), applies small rotation/flip, and scales to fit the background.
* **Collision control:** rejects placements if IoU with existing boxes > 0.25 or if the overlap covers too much of either object (percent‑area check).
* **Domain harmonisation:** if the crop is IR/gray and the background is RGB (or vice‑versa), both are converted to grayscale before blending to avoid odd color seams.
* Copies any original boxes from the background into the new synthetic image and adds the pasted animal box. Updates a balanced COCO JSON.

**Inputs → Outputs:**

* In: `train_fixed.json`, `data/images/`
* Out: `data/images/train_balanced/aug_*.jpg` + `train_balanced.json` (COCO)

**Run:**

```bash
python scripts/augmentation/balance_class.py
```

---

## Dataset

### `dataset/coco_to_yolo.py`

**What it is:** Converts my cleaned COCO JSONs into **YOLOv8 layout** per split and writes `names.txt`.

**What it does:**

* Builds a **stable class‑ID map** from `train_fixed.json` (so train/val/test stay aligned).
* Creates `images/` + `labels/` under each split. Labels are YOLO text files with normalised `cx cy w h`.
* Uses **symlinks by default** to avoid duplicating images; `--copy` toggles real copies if needed.

**Inputs → Outputs:**

* In: `data/preprocessed/annotations/cleaned/*.json`, images under `data/images/`
* Out: `data/yolo_images/<split>/images|labels/`, plus `data/yolo_images/names.txt`

**Run:**

```bash
python scripts/dataset/coco_to_yolo.py \
  --anno_dir data/preprocessed/annotations/cleaned \
  --img_root data/images \
  --out_root data/yolo_images \
  --copy   # optional
```

---

### `dataset/megadetector_dataset.py`

**What it is:** Makes a **binary YOLO dataset** (0=animal, 1=vehicle) for MegaDetector‑style training.

**What it does:**

* Symlinks images and rewrites each label line: if the original class is in `--vehicle-ids`, map to 1, else 0.
* Writes a new `names.txt` with two classes.

**Inputs → Outputs:**

* In: `data/yolo_images/<split>` structure
* Out: `data/megadetector_images/<split>` structure with remapped labels

**Run:**

```bash
python scripts/dataset/megadetector_dataset.py \
  --src data/yolo_images \
  --dst data/megadetector_images \
  --vehicle-ids 11
```

---

### `dataset/merge_val_splits.py`

**What it is:** Merge `cis_val_fixed.json` + `trans_val_fixed.json` into `val_all_fixed.json`.

**What it does:**

* Checks that `categories` match.
* If any annotation IDs collide, it renames the trans IDs as `tva_<id>`.

**Run:**

```bash
python scripts/dataset/merge_val_splits.py
```

---

### `dataset/move_verification.py`

**What it is:** Copies only the images referenced by `verify_test.json` into a flat `data/verification/` folder – handy for demos or quick notebooks.

**Run:**

```bash
python scripts/dataset/move_verification.py
```

---

### `dataset/patch_coco_fields.py`

**What it is:** Makes sure every annotation has COCO‑spec fields `area` and `iscrowd`. Safe to run multiple times.

**Default target folder:** `../../data/preprocessed/annotations/cleaned`

**Run:**

```bash
python scripts/dataset/patch_coco_fields.py                 # default dir
python scripts/dataset/patch_coco_fields.py some/other/dir  # custom dir
```

---

### `dataset/remove_empty.py`

**What it is:** Drops the `empty` class from all `*_fixed.json` and removes its annotations.

**Run:**

```bash
python scripts/dataset/remove_empty.py
```

---

### `dataset/rescale_bbox.py`

**What it is:** Recomputes bbox coordinates so they match the **actual JPEG sizes on disk** (EXIF/orientation can change dims). Also updates image `width/height` in JSON. Includes a tiny visual sanity check at the end.

**Run:**

```bash
python scripts/dataset/rescale_bbox.py
```

---

### `dataset/save_crops.py`

**What it is:** Saves **padded ground‑truth crops** per species into folders and logs a CSV‑like manifest with crop metadata.

**Run:**

```bash
python scripts/dataset/save_crops.py \
  --json data/preprocessed/annotations/cleaned/train_fixed.json \
  --img_root data/images \
  --split train
```

---

### `dataset/verification_set.py`

**What it is:** Creates small **verification subsets** (default 5% of images per class) from cis/trans test JSONs and writes `verify_cis_test.json`, `verify_trans_test.json`, and merged `verify_test.json`.

**Run:**

```bash
python scripts/dataset/verification_set.py
```

---

## Training

### `train/train_yolov8.py` — YOLOv8 with progressive Albumentations

**Goal:** Train a detector while **gradually increasing augmentation strength** and **freezing → unfreezing** the backbone for stability.

**What it does:**

* Wraps Ultralytics’ `YOLODataset` with my `ProgressiveAugmentationDataset` to rebuild the Albumentations pipeline each epoch and ramp intensity using a cosine schedule.
* Freezes backbone layers for the first N epochs, then **unfreezes** and **reduces LR 10×** for fine‑tuning.
* Builds a **weighted sampler** from label frequencies to show rare classes more often.
* Logs per‑class metrics and loss components (box/cls/dfl) to W\&B (offline mode).

**Run:**

```bash
python scripts/train/train_yolov8.py
```

---

### `train/train_megadetector.py` — MegaDetector v6 (binary animal/vehicle)

**Goal:** Train a YOLO‑family model on a **binary** dataset with moderate built‑in augmentations and TensorBoard logging.

**What it does:**

* Uses TensorBoard (W\&B disabled).
* Freezes the backbone then **unfreezes** mid‑training and drops LR for fine‑tuning.
* Adds an **image‑level weighted sampler** (weight = max of its classes), and logs per‑class P/R/AP + mAP\@50/95.

**Run:**

```bash
python scripts/train/train_megadetector.py
```

---

### `train/train_baseline.py` — YOLOv8 baseline (no extra aug)

**Goal:** Provide a clean reference: YOLOv8 with built‑in augs turned **off**.

**What it does:**

* Keeps training simple; logs total loss and the box/cls/dfl ratios each epoch.

**Run:**

```bash
python scripts/train/train_baseline.py
```

---

### `train/train_convnext.py` — ConvNeXt on ground‑truth crops

**Goal:** Train a species classifier on GT crops while helping tail classes.

**What it does:**

* **TailAwareFolder** gives stronger transforms to rare classes (<200 samples).
* **TailMixCollate** optionally mixes images when a tail class is in the batch (soft labels like MixUp).
* Uses **Class‑Balanced Focal Loss** (Cui et al.) and AdamW with higher LR on the classifier head.
* Saves best checkpoint, confusion matrix, classification report, and training curves.

**Run:**

```bash
python scripts/train/train_convnext.py
```

---

### `train/finetune_Convnext_md_crops.py` — ConvNeXt fine‑tune on MD crops

**Goal:** Start from the earlier ConvNeXt checkpoint and fine‑tune on a different crop set that contains a `background` class.

**What it does:**

* **Removes `background`** from both train/val and **remaps** label indices safely.
* Loads prior weights, lowers LR for the backbone, and trains the head a bit faster.
* Logs a **normalised** confusion matrix and a detailed classification report.

**Run:**

```bash
python scripts/train/finetune_Convnext_md_crops.py
```

---

### `train/train_efficientnet.py` — EfficientNetV2‑S + CBAM (3 layers)

**Goal:** Add lightweight attention into EfficientNetV2‑S by inserting **three CBAM modules** at the end stages and train on GT crops.

**What it does:**

* Finds the right channel sizes per stage and inserts `CbamModule` after the last 3 stages.
* Same tail‑aware idea as ConvNeXt (stronger augs for rare classes) + class‑balanced focal loss.
* Uses One‑Cycle LR with warmup; saves best checkpoint, report, curves, and confusion matrix.

**Run:**

```bash
python scripts/train/train_efficientnet.py
```

---

## Notes

* I fix seeds (`torch`, `numpy`, `random`) and control CuDNN flags for stability where it helps.
* I call `torch.cuda.empty_cache()` and `gc.collect()` occasionally to prevent memory fragmentation.
* Class names and split paths are read from YAML/JSON in `configs/model/` and `data/**` as referenced in each script.
* The scripts train_yolov8.py, train_megadetector.py, and train_efficientnet.py were reused and modified for multiple training runs. I didn’t create separate files for each run; the versions you see here are the last modified forms. The changes included:

    * YOLOv8: Adjusted augmentation schedules between runs.
    * EfficientNet: Ran both single‑CBAM and 3‑CBAM‑layer variants.
    * MegaDetector: Trained once on all classes (single stage) and once as a binary animal/vehicle detector.

