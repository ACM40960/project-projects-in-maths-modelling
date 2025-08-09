#!/usr/bin/env python
"""
Rescale bounding-boxes to match the downsized JPEGs and
write fixed COCO JSONs for all splits.

• input  JSONs :  ../data/annotations/<split>_annotations.json
• input  JPEGs :  ../data/images/<file_name>.jpg
• output JSONs :  ../data/preprocessed/annotations/<split>_fixed.json

Notes
-----
* “Empty” frames (category_id == 30) keep their annotation record,
  but have no 'bbox' key — these are copied unchanged.
* Width/height are updated for *every* image, even empty ones.
"""

import json, os, random, pathlib, tqdm
from collections import defaultdict
from PIL import Image, ImageOps
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# ------------------------------------------------------------------
RAW_ANNO_DIR = pathlib.Path("data/annotations")
RAW_IMG_DIR  = pathlib.Path("data/images")
OUT_DIR      = pathlib.Path("data/preprocessed/annotations")
OUT_DIR.mkdir(parents=True, exist_ok=True)

split_files = {
    "train"      : RAW_ANNO_DIR / "train_annotations.json",
    "cis_val"    : RAW_ANNO_DIR / "cis_val_annotations.json",
    "cis_test"   : RAW_ANNO_DIR / "cis_test_annotations.json",
    "trans_val"  : RAW_ANNO_DIR / "trans_val_annotations.json",
    "trans_test" : RAW_ANNO_DIR / "trans_test_annotations.json",
}
# ------------------------------------------------------------------

def rescale_split(json_in: pathlib.Path, json_out: pathlib.Path):
    coco = json.loads(json_in.read_text())

    # group annotations by image_id
    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_img[ann["image_id"]].append(ann)

    # iterate images, compute scale, update
    for img in tqdm.tqdm(coco["images"], desc=json_in.stem):
        file_path = RAW_IMG_DIR / img["file_name"]
        rgb = ImageOps.exif_transpose(Image.open(file_path))
        w_disk, h_disk = rgb.size
        sx, sy = w_disk / img["width"], h_disk / img["height"]

        # update image record
        img["width"], img["height"] = w_disk, h_disk

        # scale boxes for that image
        for ann in anns_by_img.get(img["id"], []):
            if "bbox" in ann and ann["bbox"]:          # skip true empty labels
                x, y, w, h = ann["bbox"]
                ann["bbox"] = [x*sx, y*sy, w*sx, h*sy]
                if "area" in ann:
                    ann["area"] = ann["area"] * sx * sy

    json_out.write_text(json.dumps(coco))
    print(f"wrote {json_out}")

# ------------------------------------------------------------------
# run for every split
for split, in_path in split_files.items():
    out_path = OUT_DIR / f"{split}_fixed.json"
    rescale_split(in_path, out_path)

# ------------------------------------------------------------------
# quick sanity-check: draw 5 random annotated images from train_fixed


train_fixed = json.loads((OUT_DIR / "train_fixed.json").read_text())
cat_id2name = {c["id"]: c["name"] for c in train_fixed["categories"]}

# build ann index
ann_by_img = defaultdict(list)
for ann in train_fixed["annotations"]:
    ann_by_img[ann["image_id"]].append(ann)

annotated_ids = [iid for iid, flist in ann_by_img.items()
                 if any("bbox" in a and a["bbox"] for a in flist)]
sample_ids = random.sample(annotated_ids, 5)

def show(img_id):
    meta = next(im for im in train_fixed["images"] if im["id"] == img_id)
    rgb  = np.array(Image.open(RAW_IMG_DIR / meta["file_name"]))
    plt.figure(figsize=(6,4)); plt.imshow(rgb); ax=plt.gca()
    for ann in ann_by_img[img_id]:
        if "bbox" not in ann or not ann["bbox"]:
            continue
        x, y, w, h = ann["bbox"]
        ax.add_patch(Rectangle((x,y), w,h, ec="lime", fc="none", lw=2))
        ax.text(x, y-4, cat_id2name[ann["category_id"]],
                color="lime", fontsize=8, backgroundcolor="black")
    ax.set_title(f"{img_id}"); ax.axis("off"); plt.show()

for mid in sample_ids:
    show(mid)
