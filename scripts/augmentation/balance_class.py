#!/usr/bin/env python3
"""
balance_class.py  –  Offline oversampler for camera‑trap tails
----------------------------------------------------------------
*  Crops animals from donor frames and pastes them onto (usually empty)
   backgrounds with a light feathered mask.
*  Minimal rule‑set – no Poisson, no Grab‑Cut halos.
   • ±8° rotation + optional flip
   • overlap check against existing boxes (IoU > 0.25 ⇒ reject)
   • Domain harmonisation: if crop is IR/greyscale and background is colour
     (or vice‑versa), **convert both regions to greyscale** before blending.
*  Helper `preview_synth(num)` visualises synthetic frames with red boxes.
"""

import json, random, shutil, uuid
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

# ───────────── Config ─────────────
SOURCE_JSON  = Path("../../data/preprocessed/annotations/cleaned/train_fixed.json")
SOURCE_DIR   = Path("../../data/images")
TARGET_DIR   = Path("../../data/images/train_balanced")
TARGET_JSON  = Path("../../data/preprocessed/annotations/cleaned/train_balanced.json")

TARGET_COUNT       = 200
EMPTY_PROB         = 0.6
OVERLAP_THRESHOLD  = 0.25
MAX_TRIES_POS      = 60

MIN_DONOR_SIDE     = 64
MIN_LAPLACIAN_VAR  = 40

random.seed(0); np.random.seed(0)
TARGET_DIR.mkdir(parents=True, exist_ok=True)

# ───────────── Helpers ─────────────

def is_grayish(img, thresh: int = 10) -> bool:
    """Return True if R≈G≈B across image → likely IR / greyscale."""
    if img.ndim == 2:  # already single channel
        return True
    r,g,b = img[...,0], img[...,1], img[...,2]
    return (np.abs(r-g).mean() < thresh) and (np.abs(r-b).mean() < thresh)

def light_blur_mask(h: int, w: int, core_ratio: float = 0.85) -> np.ndarray:
    """
    Smaller opaque core (85 %) + wider, softer rim.
    31×31 kernel with σ≈13 gives a much smoother fade-out.
    """
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(
        mask,
        (int(w*(1-core_ratio)/2), int(h*(1-core_ratio)/2)),
        (int(w*(1+core_ratio)/2), int(h*(1+core_ratio)/2)),
        255, -1
    )
    return cv2.GaussianBlur(mask, (31, 31), 13)


def calc_iou(b1, b2):
    x1,y1,w1,h1 = b1; x2,y2,w2,h2 = b2
    xa,ya = max(x1,x2), max(y1,y2)
    xb,yb = min(x1+w1,x2+w2), min(y1+h1,y2+h2)
    inter = max(0, xb-xa)*max(0, yb-ya)
    union = w1*h1 + w2*h2 - inter
    return inter/union if union else 0.

def random_transform(crop):
    if random.random() < 0.5:
        crop = cv2.flip(crop, 1)
    ang = random.uniform(-8, 8)
    if abs(ang) > 1:
        h, w = crop.shape[:2]
        M = cv2.getRotationMatrix2D((w//2, h//2), ang, 1)
        crop = cv2.warpAffine(crop, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    return crop

# ───────────── Load metadata ─────────────
with SOURCE_JSON.open() as f:
    coco = json.load(f)
imgs = {im["id"]: im for im in coco["images"]}
anns = coco["annotations"]
cats = {c["id"]: c["name"] for c in coco["categories"]}

by_img, by_cat, counts = defaultdict(list), defaultdict(list), defaultdict(int)
for a in anns:
    by_img[a["image_id"]].append(a)
    by_cat[a["category_id"]].append(a)
    counts[a["category_id"]] += 1

empty_ids = {i for i in imgs if i not in by_img}
all_ids   = list(imgs)

if not any(TARGET_DIR.iterdir()):
    for m in imgs.values():
        shutil.copy2(SOURCE_DIR/m["file_name"], TARGET_DIR/m["file_name"])
    print("✓ originals copied")

new_imgs, new_anns = [], []

# ───────────── Oversample ─────────────
for cid, name in cats.items():
    need = max(0, TARGET_COUNT - counts[cid])
    print(f"{name:<15} need {need}")
    tries = 0
    while need and tries < need * 10:
        tries += 1
        donor_ann = random.choice(by_cat[cid])
        src_meta  = imgs[donor_ann["image_id"]]
        src_img   = np.array(Image.open(SOURCE_DIR/src_meta["file_name"]).convert("RGB"))
        x,y,w,h   = map(int, donor_ann["bbox"])
        if min(w,h) < MIN_DONOR_SIDE:
            continue
        crop = src_img[y:y+h, x:x+w]
        if cv2.Laplacian(crop, cv2.CV_64F).var() < MIN_LAPLACIAN_VAR:
            continue
        crop = random_transform(crop)
        ch, cw = crop.shape[:2]

        # pick background
        bg_id = random.choice(list(empty_ids)) if random.random()<EMPTY_PROB else random.choice(all_ids)
        bg_meta = imgs[bg_id]
        bg      = np.array(Image.open(SOURCE_DIR/bg_meta["file_name"]).convert("RGB"))
        H, W    = bg.shape[:2]

        # domain harmonisation
        domain_mismatch = is_grayish(crop) != is_grayish(bg)
        if domain_mismatch:
            # convert BOTH regions to greyscale (keep 3‑ch)
            crop = cv2.cvtColor(cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            bg   = cv2.cvtColor(cv2.cvtColor(bg,   cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)

        # ensure crop fits on background – scale once if needed
        if cw >= W-4 or ch >= H-4:
            s = min((W-4)/cw, (H-4)/ch) * 0.9  # leave margin
            crop = cv2.resize(crop, (int(cw*s), int(ch*s)))
            ch, cw = crop.shape[:2]

        alpha = light_blur_mask(ch, cw).astype(float) / 255.0   # matches new size

        # find placement
        for _ in range(MAX_TRIES_POS):
            px = random.randint(0, W - cw - 1)
            py = random.randint(0, H - ch - 1)
            new_box = [px, py, cw, ch]

            clash = False
            for b in by_img.get(bg_id, []):
                # (A) classic IoU test
                if calc_iou(new_box, b["bbox"]) > OVERLAP_THRESHOLD:
                    clash = True; break

                # (B) NEW –  “percent-area” occlusion test
                xi, yi = max(px, b["bbox"][0]), max(py, b["bbox"][1])
                xa, ya = min(px+cw, b["bbox"][0]+b["bbox"][2]), min(py+ch, b["bbox"][1]+b["bbox"][3])
                inter_area = max(0, xa - xi) * max(0, ya - yi)

                if inter_area:            # overlap exists
                    if (inter_area / (cw*ch) > 0.4) or (inter_area / (b["bbox"][2]*b["bbox"][3]) > 0.3):
                        clash = True; break

            if clash:
                continue          # try another spot
            break
        else:
            continue  # no slot found


        # blend
        alpha = light_blur_mask(ch, cw).astype(float)/255.0
        bg_patch = bg[py:py+ch, px:px+cw]
        bg[py:py+ch, px:px+cw] = (alpha[...,None]*crop + (1-alpha[...,None])*bg_patch).astype(np.uint8)

       # ── save synthetic frame ─────────────────────────────
        fname  = f"aug_{name}_{uuid.uuid4().hex[:6]}.jpg"
        img_id = uuid.uuid4().hex            # generate the new frame-ID first

        # 1) copy any bboxes that already lived in the background frame
        for orig in by_img.get(bg_id, []):
            bb = orig["bbox"]
            new_anns.append({
                "id": uuid.uuid4().hex,
                "image_id": img_id,              # point to the NEW frame
                "category_id": orig["category_id"],
                "bbox": bb,
                "area": bb[2] * bb[3],
                "iscrowd": orig.get("iscrowd", 0),
            })
            counts[orig["category_id"]] += 1     # keep  per-class counter in sync

        # 2) write the synthetic image to disk
        Image.fromarray(bg).save(TARGET_DIR / fname, quality=95)
        new_imgs.append({**bg_meta, "id": img_id, "file_name": fname})

        # 3) add the annotation for the *pasted* crop 
        new_anns.append({
            "id": uuid.uuid4().hex,
            "image_id": img_id,
            "category_id": cid,
            "bbox": [px, py, cw, ch],
            "area": cw * ch,
            "iscrowd": 0,
        })
        counts[cid] += 1
        need       -= 1


# ───────────── Save JSON ─────────────
for a in anns:
    a.setdefault("area", a["bbox"][2]*a["bbox"][3])
    a.setdefault("iscrowd", 0)

balanced = {
    "images": list(imgs.values()) + new_imgs,
    "annotations": anns + new_anns,
    "categories": coco["categories"],
}
TARGET_JSON.parent.mkdir(parents=True, exist_ok=True)
TARGET_JSON.write_text(json.dumps(balanced, indent=2))
print(f" Synthetic: {len(new_imgs)} new images saved - {TARGET_JSON}")


# Preview
if __name__ == "__main__":
    num = 10
    """Show *num* random synthetic images with their bboxes."""
    import matplotlib.pyplot as plt
    synth = [m for m in balanced["images"] if m["file_name"].startswith("aug_")]
    if not synth:
        print("No synthetic images found — run generator first.")
    grid = random.sample(synth, min(num, len(synth)))
    cols = min(5, len(grid)); rows = (len(grid) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(3*cols, 3*rows))
    axes = np.atleast_1d(axes).ravel()
    im2ann = defaultdict(list)
    for a in balanced["annotations"]:
        if a["image_id"] in {m["id"] for m in synth}:
            im2ann[a["image_id"]].append(a)
    for ax, meta in zip(axes, grid):
        img = Image.open(TARGET_DIR / meta["file_name"]).convert("RGB")
        d   = ImageDraw.Draw(img)
        for b in im2ann[meta["id"]]:
            x,y,w,h = b["bbox"]
            d.rectangle([x,y,x+w,y+h], outline="red", width=2)
        ax.imshow(img); ax.axis("off")
    for ax in axes[len(grid):]:
        ax.axis("off")
    plt.tight_layout(); plt.show()