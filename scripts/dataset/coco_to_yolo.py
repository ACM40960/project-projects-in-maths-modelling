# -------------------------------------------------------------
# Create YOLO-v8 folders + labels for every split we use later:
#   train, val, cis_val, trans_val, cis_test, trans_test
# -------------------------------------------------------------
import json, os, shutil, argparse
from pathlib import Path
from collections import defaultdict

# ---------------- helper -------------------------------------------------
def build_id_map(coco):
    cats_sorted = sorted(c["id"] for c in coco["categories"])
    return {old: new for new, old in enumerate(cats_sorted)}

def convert_split(coco_path, img_root, out_split, id_map, copy=False):
    if (out_split / "images").exists():      # <-- early-exit
        print(f"✓ {out_split.name:11}  already present – skipped")
        return
    coco_path, img_root, out_split = map(Path, (coco_path, img_root, out_split))
    (out_split / "images").mkdir(parents=True, exist_ok=True)
    (out_split / "labels").mkdir(parents=True, exist_ok=True)

    coco = json.loads(coco_path.read_text())
    imgs = {im["id"]: im for im in coco["images"]}

    anns_by_img = defaultdict(list)
    for ann in coco["annotations"]:
        if "bbox" in ann:  # still ignore malformed ones
            anns_by_img[ann["image_id"]].append(ann)

    # iterate over *all* images in images[]
    for img_id, meta in imgs.items():
        lbl_fp = out_split / "labels" / f"{Path(meta['file_name']).stem}.txt"
        w_img, h_img = meta["width"], meta["height"]

        if img_id in anns_by_img:
            lines = []
            for ann in anns_by_img[img_id]:
                x, y, w, h = ann["bbox"]
                xc = (x + w / 2) / w_img
                yc = (y + h / 2) / h_img
                wn = w / w_img
                hn = h / h_img
                cls = id_map[ann["category_id"]]
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")
            lbl_fp.write_text("\n".join(lines))
        else:
            # Empty image — write empty file
            lbl_fp.touch()

        # Copy or symlink image
        dst_img = out_split / "images" / meta["file_name"]
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        src_img = img_root / meta["file_name"]
        if copy:
            shutil.copy2(src_img, dst_img)
        else:
            try:
                os.symlink(src_img.resolve(), dst_img)
            except OSError:
                shutil.copy2(src_img, dst_img)

    print(f" {out_split.name:11}  {len(imgs):,} images   "
          f"|  {sum(len(v) for v in anns_by_img.values()):,} boxes")


# ---------------- main ---------------------------------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--anno_dir", required=True, help="folder with *_fixed.json files")
    ap.add_argument("--img_root", required=True, help="root folder of all JPEGs")
    ap.add_argument("--out_root", required=True, help="output folder for YOLO splits")
    ap.add_argument("--copy", action="store_true", help="copy images instead of symlink")
    args = ap.parse_args()

    anno_dir = Path(args.anno_dir)
    out_root = Path(args.out_root)

    # global class-id map from train split
    id_map = build_id_map(json.loads((anno_dir / "train_fixed.json").read_text()))

    # split-name mapping: COCO file -  YOLO folder
    SPLITS = {
        "train_fixed.json":       "train",
        "train_balanced.json": "train_balanced",
        "val_all_fixed.json":     "val",
        "cis_val_fixed.json":     "cis_val",
        "trans_val_fixed.json":   "trans_val",
        "cis_test_noverify.json":    "cis_test",
        "trans_test_noverify.json":  "trans_test"
    }

    for coco_file, yolo_split in SPLITS.items():
        convert_split(
            anno_dir / coco_file,
            Path(args.img_root),
            out_root / yolo_split,
            id_map,
            copy=args.copy
        )

    # write names.txt (one class per line)
    names = [None] * len(id_map)
    cats = json.loads((anno_dir / "train_fixed.json").read_text())["categories"]
    for c in cats:
        names[id_map[c["id"]]] = c["name"]
    (out_root / "names.txt").write_text("\n".join(names))
    print("\nConversion finished. YOLO formatted dataset lives in", out_root)
