#!/usr/bin/env python3
import argparse
from pathlib import Path

def link_and_remap_split(src_split: Path, dst_split: Path, vehicle_ids: set):
    """
    Given a split folder (e.g. 'train_balanced') under src_root,
    create symlinks for images and remap labels into dst_root/<split>.
    """
    # Detect images/ & labels/ subfolders
    if (src_split / "images").is_dir() and (src_split / "labels").is_dir():
        img_src = src_split / "images"
        lbl_src = src_split / "labels"
    else:
        img_src = src_split
        lbl_src = src_split

    img_dst = dst_split / "images"
    lbl_dst = dst_split / "labels"
    img_dst.mkdir(parents=True, exist_ok=True)
    lbl_dst.mkdir(parents=True, exist_ok=True)

    for img_path in img_src.glob("*"):
        if img_path.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue

        # create a symlink instead of copy
        dst_img = img_dst / img_path.name
        if not dst_img.exists():
            dst_img.symlink_to(img_path.resolve())

        # remap the corresponding label file
        stem = img_path.stem
        lbl_in = lbl_src / f"{stem}.txt"
        if not lbl_in.exists():
            continue

        lines = lbl_in.read_text().splitlines()
        remapped = []
        for line in lines:
            orig_cls, *coords = line.split()
            new_cls = 1 if int(orig_cls) in vehicle_ids else 0
            remapped.append(" ".join([str(new_cls), *coords]))

        (lbl_dst / f"{stem}.txt").write_text("\n".join(remapped))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src",  required=True, help="Path to original yolo_images folder")
    p.add_argument("--dst",  required=True, help="Output path for megadetector_images")
    p.add_argument("--vehicle-ids", nargs="+", type=int, default=[11],
                   help="Original class IDs to treat as vehicle")
    args = p.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    vehicle_ids = set(args.vehicle_ids)

    splits = [d for d in src_root.iterdir() if d.is_dir()]
    for split in splits:

        dst_split = dst_root / split.name

        if dst_split.exists():
            print(f"  Skipping '{split.name}' — already exists at {dst_split}")
            continue

        print(f"[{split.name}] → {dst_split}")
        link_and_remap_split(split, dst_split, vehicle_ids)

    # write new names.txt
    names_txt = dst_root / "names.txt"
    names_txt.write_text("0: animal\n1: vehicle\n")
    print(f"Wrote new names.txt at {names_txt}")
