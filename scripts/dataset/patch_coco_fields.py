#!/usr/bin/env python3
"""
patch_coco_fields.py
====================

Add missing COCO-spec keys (`area`, `iscrowd`) to every cleaned-split JSON.
Run it once after generating / copying your splits.

Usage
-----
python patch_coco_fields.py  [optional_directory]

• If no directory is given, the script defaults to:
  ../../data/preprocessed/annotations/cleaned
• Files that already contain the fields are left untouched.

"""

import json, sys, pathlib

# ─── target directory (can be overridden by CLI arg) ───
JSON_DIR = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else (
    pathlib.Path(__file__).resolve().parent /
    "../../data/preprocessed/annotations/cleaned"
).resolve()

def patch_file(path: pathlib.Path) -> None:
    data = json.loads(path.read_text())
    changed = False

    for ann in data.get("annotations", []):
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = w * h
            changed = True
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0
            changed = True

    if changed:
        path.write_text(json.dumps(data, indent=2))
        print(f"patched  {path.name}")
    else:
        print(f"ok       {path.name}")

def main() -> None:
    if not JSON_DIR.is_dir():
        sys.exit(f"Directory not found: {JSON_DIR}")

    for jp in JSON_DIR.glob("*.json"):
        patch_file(jp)

if __name__ == "__main__":
    main()
