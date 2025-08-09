"""
merge_val_splits.py
Create val_all_fixed.json (cis+trans) — only annotation IDs are remapped.
"""

import json, pathlib

SRC_DIR = pathlib.Path("data/preprocessed/annotations")
OUT     = SRC_DIR / "val_all_fixed.json"

cis_path   = SRC_DIR / "cis_val_fixed.json"
trans_path = SRC_DIR / "trans_val_fixed.json"

cis   = json.loads(cis_path.read_text())
trans = json.loads(trans_path.read_text())

# categories must match
assert cis["categories"] == trans["categories"], "category lists differ!"

#  build a set of annotation-IDs already used by cis
used_ann_ids = {ann["id"] for ann in cis["annotations"]}

#  remap any colliding annotation IDs from trans
for ann in trans["annotations"]:
    if ann["id"] in used_ann_ids:
        ann["id"] = f"tva_{ann['id']}"

#  concatenate lists
merged = cis
merged["images"]      += trans["images"]        
merged["annotations"] += trans["annotations"]

#  write out
OUT.write_text(json.dumps(merged))
print("wrote", OUT)
