import json
from pathlib import Path
import shutil

# Paths
VERIFY_JSON = Path("../../data/preprocessed/annotations/cleaned/verify_test.json")
SOURCE_DIR = Path("../../data/images")
DEST_DIR = Path("../../data/verification")
DEST_DIR.mkdir(parents=True, exist_ok=True)

# Load verification image filenames from JSON
with open(VERIFY_JSON, "r") as f:
    data = json.load(f)

filenames = [img["file_name"] for img in data["images"]]

print(f"Total images to copy: {len(filenames)}")

# Copy files
copied, missing = 0, 0
for fn in filenames:
    src = SOURCE_DIR / fn
    dst = DEST_DIR / fn

    if src.exists():
        shutil.copy2(src, dst)
        copied += 1
    else:
        print(f" Missing: {fn}")
        missing += 1

print(f"\n Done: {copied} files copied.")
if missing > 0:
    print(f" {missing} files were missing.")
