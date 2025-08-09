import json
from pathlib import Path

def remove_empty_class(coco_path, output_path):
    with open(coco_path, 'r') as f:
        coco = json.load(f)

    # Step 1: Find the ID of the 'empty' class
    empty_category_id = None
    new_categories = []
    for cat in coco['categories']:
        if cat['name'].lower() == 'empty':
            empty_category_id = cat['id']
        else:
            new_categories.append(cat)

    if empty_category_id is None:
        print(f"[{coco_path}] No 'empty' category found. Skipping.")
        return

    # Step 2: Filter out annotations with that ID
    new_annotations = [ann for ann in coco['annotations'] if ann['category_id'] != empty_category_id]

    # Step 3: Construct cleaned COCO JSON
    cleaned_coco = {
        'images': coco['images'],  # keep all images
        'annotations': new_annotations,
        'categories': new_categories
    }

    # Save result
    with open(output_path, 'w') as f:
        json.dump(cleaned_coco, f)
    print(f" Cleaned saved to {output_path}")

# === Apply to all your files ===
input_dir = Path("data/preprocessed/annotations")
output_dir = Path("data/preprocessed/annotations/cleaned")
output_dir.mkdir(parents=True, exist_ok=True)

for coco_file in input_dir.glob("*_fixed.json"):
    out_file = output_dir / coco_file.name
    remove_empty_class(coco_file, out_file)
