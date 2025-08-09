import json
import random
from collections import defaultdict
from pathlib import Path


output_dir = Path("../../data/preprocessed/annotations/cleaned")

# Paths
cis_path = Path(output_dir /"cis_test_fixed.json")
trans_path = Path(output_dir /"trans_test_fixed.json")


def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(data, path):
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)

def get_image_map(images):
    return {img["id"]: img for img in images}

def get_class_id_to_name(categories):
    return {cat["id"]: cat["name"] for cat in categories}

def get_name_to_class_id(categories):
    return {cat["name"]: cat["id"] for cat in categories}

def subset_images(images, annotations, categories, selection_ratio=0.05):
    random.seed(42)  # Reproducibility

    image_to_anns = defaultdict(list)
    for ann in annotations:
        image_to_anns[ann["image_id"]].append(ann)

    selected_image_ids = set()
    class_to_images = defaultdict(set)

    for ann in annotations:
        cid = ann["category_id"]
        class_name = [c["name"] for c in categories if c["id"] == cid][0]
        class_to_images[class_name].add(ann["image_id"])

    for class_name, all_image_ids in class_to_images.items():
        total = len(all_image_ids)
        n_select = max(1, int(total * selection_ratio))
        selected = random.sample(list(all_image_ids), min(n_select, total))
        selected_image_ids.update(selected)

    selected_images = [img for img in images if img["id"] in selected_image_ids]
    selected_annotations = [ann for ann in annotations if ann["image_id"] in selected_image_ids]

    return {
        "images": selected_images,
        "annotations": selected_annotations,
        "categories": categories
    }

# Load original data
cis_data = load_json(cis_path)
trans_data = load_json(trans_path)

# Create verification subsets
verify_cis = subset_images(
    images=cis_data["images"],
    annotations=cis_data["annotations"],
    categories=cis_data["categories"],
)

verify_trans = subset_images(
    images=trans_data["images"],
    annotations=trans_data["annotations"],
    categories=trans_data["categories"],
)

# Merge into single verify_test
merged_verify = {
    "images": verify_cis["images"] + verify_trans["images"],
    "annotations": verify_cis["annotations"] + verify_trans["annotations"],
    "categories": cis_data["categories"]
}

# Save all
save_json(verify_cis, output_dir / "verify_cis_test.json")
save_json(verify_trans, output_dir / "verify_trans_test.json")
save_json(merged_verify, output_dir / "verify_test.json")
