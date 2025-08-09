import argparse, json, os
from pathlib import Path
from PIL import Image

PAD = 0.08           # 8 % padding
SKIP_CAT = 33        # “car”

def pad_bbox(x, y, w, h, W, H):
    px, py = w*PAD, h*PAD
    return (max(int(x-px), 0),
            max(int(y-py), 0),
            min(int(x+w+px), W),
            min(int(y+h+py), H))

def main(json_file, img_root, out_split):
    out_dir = Path("../../data/crops_gt") / out_split
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(Path(json_file).read_text())
    cats = {c["id"]: c["name"] for c in meta["categories"]}
    imgs = {im["id"]: im for im in meta["images"]}

    for ann in meta["annotations"]:
        if ann["category_id"] == SKIP_CAT:
            continue
        iminfo = imgs[ann["image_id"]]
        im_path = Path(img_root) / iminfo["file_name"]
        img = Image.open(im_path).convert("RGB")
        W, H = img.size
        x1, y1, x2, y2 = pad_bbox(*ann["bbox"], W=W, H=H)
        crop = img.crop((x1, y1, x2, y2))

        species = cats[ann["category_id"]]
        species_dir = out_dir / species
        species_dir.mkdir(parents=True, exist_ok=True)
        crop_name = f"{iminfo['id']}_{ann['id']}.jpg"
        crop.save(species_dir / f"{iminfo['id']}_{ann['id']}.jpg", quality=95)

        manifest_path = out_dir / f"{out_split}_manifest.json"
        header = not manifest_path.exists()
        with open(manifest_path, "a", newline="") as f:
            if header:
                f.write("crop_path,species,source_image,x1,y1,x2,y2,det_score\n")
            f.write(f"{species_dir / crop_name},{species},{iminfo['file_name']},"
                    f"{x1},{y1},{x2},{y2},1.0\n")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--img_root", required=True)
    ap.add_argument("--split", required=True)
    args = ap.parse_args()
    main(args.json, args.img_root, args.split)
