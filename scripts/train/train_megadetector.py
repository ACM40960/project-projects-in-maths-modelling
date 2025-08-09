import os

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

from pathlib import Path
from ultralytics import YOLO
from torch.utils.tensorboard import SummaryWriter 
import yaml
import torch

from collections import Counter
import sys
from torch.utils.data import DataLoader, WeightedRandomSampler

import gc
# Before training
torch.cuda.empty_cache()
gc.collect()

os.environ["WANDB_DISABLED"] = "true"
writer = SummaryWriter(log_dir="tensorlogs/megadetectorv6_augmented")

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
DATA_YAML = ROOT / "configs" / "model" / "megadetector.yaml"


torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

STEP=0


with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
raw_names = data_cfg.get("names", [])
if isinstance(raw_names, dict):

    class_names = [name for _, name in sorted(raw_names.items(), key=lambda x: int(x[0]))]
else:
    class_names = list(raw_names)

# Load class names from yaml
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg.get("names", [])


def log_class_distribution(trainer, class_names):
    labels = getattr(trainer, "batch", None)
    if labels is None or len(labels) == 0:
        return

    # YOLOv9 format
    all_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            classes = label[:, 1].detach().cpu().numpy().astype(int)
            all_labels.extend(classes.tolist())
    
    class_counts = Counter(all_labels)
    log_data = {}
    for idx, name in enumerate(class_names):
        log_data[f"epoch_class_dist/{name}"] = class_counts.get(idx, 0)

    writer.add_scalar(f"epoch_class_dist/{name}", class_counts.get(idx, 0), trainer.epoch)


FREEZE_EPOCHS = 15

def set_backbone_requires_grad(model, requires_grad):
    """
    Set requires_grad for backbone layers in YOLOv8
    """
    # Access the actual model
    yolo_model = model.model if hasattr(model, 'model') else model
    
    if requires_grad:
        for i in range(8,10): 
            for param in yolo_model[i].parameters():
                param.requires_grad = True
        print("Unfrozen last 2 model parameters of backbone")
    else:
        # When freezing, only freeze backbone layers 0-9
        for i in range(10):  # Layers 0-9 are backbone
            for param in yolo_model[i].parameters():
                param.requires_grad = False
        print("Frozen backbone layers 0-9")

# Progressive augmentation update
def on_train_epoch_start(trainer):

    epoch = trainer.epoch
    model = trainer.model

    if epoch==1:
        batch = next(iter(trainer.train_loader)) 
        labels = [int(l) for l in batch['cls'].flatten() if l >= 0] 
        counts = Counter(labels) 
        print(f"Sampler check - Classes in batch: {dict(counts)}")   
    
    # Freeze backbone for first N epochs
    if epoch == 0:
        print(f"\n Freezing backbone for first {FREEZE_EPOCHS} epochs")
        set_backbone_requires_grad(model, False)

            # Quick verification
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        total = sum(1 for p in model.parameters())
        print(f" Status: {frozen}/{total} parameters frozen")

            # Your existing code but with safeguards
        print("Weight sampler fired")

        ds = trainer.train_loader.dataset
        all_labels_per_image = [ann['cls'].flatten().astype(int) for ann in ds.labels]

        # Flatten all labels to calculate class frequencies
        all_labels_flat = [label for sublist in all_labels_per_image for label in sublist]
        if not all_labels_flat:
            print(" No labels found in dataset to create sampler. Skipping.")
            return

        class_counts = Counter(all_labels_flat)

        # Calculate a weight for each class
        class_weights = {i: 1.0 / count for i, count in class_counts.items()}

        # The weight of an image is the max weight of the classes it contains
        sample_weights = []
        for image_labels in all_labels_per_image:
            if len(image_labels) > 0:
                # Get the weight for each class in the image and find the max
                max_weight = max(class_weights.get(l, 0) for l in image_labels)
                sample_weights.append(max_weight)
            else:
                # Give images with no objects a neutral weight or a small weight
                sample_weights.append(min(class_weights.values()) if class_weights else 1.0)
        
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )
        
        # Preserve original DataLoader settings
        original_loader = trainer.train_loader
        trainer.train_loader = DataLoader(
            ds,
            batch_size=original_loader.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=original_loader.num_workers,
            persistent_workers=True,
            collate_fn=getattr(original_loader, 'collate_fn', None),
            drop_last=getattr(original_loader, 'drop_last', False),
        )

    
    # Unfreeze when hitting threshold
    if epoch == FREEZE_EPOCHS:
        print(f"\n Unfreezing full model at epoch {epoch}")
        set_backbone_requires_grad(model, True)

            #  verification
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        total = sum(1 for p in model.parameters())
        print(f" Status: {frozen}/{total} parameters frozen")
        
        # update optimizer with initial_lr parameter
        new_lr = trainer.args.lr0 * 0.1  # 10x lower LR for fine-tuning
        
        # Update existing optimizer
        for param_group in trainer.optimizer.param_groups:
            param_group['lr'] = new_lr
            param_group['initial_lr'] = new_lr
        
        print(f"Reduced learning rate to {new_lr:.6f} for fine-tuning")



# Custom logging callback: end of training epoch
def on_train_epoch_end(trainer):
    

    losses = getattr(trainer, "loss_items", None)
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy().tolist()

    if isinstance(losses, (list, tuple)) and len(losses) >= 3:
        box, cls, dfl = map(float, losses[:3])
        total = box + cls + dfl
        if total > 0:
            writer.add_scalar("loss/total",        total, trainer.epoch)
            writer.add_scalar("loss/box",          box,   trainer.epoch)
            writer.add_scalar("loss/cls",          cls,   trainer.epoch)
            writer.add_scalar("loss/dfl",          dfl,   trainer.epoch)
            writer.add_scalars("loss/ratio", {
            "box": box / total if total else 0,
            "cls": cls / total if total else 0,
            "dfl": dfl / total if total else 0,
        }, trainer.epoch)

     # Log class distribution
    log_class_distribution(trainer, class_names)

    if trainer.epoch % 2 == 0:
        torch.cuda.empty_cache()
        gc.collect()

# Custom logging callback: end of validation epoch
def on_val_end(validator):
    global STEP

    try:
        epoch = STEP
        STEP+=1
        metrics = getattr(validator, "metrics", None)
        if not (metrics and hasattr(metrics, 'box')):
            return

        box = metrics.box
        

        if hasattr(box, "map50"): writer.add_scalar("val/mAP50",     float(box.map50), epoch)
        if hasattr(box, "map"):   writer.add_scalar("val/mAP50-95",  float(box.map),   epoch)
        if hasattr(box, "p") and hasattr(box, "r") and hasattr(box, "ap"):
            for i, name in enumerate(class_names):
                writer.add_scalar(f"val/{name}_precision", float(box.p[i]),  epoch)
                writer.add_scalar(f"val/{name}_recall",    float(box.r[i]),  epoch)
                writer.add_scalar(f"val/{name}_AP50-95",   float(box.ap[i]), epoch)

        

    except Exception as e:
        print(f"Error in validation callback: {e}")


# Training 
def run():
    

    model = YOLO("MDV6-yolov9-c.pt")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)

    

    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        device=0,
        deterministic=False,
        half=True,
        cache=False,
        epochs=80,
        batch=16,
        workers=4,
        project="megadetector_v6",
        name="megadetector_augmented",
        optimizer="AdamW",
        lr0=0.001,
        lrf=0.1,
        cos_lr=True,
        amp=True,
        dropout=0.1,
        label_smoothing=0.05,
        weight_decay=5e-4,
        patience=10,
        warmup_epochs=3,
        save_period=1,
        auto_augment=None,
        augment=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
        degrees=10, translate=0.1, scale=0.1, shear=0, perspective=0.001,
        flipud=0, fliplr=0.5,
        mosaic=0.1, mixup=0.1, copy_paste=0.1,erasing=0.1,plots=True,save_json=True, val=True, close_mosaic=10
    )

writer.close()

if __name__ == "__main__":
    run()
