from pathlib import Path
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import yaml
import torch
import os
from collections import Counter
import sys
import numpy as np
from ultralytics.data.dataset import YOLODataset
from ultralytics.data import dataset   as _ds
from ultralytics.data import build     as _dsbuild 
import torchvision.utils as vutils
import cv2
import random
import matplotlib.pyplot as plt
import math
from torch.utils.data import DataLoader, WeightedRandomSampler
from collections import Counter


cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

# Setup paths
ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT))
DATA_YAML = ROOT / "configs" / "model" / "yolo_balanced.yaml"

from scripts.augmentation.augment_cct import get_detection_augmentation

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

os.environ['WANDB_MODE'] = 'offline'


with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
raw_names = data_cfg.get("names", [])
if isinstance(raw_names, dict):
    # sort by key and extract values into a list
    class_names = [name for _, name in sorted(raw_names.items(), key=lambda x: int(x[0]))]
else:
    class_names = list(raw_names)


# ────────────────── dataset ────────────────────────────────────────────
class ProgressiveAugmentationDataset(YOLODataset):
    """
    Wraps Ultralytics' YOLODataset so that **Albumentations** is applied
    _after_ the default image/label loading, with an intensity that grows
    linearly (or any schedule you like) over training epochs.
    """

    def __init__(self,
                 *args,
                 intensity_start: float = 0.10,
                 intensity_max:   float = 0.60,
                 **kwargs):
        """
        Args
        ----
        intensity_start : augmentation strength for the very first epoch
        intensity_max   : strength once `epoch == max_epochs – 1`
        All other *args/**kwargs are forwarded to YOLODataset.
        """
        super().__init__(*args, **kwargs)

        self.intensity_start = intensity_start
        self.intensity_max   = intensity_max
        self.epoch           = 0
        self.max_epochs      = 1 

        # Build the first Albumentations pipeline
        self._build_aug_pipeline()

    # ────────────────────────────────────────────────────────────────
    # Hooks called from the trainer callbacks
    def set_epoch_info(self, epoch: int, max_epochs: int):
        self.epoch      = epoch
        self.max_epochs = max_epochs
        self._build_aug_pipeline()                     # rebuild with new intensity

    def get_augmentation_intensity(self) -> float:
        """Current scalar intensity ∈ [intensity_start, intensity_max]."""

        progress = self.epoch / max(1, self.max_epochs - 1)
        frac = 0.5 * (1 - math.cos(math.pi * progress))
        return self.intensity_start + (self.intensity_max - self.intensity_start) * frac

    # ────────────────────────────────────────────────────────────────
    # Standard Dataset API
    # ────────────────────────────────────────────────────────────────
    def __getitem__(self, index):
        """
        Returns
        -------
        img    : torch.FloatTensor shape [3,H,W]  (0-255, RGB)
        labels : torch.FloatTensor shape [N,6]     (class,x,y,w,h,??) — same
                 layout Ultralytics expects, so the trainer is unchanged.
        """
        out= super().__getitem__(index)   # Ultralytics call
        img      = out["img"].numpy().transpose(1, 2, 0)   # CHW - HWC np.uint8
        bboxes   = out["bboxes"]
        classes  = out["cls"]

        # Check if we should apply augmentation (only for training)
        is_train = getattr(self, "augment", False)
        if not is_train:
            # For validation, return the original data without augmentation
            return out

        # Check the shape and type of classes
       

        if isinstance(classes, torch.Tensor):
            if classes.dim() == 0:  # scalar tensor
                classes_list = [int(classes.item())]
            elif classes.dim() == 1:  # 1D tensor
                classes_list = classes.int().tolist()
            else:  # multi-dimensional tensor, flatten
                classes_list = classes.flatten().int().tolist()
        else:
            # If it's already a list or numpy array
            classes_list = np.array(classes).flatten().astype(int).tolist()
        
        # Convert bboxes to proper format
        if isinstance(bboxes, torch.Tensor):
            bboxes_list = bboxes.tolist()
        else:
            bboxes_list = bboxes
        
        # YOLODataset already gives RGB uint8 as np.ndarray and a (n, 5) label
        # tensor  (class, x, y, w, h).  Convert to Albumentations format,
        # run the transform, then back to torch.
        albu_dict = self.albu_pipeline(
            image=img,
            bboxes=bboxes_list,                 # drop class column
            cls   =classes_list
        )

        img_aug      = albu_dict["image"]
        bboxes_aug   = albu_dict["bboxes"]
        classes_aug  = albu_dict["cls"]

        if len(bboxes_aug) == 0:                  # all boxes got dropped
            # fall back to the un-augmented sample to avoid trouble
            return out

        
        n = len(bboxes_aug)
           
        # write the aug data back into the dict
        out["img"]     = torch.from_numpy(img_aug.transpose(2, 0, 1)).float()
        out["bboxes"]  = torch.tensor(bboxes_aug,  dtype=torch.float32)
        out["cls"]     = torch.tensor(classes_aug, dtype=torch.float32)


        

        # Fixed batch_idx handling
        if "batch_idx" in out and len(out["batch_idx"]) > 0:
            idx_value = out["batch_idx"][0].item()
        else:
            idx_value = index
        out["batch_idx"] = torch.full((n,), idx_value, dtype=torch.float32)

        return out

    # ────────────────────────────────────────────────────────────────
    # Internals

    def _build_aug_pipeline(self):
        is_train = getattr(self, "augment", False)     
        print(f"Debug - Building aug pipeline: is_train={is_train}")

        mode = "train" if is_train else "val"          
        intensity = self.get_augmentation_intensity() if is_train else 0.0


        self.albu_pipeline = get_detection_augmentation(
            mode      = mode,
            img_size  = self.imgsz,
            intensity = intensity,
            epoch     = self.epoch,
            max_epochs= self.max_epochs,
        )

    
_ds.YOLODataset       = ProgressiveAugmentationDataset
_dsbuild.YOLODataset  = ProgressiveAugmentationDataset 

# Load class names from yaml
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg.get("names", [])


def log_class_distribution(trainer, class_names):
    labels = getattr(trainer, "batch", None)
    if labels is None or len(labels) == 0:
        return

    # YOLOv8 format: (batch_idx, class, x, y, w, h)
    all_labels = []
    for label in labels:
        if isinstance(label, torch.Tensor):
            classes = label[:, 1].detach().cpu().numpy().astype(int)
            all_labels.extend(classes.tolist())
    
    class_counts = Counter(all_labels)
    log_data = {}
    for idx, name in enumerate(class_names):
        log_data[f"epoch_class_dist/{name}"] = class_counts.get(idx, 0)

    wandb.log(log_data, step=trainer.epoch)


FREEZE_EPOCHS = 20

def set_backbone_requires_grad(model, requires_grad):
    """
    Set requires_grad for backbone layers in YOLOv8
    """
    # Access the actual model
    yolo_model = model.model if hasattr(model, 'model') else model
    
    if requires_grad:
        for i in range(10):  # Layers 0-9 are backbone
            for param in yolo_model[i].parameters():
                param.requires_grad = True
        print("🔓 Unfrozen ALL model parameters")
    else:
        # When freezing, only freeze backbone layers 0-9
        for i in range(10):  # Layers 0-9 are backbone
            for param in yolo_model[i].parameters():
                param.requires_grad = False
        print("🔒 Frozen backbone layers 0-9")



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
        all_labels = []
        for ann_dict in ds.labels:
            # ann_dict['cls'] is a numpy array of shape (n_objects, 1)
            cls_array = ann_dict['cls']
            if cls_array.size > 0:  # Check if there are any annotations
                for cls_val in cls_array.flatten():
                    all_labels.append(int(cls_val))
        counts = Counter(all_labels)
        
        total_samples = len(all_labels)
        n_classes = len(counts)
        sample_weights = [n_classes / (counts[c] * total_samples) for c in all_labels]
        
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
            num_workers=original_loader.num_workers,
            pin_memory=original_loader.pin_memory,
            collate_fn=getattr(original_loader, 'collate_fn', None),
            drop_last=getattr(original_loader, 'drop_last', False),
        )

    
    # Unfreeze when hitting threshold
    if epoch == FREEZE_EPOCHS:
        print(f"\n Unfreezing full model at epoch {epoch}")
        set_backbone_requires_grad(model, True)

            # Quick verification
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

    if hasattr(trainer.train_loader, 'dataset'):
            print("Dataset found\n")
            dataset = trainer.train_loader.dataset
            if isinstance(dataset, ProgressiveAugmentationDataset):
                max_epochs = trainer.args.epochs if hasattr(trainer.args, 'epochs') else 150
                dataset.set_epoch_info(trainer.epoch, max_epochs)

                intensity = dataset.get_augmentation_intensity()
                print(f"Epoch {trainer.epoch}: Augmentation intensity = {intensity:.3f}")
            else:
                print("No Augmentation\n")



# Custom logging callback: end of training epoch
def on_train_epoch_end(trainer):
    if not wandb.run:
        return
    

    losses = getattr(trainer, "loss_items", None)
    if isinstance(losses, torch.Tensor):
        losses = losses.detach().cpu().numpy().tolist()

    if isinstance(losses, (list, tuple)) and len(losses) >= 3:
        box, cls, dfl = map(float, losses[:3])
        total = box + cls + dfl
        if total > 0:
            wandb.log({
                "custom/box_loss_ratio": box / total,
                "custom/cls_loss_ratio": cls / total,
                "custom/dfl_loss_ratio": dfl / total,
                "custom/train_loss_total": total,
                "custom/box_loss": box,
                "custom/cls_loss": cls,
                "custom/dfl_loss": dfl,
            }, step=trainer.epoch)

     # Log class distribution
    log_class_distribution(trainer, class_names)

    if trainer.epoch % 5 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()

# Custom logging callback: end of validation epoch
def on_val_end(validator):

    if not wandb.run:
        return

    try:
        epoch = getattr(validator, 'epoch', wandb.run.step if wandb.run else 0)
        metrics = getattr(validator, "metrics", None)
        if not (metrics and hasattr(metrics, 'box')):
            return

        box = metrics.box
        log_dict = {}

        if hasattr(box, 'map50'):
            log_dict['custom/val_map50'] = float(box.map50)
        if hasattr(box, 'map'):
            log_dict['custom/val_map_all'] = float(box.map)
        if 'custom/val_map50' in log_dict and log_dict['custom/val_map50'] > 0:
            log_dict['custom/val_map_ratio'] = log_dict['custom/val_map_all'] / log_dict['custom/val_map50']

        if hasattr(box, "p") and hasattr(box, "r") and hasattr(box, "ap"):
            for i, name in enumerate(class_names):
                if i < len(box.p) and i < len(box.r) and i < len(box.ap):
                    name = class_names[i]
                    log_dict[f"per_class/{name}_precision"] = float(box.p[i])
                    log_dict[f"per_class/{name}_recall"] = float(box.r[i])
                    log_dict[f"per_class/{name}_AP50-95"] = float(box.ap[i])

        if log_dict:
            wandb.log(log_dict, step=epoch)

    except Exception as e:
        print(f"Error in validation callback: {e}")




# Training logic
def run():


    if wandb.run is not None:
        wandb.finish()

    wandb.init(
        project="yolov8-training",
        name="y8_augmented_freezed_layers",
        mode="offline",
        config={
            "model": "yolov8s",
            "epochs": 100,
            "batch_size": 20,
            "img_size": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "cos_lr": True,
            "weight_decay": 0.008,
            "dropout": 0.1,
            "label_smoothing": 0.05,
            "patience": 15,
            "augmentation": {
                "type": "custom_camera_trap_aug",
                "progressive": True,
                "intensity_start": 0.1,
                "intensity_max": 0.6,
                "custom_augments": [
                    "illumination", "geometric", "motion_blur",
                    "weather_noise", "occlusion", "sensor_variation",
                    "perspective", "seasonal", "advanced_aug"
                ]
            }
        }
    )

    model = YOLO("yolov8s.pt")

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    add_wandb_callback(model, enable_model_checkpointing=True)

    
    print(f"Torch sees CUDA: {torch.cuda.is_available()} | Device count: {torch.cuda.device_count()}")

    # Before training, check the current epoch
    if hasattr(model, 'trainer') and model.trainer:
        print("Current epoch:", getattr(model.trainer, 'epoch', 'Unknown'))


    model.train( 
    data=str(DATA_YAML), 
    imgsz=640,
    device=0, 
    half=True, 
    epochs=100, 
    batch=16,
    workers=4, 
    project="yolov8",
    name="yolov8_augmented_freezed_layers", 
    optimizer="AdamW", 
    lr0=0.001,
    lrf = 0.1, 
    cos_lr=True, 
    amp=True,
    cache=False,
    dropout=0.1,
    save_period=1,
    val=True,
    plots=True, 
    label_smoothing=0.05, 
    weight_decay=0.005, 
    patience=15, 
    warmup_epochs=3, 
    auto_augment=None, 
    hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, 
    translate=0, scale=0, shear=0, perspective=0, 
    flipud=0, fliplr=0, mosaic=0, mixup=0.1, copy_paste=0.1,erasing=0 
    )

    wandb.finish()

if __name__ == "__main__":
    run()
