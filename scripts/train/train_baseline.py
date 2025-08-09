from pathlib import Path
from ultralytics import YOLO
import wandb
from wandb.integration.ultralytics import add_wandb_callback
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np
import torch
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.empty_cache()
torch.backends.cudnn.benchmark = False  # More stable memory usage
torch.backends.cudnn.deterministic = True



#  Setup paths 
ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = ROOT / "configs" / "model" / "yolo.yaml"

#  Load class names from yaml file 
with open(DATA_YAML, 'r') as f:
    data_cfg = yaml.safe_load(f)
class_names = data_cfg.get("names", [])

# Custom callback: end of training epoch 
def on_train_epoch_end(trainer):
    if not wandb.run:
        return

    losses = getattr(trainer, "loss_items", None)
    
    # Handle PyTorch tensor format
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
    
    if trainer.epoch % 10 == 0:
        torch.cuda.empty_cache()
        import gc
        gc.collect()

# Custom callback: end of validation epoch
def on_val_end(validator):
    if not wandb.run:
        return

    try:
        # Get epoch 
        epoch = getattr(validator, 'epoch', None)
        if epoch is None:
        
            args = getattr(validator, 'args', None)
            if args and hasattr(args, 'epoch'):
                epoch = args.epoch
            else:
                # Use current wandb step as fallback
                epoch = wandb.run.step if wandb.run else 0

        metrics = getattr(validator, "metrics", None)
        if not (metrics and hasattr(metrics, 'box')):
            return

        box = metrics.box
        log_dict = {}

        # Log aggregate metrics
        if hasattr(box, 'map50'):
            log_dict['custom/val_map50'] = float(box.map50)
        if hasattr(box, 'map'):
            log_dict['custom/val_map_all'] = float(box.map)
        if 'custom/val_map50' in log_dict and log_dict['custom/val_map50'] > 0:
            log_dict['custom/val_map_ratio'] = log_dict['custom/val_map_all'] / log_dict['custom/val_map50']

        # Log per-class metrics
        if hasattr(box, "p") and hasattr(box, "r") and hasattr(box, "ap"):
            # Handle both dict and list formats for class names
            if isinstance(class_names, dict):
                for i in range(min(len(box.p), len(box.r), len(box.ap))):
                    if i in class_names:
                        name = class_names[i]
                        log_dict[f"per_class/{name}_precision"] = float(box.p[i])
                        log_dict[f"per_class/{name}_recall"] = float(box.r[i]) 
                        log_dict[f"per_class/{name}_AP50-95"] = float(box.ap[i])
            else:
                for i, name in enumerate(class_names):
                    if i < len(box.p) and i < len(box.r) and i < len(box.ap):
                        log_dict[f"per_class/{name}_precision"] = float(box.p[i])
                        log_dict[f"per_class/{name}_recall"] = float(box.r[i])
                        log_dict[f"per_class/{name}_AP50-95"] = float(box.ap[i])

        if log_dict:
            wandb.log(log_dict, step=epoch)


    except Exception as e:
        print(f" Error in validation callback: {e}")

# Training logic
def run():
    wandb.init(
        project="yolov8-training",
        name="y8_baseline_no_aug",
        config={
            "model": "yolov8s",
            "epochs": 100,
            "batch_size": 16,
            "img_size": 640,
            "optimizer": "AdamW",
            "lr0": 0.001,
            "weight_decay": 0.01,
            "augmentation": "none"
        }
    )

    assert wandb.run is not None, "W&B run was not initialized"

    model = YOLO("yolov8s.pt")

    # Register custom callbacks
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_val_end", on_val_end)
    
    # Register W&B integration
    add_wandb_callback(model, enable_model_checkpointing=True)

    model.train(
        data=str(DATA_YAML),
        imgsz=640,
        epochs=100,
        batch=16,
        workers=4,
        project="yolov8",
        name="y8_baseline_no_aug",
        optimizer="AdamW",
        lr0=0.001,
        dropout=0.1,
        label_smoothing=0.1,
        weight_decay=0.01,
        hsv_h=0, hsv_s=0, hsv_v=0,
        degrees=0, translate=0, scale=0, shear=0, perspective=0,
        flipud=0, fliplr=0,
        mosaic=0, mixup=0, copy_paste=0,exist_ok= True,
    )

    wandb.finish()

if __name__ == "__main__":
    run()