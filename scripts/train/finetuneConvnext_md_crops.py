import timm, torch, torchvision as tv, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from tqdm import tqdm
from torch.utils.data import WeightedRandomSampler
import random
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast

torch.manual_seed(123)
np.random.seed(123)
random.seed(123)
torch.backends.cudnn.deterministic = True 

# Model
class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Small with enhanced head for camera trap classification
    """
    def __init__(self, n_cls, dropout=0.5):
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, features_only=True)
        
        feature_dim = self.backbone.feature_info[-1]['num_chs']
        
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_cls)
        )
    
    def forward(self, x):
        x = self.backbone(x)[-1]
        x = self.pool(x)
        return self.classifier(x)


#  Enhanced Loss with Background Class Handling
class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        super().__init__()
        self.gamma = gamma
        self.smoothing = 0.05

        spc = torch.as_tensor(samples_per_cls, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, spc)
        weights = (1.0 - beta) / effective_num
        self.register_buffer("class_weights",
                             weights / weights.sum() * len(spc))

    def forward(self, logits, target):
        if target.ndim == 1:
            target_onehot = torch.nn.functional.one_hot(target, num_classes=logits.size(1)).float()
            if self.smoothing > 0:
                target_onehot = target_onehot * (1.0 - self.smoothing) + self.smoothing / logits.size(1)
        else:
            target_onehot = target.float()

        logits = logits + self.class_weights.log().unsqueeze(0)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        prob = log_prob.exp()

        focal = torch.pow(1.0 - prob, self.gamma)
        loss = -(focal * target_onehot * log_prob).sum(dim=1)
        return loss.mean()

# Training Setup
if __name__ == '__main__':
    # Load new datasets with background class
    train_ds = ImageFolder("../../data/megadetector_crops/train", transform=None) 
    val_ds = ImageFolder("../../data/megadetector_crops/val", transform=None)
    
    print(f"Number of classes: {len(train_ds.classes)}")
    
    # Get class index of "background"
    bg_idx = train_ds.class_to_idx.get("background", None)

    if bg_idx is not None:
        print(f"Excluding 'background' class (index {bg_idx})")

        # Filter out background samples from train/val datasets
        train_ds.samples = [s for s in train_ds.samples if s[1] != bg_idx]
        train_ds.targets = [s[1] for s in train_ds.samples]

        val_ds.samples = [s for s in val_ds.samples if s[1] != bg_idx]
        val_ds.targets = [s[1] for s in val_ds.samples]

        # Re-map class_to_idx and classes
        new_classes = [c for c in train_ds.classes if c != "background"]
        train_ds.classes = new_classes
        train_ds.class_to_idx = {cls: i for i, cls in enumerate(new_classes)}

        # Build label_map: old index → new index
        # Must map old label indices to new ones directly
        old_class_to_idx = {v: k for k, v in ImageFolder("../../data/megadetector_crops/train").class_to_idx.items()}
        label_map = {}
        for old_idx, class_name in old_class_to_idx.items():
            if class_name != "background":
                new_idx = train_ds.class_to_idx[class_name]
                label_map[old_idx] = new_idx

        train_ds.targets = [label_map[t] for _, t in train_ds.samples]
        train_ds.samples = [(p, label_map[t]) for p, t in train_ds.samples]

        val_ds.targets = [label_map[t] for _, t in val_ds.samples]
        val_ds.samples = [(p, label_map[t]) for p, t in val_ds.samples]

    else:
        print("No 'background' class found — nothing to exclude.")

    print("Classes after BG removal:", train_ds.classes)

    # Load the pretrained model and extend it
    old_model_path = "convnext_classaware/best_model.pth"
    
    # Load old class names from checkpoint
    old_checkpoint = torch.load(old_model_path, map_location="cpu")
    old_classes = old_checkpoint['class_names']
    
    # Extend the model
    device = "cuda"
    model = ConvNeXtClassifier(n_cls=len(train_ds.classes)).to(device)
    
    ckpt = torch.load(old_model_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])


    # --- Apply transforms after model setup ---
    train_tf = tv.transforms.Compose([
        tv.transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),
        tv.transforms.RandomHorizontalFlip(),
        tv.transforms.RandomRotation(10),
        tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.01),
        tv.transforms.RandomGrayscale(p=0.05),
        tv.transforms.ToTensor(),
        tv.transforms.RandomApply([
            tv.transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
        ], p=0.1),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        tv.transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),
    ])

    val_tf = tv.transforms.Compose([
        tv.transforms.Resize(256),
        tv.transforms.CenterCrop(224),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transforms
    train_ds.transform = train_tf
    val_ds.transform = val_tf
    
    # --- Setup training with class balancing ---
    cls_counts = torch.bincount(torch.tensor(train_ds.targets))
    print(f"Class distribution: {dict(zip(train_ds.classes, cls_counts.tolist()))}")
    print("Class names with indices:")
    for idx, class_name in enumerate(train_ds.classes):
        print(f"Index {idx}: {class_name}")

    
    # Create weighted sampler
    weights = 1. / cls_counts[train_ds.targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
    
    # Data loaders
    train_dl = DataLoader(train_ds, batch_size=32, sampler=sampler,  # Reduced batch size for finetuning
                         num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    # --- Setup loss and optimizer for finetuning ---
    crit = CBFocalLoss(samples_per_cls=cls_counts, beta=0.999, gamma=1.1).to(device)
    
    # Lower learning rates for finetuning
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    opt = torch.optim.AdamW([
        {'params': backbone_params, 'lr': 1e-5},    # Very low LR for pretrained backbone
        {'params': head_params, 'lr': 1e-3}         # Higher LR for new classifier parts
    ], weight_decay=0.05)

    # Scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    sched = CosineAnnealingWarmRestarts(opt, T_0=8, T_mult=2, eta_min=1e-7)
    scaler = GradScaler()

    # Training tracking
    best_acc = 0.0
    patience = 10  # Reduced patience for finetuning
    patience_counter = 0
    train_losses, val_losses = [], []
    val_accs, val_accs_top3 = [], []

    # Create output directory
    output_dir = Path("convnext_classaware_finetuned")
    output_dir.mkdir(exist_ok=True)

    print(f"Finetuning on {len(train_ds)} samples, validating on {len(val_ds)} samples")
    
    # Shorter training for finetuning
    epochs = 25
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1:02d}/{epochs}")
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()

            with autocast(device_type="cuda"):
                pred = model(x)
                loss = crit(pred, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scale_before = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            scale_after = scaler.get_scale()

            if scale_after == scale_before:
                sched.step()
            
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == y).sum().item()
            train_total += y.size(0)
            
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'acc': f"{train_correct/train_total*100:.2f}%"
            })
        
        # Validation
        model.eval()
        val_loss, correct1, correct3, val_total = 0.0, 0, 0, 0
        all_preds, all_targets = [], []

        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                with autocast(device_type="cuda"):
                    out = model(x)
                    loss = crit(out, y)
                val_loss += loss.item() * x.size(0)
                _, pred_topk = out.topk(3, 1, True, True)
                correct = pred_topk.eq(y.view(-1, 1))
                
                correct1 += correct[:, 0].sum().item()
                correct3 += correct.any(dim=1).sum().item()
                val_total += y.size(0)
                
                all_preds.extend(pred_topk[:, 0].cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        train_acc = train_correct / train_total * 100
        val_acc = correct1 / val_total * 100
        val_top3 = correct3 / val_total * 100
        avg_train_loss = train_loss / len(train_dl)
        val_loss /= val_total
        
        train_losses.append(avg_train_loss)
        val_accs.append(val_acc)
        val_losses.append(val_loss)
        val_accs_top3.append(val_top3)
        
        print(f"Epoch {epoch+1:02d} | "
              f"Train Loss {avg_train_loss:.3f} | Train Acc {train_acc:.2f}% | "
              f"Val Loss {val_loss:.3f} | Val Top-1 Acc {val_acc:.2f}% | Val Top-3 Acc {val_top3:.2f}% | "
              f"LR {sched.get_last_lr()[0]:.2e}")
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'best_acc': best_acc,
                'class_names': train_ds.classes,  # Save new class names
                'old_classes': old_classes        # Save old classes for reference
            }, output_dir / "best_model.pth")
            
            # Save detailed classification report
            report = classification_report(all_targets, all_preds, 
                                         target_names=train_ds.classes,
                                         output_dict=True)
            with open(output_dir / "classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save confusion matrix with normalized annotations
            cm = confusion_matrix(all_targets, all_preds)
            cm_norm = cm.astype(float) / cm.sum(1, keepdims=True)
            
            plt.figure(figsize=(12, 10))
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=train_ds.classes,
                       yticklabels=train_ds.classes)
            plt.title(f'Normalized Confusion Matrix - Epoch {epoch+1} (Val Acc: {val_acc:.2f}%)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(output_dir / "confusion_matrix.png", dpi=300)
            plt.close()
            
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Final training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.title('Training/Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(val_accs, label="val top-1")
    plt.plot(val_accs_top3, label="val top-3")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Validation accuracy")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300)
    plt.show()

    print(f"\nFinetuning complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model and results saved to: {output_dir}")

    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']+1} loaded.")