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

# --- 1. Model -------------------------------------------------------------
class ConvNeXtClassifier(nn.Module):
    """
    ConvNeXt-Small with enhanced head for camera trap classification
    """
    def __init__(self, n_cls, dropout=0.5):  # Increased dropout for ConvNeXt
        super().__init__()
        self.backbone = timm.create_model('convnext_small', pretrained=True, features_only=True)
        
        #  classifier head
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
    
# -------------------------------------------------------------------------
class TailAwareFolder(torch.utils.data.Dataset):
    """
    Like ImageFolder, but applies `tail_tf` to rare classes
    and `main_tf` to everyone else.
    """
    def __init__(self, root, main_tf, tail_tf, tail_ids):
        self.ds        = ImageFolder(root)            
        self.main_tf   = main_tf
        self.tail_tf   = tail_tf
        self.tail_ids  = set(tail_ids)

    def __len__(self):
        return len(self.ds)

    @property
    def classes(self):
        return self.ds.classes

    @property
    def targets(self):
        return self.ds.targets

    def __getitem__(self, idx):
        img, y = self.ds[idx]
        tf = self.tail_tf if y in self.tail_ids else self.main_tf
        return tf(img), y
    
class TailMixCollate:
    def __init__(self, num_classes, tail_ids, alpha=0.2, p=0.5):
        self.num_classes = num_classes
        self.tail_ids    = set(tail_ids)
        self.alpha, self.p = alpha, p

    def __call__(self, batch):
        imgs, labels = list(zip(*batch))
        imgs   = torch.stack(imgs)
        labels = torch.tensor(labels)

        if any(l.item() in self.tail_ids for l in labels) and np.random.rand() < self.p:
            lam = np.random.beta(self.alpha, self.alpha)
            idx = torch.randperm(imgs.size(0))
            imgs = lam * imgs + (1 - lam) * imgs[idx]

            y1 = torch.nn.functional.one_hot(labels, self.num_classes).float()
            y2 = torch.nn.functional.one_hot(labels[idx], self.num_classes).float()
            labels = lam * y1 + (1 - lam) * y2     # soft labels

        return imgs, labels


# Transforms
train_tf = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.8, 1.2)),  
    tv.transforms.RandomHorizontalFlip(),  
    tv.transforms.RandomRotation(10),  
    tv.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.01),
    tv.transforms.RandomGrayscale(p=0.05),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    tv.transforms.RandomErasing(p=0.25, scale=(0.02, 0.08)),  
])

tail_tf = tv.transforms.Compose([
    tv.transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.RandomRotation(25),
    tv.transforms.ColorJitter(0.5,0.5,0.4,0.1),
    tv.transforms.RandomPerspective(distortion_scale=0.5, p=0.5),
    tv.transforms.RandomGrayscale(p=0.2),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485,0.456,0.406],
                            std =[0.229,0.224,0.225]),
    tv.transforms.RandomErasing(p=0.3, scale=(0.02, 0.1)),
])

val_tf = tv.transforms.Compose([
    tv.transforms.Resize(256),
    tv.transforms.CenterCrop(224),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# ----------------------------------------------------------------------
# Class-Balanced Focal Loss  (Cui et al., 2019)
class CBFocalLoss(nn.Module):
    def __init__(self, samples_per_cls, beta=0.9999, gamma=2.0):
        """
        samples_per_cls : list or 1-D tensor with the raw image counts per class
        beta            : 0→1, larger = stronger class re-weighting
        gamma           : focal-loss focusing parameter (γ=0 → plain CB-Softmax)
        """
        super().__init__()
        self.gamma = gamma
        self.smoothing = 0.1

        #  number of samples 
        spc = torch.as_tensor(samples_per_cls, dtype=torch.float32)
        effective_num = 1.0 - torch.pow(beta, spc)
        weights = (1.0 - beta) / effective_num
        self.register_buffer("class_weights",
                             weights / weights.sum() * len(spc))   # normalise

    def forward(self, logits, target):
        """
        logits : (B, C)  raw model outputs
        target : (B) integers  *or*  (B, C) soft labels (e.g. MixUp)
        """
        if target.ndim == 1:  # hard labels
            target_onehot = torch.nn.functional.one_hot(target, num_classes=logits.size(1)).float()
            if self.smoothing > 0:
                target_onehot = target_onehot * (1.0 - self.smoothing) + self.smoothing / logits.size(1)
        else:  # soft labels 
            target_onehot = target.float()

        # add per-class logit bias 
        logits = logits + self.class_weights.log().unsqueeze(0)
        log_prob = torch.nn.functional.log_softmax(logits, dim=1)
        prob     = log_prob.exp()

        focal = torch.pow(1.0 - prob, self.gamma)
        loss  = -(focal * target_onehot * log_prob).sum(dim=1)      # per sample
        return loss.mean()


if __name__ == '__main__':

    base_ds    = ImageFolder("../../data/crops_gt/train")   # plain, no tf
    cls_counts = torch.bincount(torch.tensor(base_ds.targets))

    tail_ids   = torch.where(cls_counts < 200)[0].tolist()  
    print("Tail classes:", [base_ds.classes[i] for i in tail_ids])

    # --- 3. Datasets ----------------------------------------------------------
    train_ds = TailAwareFolder("../../data/crops_gt/train",
                           main_tf=train_tf,
                           tail_tf=tail_tf,
                           tail_ids=tail_ids)
    val_ds = ImageFolder("../../data/crops_gt/val", transform=val_tf)

    # class-balanced sampler
    cls_counts = torch.bincount(torch.tensor(train_ds.targets))
    print(f"Class distribution: {dict(zip(train_ds.classes, cls_counts.tolist()))}")

    weights = 1. / cls_counts[train_ds.targets]
    sampler = WeightedRandomSampler(weights, len(weights), replacement=True)

    mix_collate = TailMixCollate(num_classes=len(train_ds.classes),
                             tail_ids=tail_ids, alpha=0.2, p=0.6)

    train_dl = DataLoader(train_ds, batch_size=64, sampler=sampler,
                      num_workers=4, pin_memory=True, collate_fn=mix_collate)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    # Enhanced Training Loop
    device = "cuda"
    
    
    model = ConvNeXtClassifier(n_cls=len(train_ds.classes)).to(device)


    # Loss function with class weights for extra imbalance handling
    class_weights = 1. / cls_counts.float()
    class_weights = class_weights / class_weights.sum() * len(class_weights)  # normalize
    crit = CBFocalLoss(samples_per_cls=cls_counts, beta=0.999, gamma=1.1).to(device)

    print("CB weights =", crit.class_weights.cpu().numpy())

    # Optimizer with layer-wise learning rates
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        if 'classifier' in name:
            head_params.append(param)
        else:
            backbone_params.append(param)

    opt = torch.optim.AdamW([
    {'params': backbone_params, 'lr': 5e-5},   
    {'params': head_params, 'lr': 2e-4}       
    ], weight_decay=0.05)  

    # Different scheduler
    from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
    sched = CosineAnnealingWarmRestarts(opt, T_0=10, T_mult=2, eta_min=1e-6)
    scaler = GradScaler()

    # Training tracking
    best_acc = 0.0
    patience = 10
    patience_counter = 0
    train_losses,val_losses = [],[]
    val_accs, val_accs_top3 = [],[]

    # Create output directory
    output_dir = Path("convnext_classaware")
    output_dir.mkdir(exist_ok=True)

    print(f"Training on {len(train_ds)} samples, validating on {len(val_ds)} samples")
    print(f"Number of classes: {len(train_ds.classes)}")


    for epoch in range(60):


        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_dl, desc=f"Epoch {epoch+1:02d}/60")
        for x, y in pbar:
            x,y = x.to(device), y.to(device)
            opt.zero_grad()

            with autocast(device_type="cuda"):
                pred = model(x); 
                is_soft = (y.ndim == 2)               # True if batch was mixed up
                if is_soft:
                    loss   = crit(pred, y)             # CB-Focal or BCEWithLogitsLoss
                    target = y.argmax(1)              # argmax gives class index per sample
                else:
                    loss   = crit(pred, y)            
                    target = y

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scale_before = scaler.get_scale()
            scaler.step(opt); scaler.update()
            scale_after  = scaler.get_scale()

            if scale_after == scale_before:    
                sched.step()
            
            train_loss += loss.item()
            train_correct += (pred.argmax(1) == target).sum().item()
            train_total += target.size(0)
            
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
                
                correct1 += correct[:, 0].sum().item()      # top-1
                correct3 += correct.any(dim=1).sum().item() # top-3
                val_total += y.size(0)
                
                
                all_preds.extend(pred_topk[:, 0].cpu().numpy())
                all_targets.extend(y.cpu().numpy()) 
        
        train_acc = train_correct / train_total * 100
        val_acc = correct1 / val_total * 100
        val_top3    = correct3 / val_total * 100
        avg_train_loss = train_loss / len(train_dl)
        val_loss   /= val_total
        
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
                'class_names': train_ds.classes
            }, output_dir / "best_model.pth")
            
            # Save detailed classification report
            report = classification_report(all_targets, all_preds, 
                                        target_names=train_ds.classes,
                                        output_dict=True)
            with open(output_dir / "classification_report.json", 'w') as f:
                json.dump(report, f, indent=2)
            
            # Save confusion matrix
            cm = confusion_matrix(all_targets, all_preds)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=train_ds.classes,
                    yticklabels=train_ds.classes)
            plt.title(f'Confusion Matrix - Epoch {epoch+1} (Val Acc: {val_acc:.2f}%)')
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
    plt.plot(val_losses,   label="val")
    plt.title('Training/Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot( val_accs, label="val top-1")
    plt.plot( val_accs_top3, label="val top-3")
    plt.xlabel("Epoch"); plt.ylabel("Accuracy (%)"); plt.title("Validation accuracy")
    plt.grid(True); plt.legend()

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300)
    plt.show()

    print(f"\nTraining complete! Best validation accuracy: {best_acc:.2f}%")
    print(f"Model and results saved to: {output_dir}")

    # Load best model for final evaluation
    checkpoint = torch.load(output_dir / "best_model.pth")
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Best model from epoch {checkpoint['epoch']+1} loaded.")
