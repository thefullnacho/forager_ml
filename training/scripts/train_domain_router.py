"""
Train the domain router classifier (tf_efficientnet_lite2).

4-class model (berry / mushroom / plant / other) that runs first in the
two-stage inference pipeline to route each frame to the right expert(s).

The "other" class is the critical first-pass OOD gate — it catches everything
that isn't a foraging target before the expensive expert models are invoked.

Uses the same tf_efficientnet_lite2 architecture and training pipeline as
the expert models (weighted loss, RandAugment, MixUp) for consistency and
Hailo 8L compatibility.

The "other" class is weighted 2x relative to target domains to encourage
high recall on non-foraging inputs (it's better to abstain than to misroute).

Dataset: router_dataset/ (built by training/scripts/build_router_dataset.py)
"Other" images: inat_dataset/other/ (downloaded by data/acquisition/other_pull_inat.py)

Usage:
    CUDA_VISIBLE_DEVICES=1 python training/scripts/train_domain_router.py

    # Then benchmark:
    CUDA_VISIBLE_DEVICES=1 python training/scripts/benchmark_router.py \\
        --checkpoint runs/efficientnet/domain_router/best.pt \\
        --dataset router_dataset
"""

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ── Architecture (Hailo 8L compatible) ────────────────────────────────────────
ARCH       = "tf_efficientnet_lite2"
IMG_SIZE   = 224
BATCH_SIZE = 64
EPOCHS     = 60
LR_INIT    = 1e-3
LR_MIN     = 1e-5
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.3

# Weight multiplier for the "other" class — false negatives (misrouting OOD
# inputs into a domain) are worse than false positives (unnecessary abstention).
OTHER_CLASS_WEIGHT = 2.0


def compute_class_weights(train_dataset, other_weight: float) -> torch.FloatTensor:
    counts = np.bincount(
        [label for _, label in train_dataset.samples],
        minlength=len(train_dataset.classes)
    ).astype(float)
    counts = np.maximum(counts, 1)
    weights = 1.0 / counts
    weights /= weights.mean()

    # Boost "other" class weight
    for i, cls in enumerate(train_dataset.classes):
        if cls == "other":
            weights[i] *= other_weight
            print(f"  Boosted 'other' class weight × {other_weight:.1f} → {weights[i]:.3f}")

    return torch.FloatTensor(weights)


def mixup_data(images, labels, alpha=0.3):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    mixed = lam * images + (1 - lam) * images[idx]
    return mixed, labels, labels[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


def build_transforms(img_size: int, augment: bool):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandAugment(num_ops=2, magnitude=9),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 1.14)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])


def train_one_epoch(model, loader, optimizer, criterion, device, mixup_alpha):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        if mixup_alpha > 0 and np.random.rand() < 0.5:
            images, y_a, y_b, lam = mixup_data(images, labels, mixup_alpha)
            logits = model(images)
            loss   = mixup_criterion(criterion, logits, y_a, y_b, lam)
        else:
            logits = model(images)
            loss   = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        loss   = criterion(logits, labels)

        total_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


def parse_args():
    p = argparse.ArgumentParser(description="Train domain router (EfficientNet Lite2)")
    p.add_argument("--dataset",       default="router_dataset",
                   help="Dataset root with train/ and val/ subdirs")
    p.add_argument("--name",          default="domain_router",
                   help="Run name (output: runs/efficientnet/<name>/)")
    p.add_argument("--epochs",        type=int,   default=EPOCHS)
    p.add_argument("--batch-size",    type=int,   default=BATCH_SIZE)
    p.add_argument("--lr",            type=float, default=LR_INIT)
    p.add_argument("--mixup-alpha",   type=float, default=MIXUP_ALPHA)
    p.add_argument("--other-weight",  type=float, default=OTHER_CLASS_WEIGHT,
                   help="Weight multiplier for 'other' class in loss function")
    p.add_argument("--workers",       type=int,   default=4)
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dataset_dir = args.dataset if os.path.isabs(args.dataset) \
                  else os.path.join(repo_root, args.dataset)
    output_dir  = os.path.join(repo_root, "runs", "efficientnet", args.name)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")

    # ── Datasets ─────────────────────────────────────────────────────────────
    train_transform = build_transforms(IMG_SIZE, augment=True)
    val_transform   = build_transforms(IMG_SIZE, augment=False)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir   = os.path.join(dataset_dir, "val")

    if not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
        print(f"ERROR: Expected train/ and val/ in {dataset_dir}")
        print(f"  Run: python training/scripts/build_router_dataset.py")
        sys.exit(1)

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset   = datasets.ImageFolder(val_dir,   transform=val_transform)

    classes = train_dataset.classes
    if "other" not in classes:
        print("\nWARNING: 'other' class not found in dataset!")
        print("  The router will have no OOD awareness.")
        print("  Run: python data/acquisition/other_pull_inat.py")
        print("  Then: python training/scripts/build_router_dataset.py\n")

    print(f"\nDataset: {dataset_dir}")
    print(f"Classes ({len(classes)}): {classes}")
    train_counts = np.bincount([l for _, l in train_dataset.samples], minlength=len(classes))
    val_counts   = np.bincount([l for _, l in val_dataset.samples],   minlength=len(classes))
    for i, cls in enumerate(classes):
        print(f"  {cls:<15}  train: {train_counts[i]:>5}  val: {val_counts[i]:>5}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"\nBuilding {ARCH} with {len(classes)} output classes ...")
    model = timm.create_model(ARCH, pretrained=True, num_classes=len(classes))
    model = model.to(device)

    # ── Loss with class weighting ─────────────────────────────────────────────
    class_weights = compute_class_weights(train_dataset, args.other_weight)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=LABEL_SMOOTHING)
    print(f"Class weights: {dict(zip(classes, class_weights.cpu().numpy().round(3)))}")

    # ── Optimiser + cosine LR ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=LR_MIN
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val_acc = 0.0
    best_ckpt    = os.path.join(output_dir, "best.pt")

    print(f"\n{'Epoch':>6}  {'TrainLoss':>10}  {'TrainAcc':>9}  "
          f"{'ValLoss':>8}  {'ValAcc':>8}  {'LR':>10}")
    print("-" * 65)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device, args.mixup_alpha
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(f"{epoch:>6}  {train_loss:>10.4f}  {train_acc:>9.4f}  "
              f"{val_loss:>8.4f}  {val_acc:>8.4f}  {lr_now:>10.2e}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch":            epoch,
                "model_name":       args.name,
                "arch":             ARCH,
                "img_size":         IMG_SIZE,
                "num_classes":      len(classes),
                "classes":          classes,
                "model_state_dict": model.state_dict(),
                "val_accuracy":     val_acc,
            }, best_ckpt)
            print(f"  ★ New best: {val_acc:.4f}  → saved to {best_ckpt}")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Checkpoint: {best_ckpt}")
    print(f"\nNext: CUDA_VISIBLE_DEVICES=1 python training/scripts/benchmark_router.py "
          f"--checkpoint {best_ckpt} --dataset {dataset_dir}")


if __name__ == "__main__":
    main()
