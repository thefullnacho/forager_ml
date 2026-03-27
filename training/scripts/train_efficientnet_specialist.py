"""
Train an EfficientNet-B2 specialist classifier using PyTorch / timm.

Replaces the YOLO-based specialist training scripts with a proper
single-shot image classification architecture suited for FGVC on Hailo 8L.

The model outputs raw logits (no softmax/sigmoid) so we can later:
  1. Export ONNX without activation for DFC compilation
  2. Compute energy scores for OOD detection
  3. Apply temperature scaling for calibrated confidence

Dataset layout expected (ImageFolder):
    dataset_split/
        train/
            class_a/  img001.jpg ...
            class_b/  img002.jpg ...
        val/
            class_a/  img101.jpg ...
            class_b/  img102.jpg ...

Usage:
    python training/scripts/train_efficientnet_specialist.py \
        --dataset psychedelics_dataset_split \
        --name psychedelics_expert \
        --epochs 50

    python training/scripts/train_efficientnet_specialist.py \
        --dataset berry_dataset_split \
        --name berry_expert \
        --epochs 50
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile
import timm

ImageFile.LOAD_TRUNCATED_IMAGES = True   # skip corrupt/truncated images rather than crashing

# ── GPU diagnostics ──────────────────────────────────────────────────────────

def print_gpu_diagnostics():
    print("=" * 60)
    print("GPU / CUDA DIAGNOSTICS")
    print("=" * 60)
    print(f"PyTorch version   : {torch.__version__}")
    print(f"CUDA available    : {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version      : {torch.version.cuda}")
        print(f"cuDNN version     : {torch.backends.cudnn.version()}")
        print(f"GPU count         : {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            free, total = torch.cuda.mem_get_info(i)
            print(f"  GPU {i}: {props.name}")
            print(f"    VRAM total : {total / 1e9:.1f} GB")
            print(f"    VRAM free  : {free / 1e9:.1f} GB")
            print(f"    Compute cap: {props.major}.{props.minor}")
    else:
        print("\n⚠  No CUDA GPU detected — training will run on CPU.")

    print("=" * 60)


def gpu_smoke_test(device):
    if device.type != "cuda":
        return
    print("\nRunning GPU smoke test...")
    try:
        x = torch.randn(64, 512, device=device)
        _ = x @ x.T
        del x
        torch.cuda.empty_cache()
        print("Smoke test passed.\n")
    except Exception as e:
        print(f"\n✗ GPU smoke test FAILED: {e}")
        sys.exit(1)


def get_amp_dtype(device: torch.device) -> torch.dtype:
    """
    Return the best AMP dtype for the active GPU.

    Blackwell (sm_120+) and Hopper (sm_90) prefer bfloat16 — float16 causes
    CUDA launch failures on these architectures.
    Ampere (sm_80) and Ada (sm_89) work fine with float16.
    CPU: no AMP.
    """
    if device.type != "cuda":
        return torch.float32
    major = torch.cuda.get_device_properties(device).major
    if major >= 9:   # Hopper (sm_90) and Blackwell (sm_120)
        return torch.bfloat16
    return torch.float16


# ── Data augmentation ────────────────────────────────────────────────────────

def build_transforms(img_size: int = 224):
    """
    Training: aggressive augmentation for FGVC (fine-grained visual classification).
    Validation: deterministic center crop.
    """
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0), ratio=(0.75, 1.33)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.05),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),   # 256 for 224
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return train_transform, val_transform


# ── Model ────────────────────────────────────────────────────────────────────

def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    EfficientNet-Lite2 (TF weights) from timm with a fresh classification head.
    Lite variant replaces SE blocks with Identity, which is required for hailo8l
    compilation — real SE avgpool layers exceed the chip's shift-delta limit of
    2.0 during int8 quantization. tf_efficientnet_lite2 has pretrained weights
    available; bare efficientnet_lite2 does not.
    Output is raw logits — no softmax/sigmoid.
    """
    model = timm.create_model(
        "tf_efficientnet_lite2",
        pretrained=pretrained,
        num_classes=num_classes,
    )
    return model


# ── Training loop ────────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, scaler, device, amp_dtype, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        if scaler is not None:
            # float16 path — GradScaler prevents underflow
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # bfloat16 / cpu path — no scaling needed
            loss.backward()
            optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    batch {batch_idx+1}/{len(loader)}  "
                  f"loss={loss.item():.4f}  acc={correct/total:.4f}")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model, loader, criterion, device, amp_dtype):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, labels)

        running_loss += loss.item() * images.size(0)
        _, predicted = logits.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    val_loss = running_loss / total
    val_acc = correct / total
    return val_loss, val_acc


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Train EfficientNet-B2 specialist")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset with train/ and val/ subdirs (ImageFolder)")
    p.add_argument("--name", required=True,
                   help="Model name, e.g. 'psychedelics_expert'")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Initial learning rate (cosine schedule)")
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--output-dir", default=None,
                   help="Where to save checkpoints (default: runs/efficientnet/<name>)")
    p.add_argument("--resume", default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--no-pretrained", action="store_true",
                   help="Train from scratch (not recommended)")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Resolve dataset path
    dataset_dir = args.dataset
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(repo_root, dataset_dir)

    train_dir = os.path.join(dataset_dir, "train")
    val_dir = os.path.join(dataset_dir, "val")

    if not os.path.isdir(train_dir):
        print(f"ERROR: train directory not found: {train_dir}")
        sys.exit(1)
    if not os.path.isdir(val_dir):
        print(f"ERROR: val directory not found: {val_dir}")
        sys.exit(1)

    # Output directory
    output_dir = args.output_dir or os.path.join(repo_root, "runs", "efficientnet", args.name)
    os.makedirs(output_dir, exist_ok=True)

    # Device
    print_gpu_diagnostics()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_smoke_test(device)

    # Data
    train_transform, val_transform = build_transforms(args.img_size)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    classes = train_dataset.classes
    num_classes = len(classes)

    print(f"\nDataset   : {dataset_dir}")
    print(f"Classes   : {classes}")
    print(f"Num classes: {num_classes}")
    print(f"Train size: {len(train_dataset)}")
    print(f"Val size  : {len(val_dataset)}")
    print(f"Device    : {device}")
    print(f"Batch     : {args.batch_size}")
    print(f"Epochs    : {args.epochs}")
    print(f"LR        : {args.lr}")
    print(f"Output    : {output_dir}\n")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=False,
        persistent_workers=False,
    )

    # Model
    model = build_model(num_classes, pretrained=not args.no_pretrained)

    start_epoch = 0
    if args.resume:
        print(f"Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1

    model = model.to(device)

    # Loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=0.01,
        foreach=False,   # disable fused multi-tensor kernels — unstable on sm_120 (Blackwell)
        fused=False,     # disable fused CUDA kernel — same issue
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # Skip scheduler steps if resuming
    for _ in range(start_epoch):
        scheduler.step()

    amp_dtype = get_amp_dtype(device)
    use_amp = device.type == "cuda"

    # bfloat16 doesn't suffer from underflow so GradScaler is unnecessary
    # and in fact unsupported with bf16. Only use it for float16.
    if use_amp and amp_dtype == torch.float16:
        scaler = torch.amp.GradScaler("cuda")
        print(f"AMP dtype  : float16 (GradScaler enabled)")
    else:
        scaler = None
        amp_label = str(amp_dtype).replace("torch.", "") if use_amp else "disabled"
        print(f"AMP dtype  : {amp_label} (no GradScaler)")

    # Training
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1}/{args.epochs}  (lr={lr_now:.6f})")

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, amp_dtype, epoch
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device, amp_dtype)

        scheduler.step()

        elapsed = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  train_acc={train_acc:.4f}")
        print(f"  val_loss={val_loss:.4f}    val_acc={val_acc:.4f}")
        print(f"  time={elapsed:.1f}s")

        # Save checkpoint
        ckpt = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_acc": val_acc,
            "val_loss": val_loss,
            "classes": classes,
            "num_classes": num_classes,
            "model_name": args.name,
            "arch": "tf_efficientnet_lite2",
            "img_size": args.img_size,
        }

        torch.save(ckpt, os.path.join(output_dir, "last.pt"))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(ckpt, os.path.join(output_dir, "best.pt"))
            print(f"  ★ New best: {val_acc:.4f}")

        print()

    # Save class manifest (matches existing format)
    manifest = {"model": args.name, "classes": classes}
    manifest_path = os.path.join(output_dir, f"{args.name}_classes.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("=" * 60)
    print("Training complete.")
    print(f"Best val accuracy: {best_val_acc:.4f} (epoch {best_epoch+1})")
    print(f"Best weights     : {output_dir}/best.pt")
    print(f"Class manifest   : {manifest_path}")
    print("=" * 60)
    print(f"\nNext step:")
    print(f"  python training/scripts/export_efficientnet_onnx.py \\")
    print(f"      --checkpoint {output_dir}/best.pt \\")
    print(f"      --no-activation")


if __name__ == "__main__":
    main()
