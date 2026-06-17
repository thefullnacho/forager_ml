"""
Calibrate energy-based OOD rejection thresholds for EfficientNet-B2 specialists.

Runs the trained model (PyTorch checkpoint) on the validation set, computes
energy scores from raw logits, and determines the 95th-percentile rejection
threshold. Inputs exceeding this threshold at inference time are flagged as
out-of-distribution and trigger abstention.

Energy score:
    E(x) = -log(sum(exp(logits)))
    Low energy  → in-distribution (model confident in a known class)
    High energy → out-of-distribution (model confused / unfamiliar input)

Outputs:
    runs/efficientnet/<name>/energy_calibration.json
        {
            "model": "psychedelics_expert",
            "threshold_p95": -2.34,
            "threshold_p99": -1.87,
            "temperature": 1.0,
            "num_samples": 2400,
            "energy_stats": { "mean": -5.12, "std": 0.83, "min": -8.91, "max": -0.42 }
        }

Usage:
    python training/scripts/calibrate_energy_threshold.py \
        --checkpoint runs/efficientnet/psychedelics_expert/best.pt \
        --dataset psychedelics_dataset_split

    # With temperature scaling:
    python training/scripts/calibrate_energy_threshold.py \
        --checkpoint runs/efficientnet/psychedelics_expert/best.pt \
        --dataset psychedelics_dataset_split \
        --temperature 2.0
"""

import argparse
import json
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


def parse_args():
    p = argparse.ArgumentParser(description="Calibrate energy OOD thresholds")
    p.add_argument("--checkpoint", required=True, help="Path to best.pt")
    p.add_argument("--dataset", required=True,
                   help="Dataset with val/ subdir (ImageFolder format)")
    p.add_argument("--temperature", type=float, default=1.0,
                   help="Temperature for energy score (higher = softer distribution)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--percentiles", nargs="+", type=float,
                   default=[90, 95, 99],
                   help="Percentiles to compute for threshold selection")
    return p.parse_args()


def compute_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """
    Energy score: E(x) = -T * log(sum(exp(logits / T)))

    Lower energy = higher confidence in a known class.
    Higher energy = less familiar input (OOD candidate).

    Uses the numerically stable log-sum-exp (subtract the row max, then add it
    back) so large logits can't overflow np.exp. This must stay identical to
    runner._energy_score so calibrated thresholds match inference exactly.
    """
    scaled = logits / temperature
    m = np.max(scaled, axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.sum(np.exp(scaled - m), axis=1))   # stable log-sum-exp
    return -temperature * lse


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run all validation images through the model.
    Returns (logits, predictions, labels) as numpy arrays.
    """
    model.eval()
    all_logits = []
    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images).cpu().numpy()
        preds = np.argmax(logits, axis=1)

        all_logits.append(logits)
        all_preds.append(preds)
        all_labels.append(labels.numpy())

    return (
        np.concatenate(all_logits),
        np.concatenate(all_preds),
        np.concatenate(all_labels),
    )


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    # Resolve paths
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_root, ckpt_path)

    dataset_dir = args.dataset
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(repo_root, dataset_dir)

    val_dir = os.path.join(dataset_dir, "val")
    if not os.path.isdir(val_dir):
        print(f"ERROR: val/ directory not found in {dataset_dir}")
        sys.exit(1)

    # Load model
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_name = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    img_size = ckpt.get("img_size", 224)
    classes = ckpt["classes"]

    print(f"Model      : {model_name}")
    print(f"Classes    : {num_classes}")
    print(f"Temperature: {args.temperature}")

    arch = ckpt.get("arch", "tf_efficientnet_lite2")
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Device     : {device}")

    # Validation data
    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    print(f"Val images : {len(val_dataset)}")

    # Collect logits
    print("\nRunning inference on validation set ...")
    logits, preds, labels = collect_logits(model, val_loader, device)

    # Accuracy check
    accuracy = np.mean(preds == labels)
    print(f"Val accuracy: {accuracy:.4f}")

    # Energy scores
    energies = compute_energy(logits, args.temperature)

    # Statistics
    energy_stats = {
        "mean": float(np.mean(energies)),
        "std": float(np.std(energies)),
        "min": float(np.min(energies)),
        "max": float(np.max(energies)),
        "median": float(np.median(energies)),
    }

    print(f"\nEnergy score statistics (in-distribution):")
    for k, v in energy_stats.items():
        print(f"  {k:8s}: {v:.4f}")

    # Percentile thresholds
    print(f"\nRejection thresholds:")
    thresholds = {}
    for pct in args.percentiles:
        val = float(np.percentile(energies, pct))
        thresholds[f"p{int(pct)}"] = val
        print(f"  {int(pct)}th percentile: {val:.4f}")

    # Per-class energy breakdown
    print(f"\nPer-class energy scores:")
    for idx, cls_name in enumerate(classes):
        mask = labels == idx
        if mask.sum() > 0:
            cls_energies = energies[mask]
            cls_acc = np.mean(preds[mask] == labels[mask])
            print(f"  {cls_name:40s}  n={mask.sum():5d}  "
                  f"energy={np.mean(cls_energies):.4f} ± {np.std(cls_energies):.4f}  "
                  f"acc={cls_acc:.4f}")

    # Save calibration
    output_dir = os.path.dirname(ckpt_path)
    calibration = {
        "model": model_name,
        "threshold_p95": thresholds.get("p95", thresholds.get(f"p{int(args.percentiles[1])}")),
        "threshold_p99": thresholds.get("p99", thresholds.get(f"p{int(args.percentiles[-1])}")),
        "thresholds": thresholds,
        "temperature": args.temperature,
        "num_samples": len(val_dataset),
        "val_accuracy": float(accuracy),
        "energy_stats": energy_stats,
        "classes": classes,
    }

    calib_path = os.path.join(output_dir, "energy_calibration.json")
    with open(calib_path, "w") as f:
        json.dump(calibration, f, indent=2)

    print(f"\n✓ Calibration saved: {calib_path}")

    # Also copy to inference/models for deployment
    inference_calib = os.path.join(repo_root, "inference", "models", f"{model_name}_energy.json")
    with open(inference_calib, "w") as f:
        json.dump(calibration, f, indent=2)
    print(f"✓ Inference copy : {inference_calib}")

    print(f"\n" + "=" * 60)
    print(f"Calibration complete for {model_name}")
    print(f"  Recommended rejection threshold (95th pct): {calibration['threshold_p95']:.4f}")
    print(f"  Images with energy above this value should trigger ABSTENTION.")
    print(f"=" * 60)


if __name__ == "__main__":
    main()
