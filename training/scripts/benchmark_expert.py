"""
benchmark_expert.py — Per-class accuracy, recall, precision, F1 + confusion matrix.

Runs the trained checkpoint on the val split and produces:
  - Console report with per-class metrics, flagging deadly/toxic classes
  - runs/efficientnet/<name>/benchmark.json  — machine-readable results
  - runs/efficientnet/<name>/confusion_matrix.png

Safety classes (DEADLY/TOXIC) are highlighted in the report since
false negatives on these are the most dangerous failure mode.

Usage:
    CUDA_VISIBLE_DEVICES=1 python training/scripts/benchmark_expert.py \\
        --checkpoint runs/efficientnet/psychedelics_expert/best.pt \\
        --dataset psychedelics_dataset_split

    # All three at once:
    for expert in psychedelics berry highvalue; do
        CUDA_VISIBLE_DEVICES=1 python training/scripts/benchmark_expert.py \\
            --checkpoint runs/efficientnet/${expert}_expert/best.pt \\
            --dataset ${expert}_dataset_split
    done
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import ImageFile
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_fscore_support, accuracy_score
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

# Classes considered safety-critical (false negatives are dangerous)
DEADLY_KEYWORDS = {
    "deadly", "toxic", "poison", "death", "destroying",
    "galerina", "amanita_phalloides", "conocybe",
}


def is_safety_critical(class_name: str) -> bool:
    name = class_name.lower()
    return any(kw in name for kw in DEADLY_KEYWORDS)


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark expert classifier")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset",    required=True,
                   help="Dataset with val/ subdir (ImageFolder format)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers",    type=int, default=4)
    return p.parse_args()


@torch.no_grad()
def run_inference(model, loader, device):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device)
        logits = model(images).cpu()
        probs  = torch.softmax(logits, dim=1).numpy()
        preds  = np.argmax(probs, axis=1)

        all_preds.append(preds)
        all_labels.append(labels.numpy())
        all_probs.append(probs)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_labels),
        np.concatenate(all_probs),
    )


def plot_confusion_matrix(cm, classes, output_path, model_name):
    n = len(classes)
    fig_size = max(12, n * 0.9)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    # Normalise by row (true class) for recall-style display
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    tick_marks = np.arange(n)
    short_labels = [c.replace("_", "\n") for c in classes]
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(short_labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(short_labels, fontsize=7)

    # Annotate cells with raw counts
    thresh = 0.5
    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if count > 0:
                color = "white" if cm_norm[i, j] > thresh else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        fontsize=6, color=color)

    # Highlight deadly class rows/cols in red
    for idx, cls in enumerate(classes):
        if is_safety_critical(cls):
            ax.get_xticklabels()[idx].set_color("red")
            ax.get_yticklabels()[idx].set_color("red")

    ax.set_ylabel("True label", fontsize=11)
    ax.set_xlabel("Predicted label", fontsize=11)
    ax.set_title(f"{model_name} — Confusion Matrix (row-normalised)\n"
                 f"Red = safety-critical class", fontsize=12)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
                else os.path.join(repo_root, args.checkpoint)
    dataset_dir = args.dataset if os.path.isabs(args.dataset) \
                  else os.path.join(repo_root, args.dataset)
    val_dir = os.path.join(dataset_dir, "val")

    if not os.path.isdir(val_dir):
        print(f"ERROR: val/ not found in {dataset_dir}")
        sys.exit(1)

    # Load checkpoint
    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_name  = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    img_size    = ckpt.get("img_size", 224)
    classes     = ckpt["classes"]
    arch        = ckpt.get("arch", "tf_efficientnet_lite2")

    print(f"Model     : {model_name}  ({arch})")
    print(f"Classes   : {num_classes}")

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"Device    : {device}")

    val_transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)
    val_loader  = DataLoader(val_dataset, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.workers,
                             pin_memory=True)

    print(f"Val imgs  : {len(val_dataset)}")

    # Run inference
    print("\nRunning inference ...")
    preds, labels, probs = run_inference(model, val_loader, device)

    # ── Metrics ──────────────────────────────────────────────────────────────
    overall_acc = accuracy_score(labels, preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))

    # Top-5 confidence per class (avg)
    top1_per_class = [
        np.mean(probs[labels == i, i]) if (labels == i).sum() > 0 else 0.0
        for i in range(num_classes)
    ]

    # ── Print report ─────────────────────────────────────────────────────────
    width = 45
    print(f"\n{'='*75}")
    print(f"  {model_name}  —  Benchmark Results")
    print(f"{'='*75}")
    print(f"  Overall accuracy : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print(f"  Val samples      : {len(val_dataset)}")
    print(f"\n  {'Class':<{width}}  {'N':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AvgConf':>8}")
    print(f"  {'-'*width}  {'-----':>5}  {'------':>6}  {'------':>6}  {'------':>6}  {'--------':>8}")

    deadly_warnings = []
    for i, cls in enumerate(classes):
        flag = " ⚠ " if is_safety_critical(cls) else "   "
        n    = int(support[i])
        p    = precision[i]
        r    = recall[i]
        f    = f1[i]
        conf = top1_per_class[i]
        print(f"{flag}{cls:<{width}}  {n:>5}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {conf:>8.3f}")

        if is_safety_critical(cls) and r < 0.90:
            deadly_warnings.append((cls, r, p))

    print(f"\n  {'='*75}")

    if deadly_warnings:
        print(f"\n  ⚠  SAFETY WARNINGS — deadly class recall < 90%:")
        for cls, r, p in deadly_warnings:
            print(f"     {cls}: recall={r:.3f}  precision={p:.3f}")
    else:
        print(f"\n  ✓  All safety-critical classes: recall ≥ 90%")

    # Top confusion pairs
    print(f"\n  Top confusion pairs (predicted wrong):")
    off_diag = []
    for i in range(num_classes):
        for j in range(num_classes):
            if i != j and cm[i, j] > 0:
                off_diag.append((cm[i, j], classes[i], classes[j]))
    off_diag.sort(reverse=True)
    for count, true_cls, pred_cls in off_diag[:10]:
        flag = "⚠ " if is_safety_critical(true_cls) or is_safety_critical(pred_cls) else "  "
        print(f"    {flag}{true_cls} → {pred_cls}  ({count}x)")

    print(f"\n{'='*75}\n")

    # ── Save outputs ─────────────────────────────────────────────────────────
    output_dir = os.path.dirname(ckpt_path)

    # Confusion matrix PNG
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(cm, classes, cm_path, model_name)

    # JSON results
    per_class = {}
    for i, cls in enumerate(classes):
        per_class[cls] = {
            "n":              int(support[i]),
            "precision":      round(float(precision[i]), 4),
            "recall":         round(float(recall[i]), 4),
            "f1":             round(float(f1[i]), 4),
            "avg_confidence": round(float(top1_per_class[i]), 4),
            "safety_critical": is_safety_critical(cls),
        }

    results = {
        "model":          model_name,
        "arch":           arch,
        "overall_accuracy": round(float(overall_acc), 4),
        "val_samples":    len(val_dataset),
        "per_class":      per_class,
        "safety_warnings": [
            {"class": cls, "recall": round(r, 4), "precision": round(p, 4)}
            for cls, r, p in deadly_warnings
        ],
    }

    bench_path = os.path.join(output_dir, "benchmark.json")
    with open(bench_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {bench_path}")


if __name__ == "__main__":
    main()
