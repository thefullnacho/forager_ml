"""
benchmark_router.py — Domain router accuracy, confusion matrix, and threshold analysis.

The domain router is the first-pass gate in the two-stage inference pipeline.
This script measures:
  - Per-domain accuracy (berry / mushroom / plant / other)
  - Confusion matrix (which domains get misrouted?)
  - Confidence threshold sweep: routing accuracy vs abstention rate tradeoff
  - "Other" rejection rate at each threshold (the OOD gate metric)

Supports two checkpoint formats:
  - YOLO (yolov8n-cls best.pt via ultralytics)  — legacy router
  - EfficientNet (tf_efficientnet_lite2 best.pt via timm)  — new router

The checkpoint format is auto-detected from the file content.

Usage:
    # New EfficientNet router (recommended):
    CUDA_VISIBLE_DEVICES=1 python training/scripts/benchmark_router.py \\
        --checkpoint runs/efficientnet/domain_router/best.pt \\
        --dataset router_dataset

    # Legacy YOLO router:
    python training/scripts/benchmark_router.py \\
        --checkpoint runs/classify/domain_router/weights/best.pt \\
        --dataset router_dataset
"""

import argparse
import json
import os
import sys

import numpy as np
from pathlib import Path
from PIL import Image, ImageFile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, precision_recall_fscore_support, accuracy_score
)

ImageFile.LOAD_TRUNCATED_IMAGES = True

ROUTER_IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description="Benchmark domain router")
    p.add_argument("--checkpoint", required=True,
                   help="Path to YOLO best.pt")
    p.add_argument("--dataset", required=True,
                   help="Dataset root with val/ subdir (ImageFolder layout)")
    p.add_argument("--threshold", type=float, default=None,
                   help="Confidence threshold to evaluate (default: sweep 0.50–0.90)")
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()


def load_val_images(val_dir: Path) -> tuple[list[np.ndarray], list[int], list[str]]:
    """
    Load all val images as resized numpy arrays.
    Returns: (images, labels, class_names)
    """
    class_dirs = sorted([d for d in val_dir.iterdir() if d.is_dir()])
    class_names = [d.name for d in class_dirs]
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    images, labels = [], []
    exts = {".jpg", ".jpeg", ".png", ".webp"}

    for cls_dir in class_dirs:
        idx = class_to_idx[cls_dir.name]
        for f in sorted(cls_dir.iterdir()):
            if f.suffix.lower() in exts:
                try:
                    img = Image.open(f).convert("RGB")
                    img = img.resize((ROUTER_IMG_SIZE, ROUTER_IMG_SIZE), Image.BILINEAR)
                    images.append(np.array(img))
                    labels.append(idx)
                except Exception:
                    continue

    print(f"  Loaded {len(images)} val images across {len(class_names)} classes")
    for cls in class_names:
        n = labels.count(class_to_idx[cls])
        print(f"    {cls}: {n}")

    return images, labels, class_names


def load_checkpoint(ckpt_path: Path):
    """
    Auto-detect and load either a YOLO or EfficientNet checkpoint.
    Returns (model, class_names, model_type) where model_type is 'yolo' or 'efficientnet'.
    """
    import torch

    # Try loading as EfficientNet checkpoint (dict with 'model_state_dict')
    try:
        ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=True)
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            import timm
            arch = ckpt.get("arch", "tf_efficientnet_lite2")
            classes = ckpt["classes"]
            num_classes = ckpt["num_classes"]
            model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
            model.load_state_dict(ckpt["model_state_dict"])
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            print(f"  Format: EfficientNet ({arch})")
            print(f"  Classes: {classes}")
            return model, classes, "efficientnet"
    except Exception:
        pass

    # Fall back to YOLO
    from ultralytics import YOLO
    model = YOLO(str(ckpt_path))
    model.model.eval()
    yolo_names = [model.names[i] for i in range(len(model.names))]
    print(f"  Format: YOLO (yolov8n-cls)")
    print(f"  Classes: {yolo_names}")
    return model, yolo_names, "yolo"


def run_yolo_inference(model, images: list[np.ndarray], batch_size: int = 64) -> np.ndarray:
    """
    Run YOLO classification inference and return class probabilities.
    shape: (N, num_classes)
    """
    all_probs = []

    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        results = model.predict(batch, verbose=False, imgsz=ROUTER_IMG_SIZE)

        for r in results:
            probs = r.probs.data.cpu().numpy()
            all_probs.append(probs)

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(images))
            print(f"  [{done}/{len(images)}] ...", end="\r")

    print()
    return np.stack(all_probs, axis=0)


@torch.no_grad()
def run_efficientnet_inference(model, images: list[np.ndarray], batch_size: int = 64,
                               img_size: int = 224) -> np.ndarray:
    """
    Run EfficientNet inference and return class probabilities.
    shape: (N, num_classes)
    """
    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    device = next(model.parameters()).device
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    all_probs = []

    for i in range(0, len(images), batch_size):
        batch_np = images[i:i + batch_size]
        batch_t  = torch.stack([transform(Image.fromarray(img)) for img in batch_np])
        batch_t  = batch_t.to(device)

        logits = model(batch_t)
        probs  = F.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

        if (i // batch_size) % 10 == 0:
            done = min(i + batch_size, len(images))
            print(f"  [{done}/{len(images)}] ...", end="\r")

    print()
    return np.concatenate(all_probs, axis=0)


def plot_confusion_matrix(cm, classes, output_path):
    n = len(classes)
    fig, ax = plt.subplots(figsize=(8, 7))

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=11)
    ax.set_yticklabels(classes, fontsize=11)

    for i in range(n):
        for j in range(n):
            count = cm[i, j]
            if count > 0:
                color = "white" if cm_norm[i, j] > 0.5 else "black"
                ax.text(j, i, str(count), ha="center", va="center",
                        fontsize=9, color=color)

    ax.set_ylabel("True domain", fontsize=12)
    ax.set_xlabel("Predicted domain", fontsize=12)
    ax.set_title("Domain Router — Confusion Matrix (row-normalised)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def plot_threshold_curves(thresholds, routed_acc, abstention_rate, other_rejection,
                          output_path, current_threshold):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: routing accuracy vs abstention rate
    ax = axes[0]
    ax.plot(abstention_rate, routed_acc, "b-o", markersize=4, label="Routing accuracy")
    ax.axhline(0.95, color="green", linestyle="--", alpha=0.7, label="95% target")
    ax.axhline(0.90, color="orange", linestyle="--", alpha=0.7, label="90% target")
    if current_threshold is not None:
        # Find closest threshold
        thr_arr = np.array(thresholds)
        idx = np.argmin(np.abs(thr_arr - current_threshold))
        ax.axvline(abstention_rate[idx], color="red", linestyle=":", alpha=0.8,
                   label=f"Current threshold ({current_threshold:.2f})")
    ax.set_xlabel("Abstention rate (% routed to 'unknown')", fontsize=11)
    ax.set_ylabel("Routing accuracy on accepted frames", fontsize=11)
    ax.set_title("Routing Accuracy vs Abstention", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0.5, 1.01)

    # Right: threshold vs other-rejection and domain-accuracy
    ax2 = axes[1]
    ax2.plot(thresholds, other_rejection, "r-o", markersize=4, label="'Other' rejection rate")
    ax2.plot(thresholds, routed_acc, "b-s", markersize=4, label="Domain accuracy (accepted)")
    ax2.axhline(0.90, color="gray", linestyle="--", alpha=0.5)
    if current_threshold is not None:
        ax2.axvline(current_threshold, color="red", linestyle=":", alpha=0.8,
                    label=f"Current ({current_threshold:.2f})")
    ax2.set_xlabel("Confidence threshold", fontsize=11)
    ax2.set_ylabel("Rate", fontsize=11)
    ax2.set_title("Threshold vs Rejection & Accuracy", fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(0.4, 1.0)
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


def threshold_analysis(probs, labels, classes, class_to_idx):
    """
    Sweep confidence threshold. At each threshold:
      - abstention_rate: fraction of samples below threshold (routed to 'unknown')
      - routed_accuracy: accuracy on samples AT or ABOVE threshold
      - other_rejection: fraction of 'other' class samples correctly abstained (or correctly predicted 'other')
    """
    other_idx = class_to_idx.get("other", -1)
    # Exclude 'other' from routing accuracy (it's the OOD class)
    target_mask = np.array(labels) != other_idx  # True for berry/mushroom/plant
    other_mask  = np.array(labels) == other_idx

    top_confs = np.max(probs, axis=1)
    top_preds = np.argmax(probs, axis=1)

    thresholds = np.arange(0.40, 0.95, 0.02)
    results = []

    for thr in thresholds:
        accepted = top_confs >= thr

        # Abstention rate = fraction of ALL samples below threshold
        abstention = float(1.0 - np.mean(accepted))

        # Routing accuracy: among target-domain samples that were accepted, how many correct?
        target_accepted = target_mask & accepted
        if target_accepted.sum() > 0:
            route_acc = float(np.mean(top_preds[target_accepted] == np.array(labels)[target_accepted]))
        else:
            route_acc = 1.0  # edge case: nothing accepted

        # Other rejection: fraction of 'other' samples NOT confidently routed to a target domain
        # = (below threshold) OR (predicted 'other')
        if other_mask.sum() > 0:
            other_rejected = ~accepted[other_mask] | (top_preds[other_mask] == other_idx)
            other_rej_rate = float(np.mean(other_rejected))
        else:
            other_rej_rate = 0.0

        results.append({
            "threshold": round(float(thr), 3),
            "abstention_rate": round(abstention, 4),
            "routing_accuracy": round(route_acc, 4),
            "other_rejection_rate": round(other_rej_rate, 4),
        })

    return results


def main():
    args = parse_args()

    repo_root = Path(__file__).resolve().parents[2]

    ckpt_path = Path(args.checkpoint) if os.path.isabs(args.checkpoint) \
                else repo_root / args.checkpoint
    dataset_dir = Path(args.dataset) if os.path.isabs(args.dataset) \
                  else repo_root / args.dataset
    val_dir = dataset_dir / "val"

    if not val_dir.is_dir():
        print(f"ERROR: val/ not found in {dataset_dir}")
        sys.exit(1)

    if not ckpt_path.is_file():
        print(f"ERROR: checkpoint not found: {ckpt_path}")
        sys.exit(1)

    # ── Load model (auto-detect format) ─────────────────────────────────────
    print(f"\nLoading router: {ckpt_path}")
    import torch
    model, model_class_names, model_type = load_checkpoint(ckpt_path)

    # ── Load val images ──────────────────────────────────────────────────────
    print(f"\nLoading val set: {val_dir}")
    images, labels, class_names = load_val_images(val_dir)
    N = len(images)

    # ── Inference ────────────────────────────────────────────────────────────
    print(f"\nRunning inference on {N} images ...")
    if model_type == "yolo":
        probs = run_yolo_inference(model, images, batch_size=args.batch_size)
    else:
        probs = run_efficientnet_inference(model, images, batch_size=args.batch_size)

    # Sanity-check class count
    if probs.shape[1] != len(model_class_names):
        print(f"ERROR: probs has {probs.shape[1]} columns but model has {len(model_class_names)} classes")
        sys.exit(1)

    # Check for missing classes
    missing = set(class_names) - set(model_class_names)
    if probs.shape[1] != len(class_names) or missing:
        print(f"\nWARNING: model was trained on {probs.shape[1]} classes, val/ has {len(class_names)}")
        if missing:
            print(f"\n  CRITICAL: Model is missing classes: {missing}")
            print(f"  The router has NO concept of these domains — they will always be misclassified.")
            print(f"  Router needs retraining with all {len(class_names)} classes.\n")

    model_class_to_idx = {c: i for i, c in enumerate(model_class_names)}
    folder_classes_sorted = sorted([d.name for d in val_dir.iterdir() if d.is_dir()])

    # Build remapped labels array: val folder index → model class index (or -1 if unknown)
    folder_to_model = {}
    for fc in folder_classes_sorted:
        folder_to_model[fc] = model_class_to_idx.get(fc, -1)

    labels_remapped = np.array([
        folder_to_model[folder_classes_sorted[l]] for l in labels
    ])

    # Use remapped labels for metrics; -1 means "unknown to model" (always wrong)
    labels_arr = labels_remapped
    class_names_model = yolo_names  # what the model actually knows

    top_preds = np.argmax(probs, axis=1)

    # ── Overall metrics (no threshold) ───────────────────────────────────────
    # Only compute metrics on classes the model knows (labels_arr >= 0)
    known_mask = labels_arr >= 0
    n_unknown = int(np.sum(~known_mask))

    labels_known = labels_arr[known_mask]
    preds_known  = top_preds[known_mask]
    probs_known  = probs[known_mask]

    overall_acc = accuracy_score(labels_known, preds_known)
    n_model_classes = len(class_names_model)
    precision, recall, f1, support = precision_recall_fscore_support(
        labels_known, preds_known, labels=list(range(n_model_classes)), zero_division=0
    )
    cm = confusion_matrix(labels_known, preds_known, labels=list(range(n_model_classes)))

    print(f"\n{'='*65}")
    print(f"  Domain Router — Benchmark Results")
    print(f"{'='*65}")
    if n_unknown > 0:
        print(f"  !! Model trained on {n_model_classes} classes; val has {len(folder_classes_sorted)} "
              f"({n_unknown} samples from unknown classes always wrong)")
    print(f"  Accuracy on known classes : {overall_acc:.4f}  ({overall_acc*100:.2f}%)")
    print(f"  Val samples total         : {N}  (known: {N - n_unknown}, unknown: {n_unknown})")
    effective_acc = float(np.sum(labels_known == preds_known)) / N
    print(f"  Effective accuracy (all)  : {effective_acc:.4f}  ({effective_acc*100:.2f}%)")
    print(f"\n  {'Domain':<15}  {'N':>5}  {'Prec':>6}  {'Rec':>6}  {'F1':>6}  {'AvgConf':>8}")
    print(f"  {'-'*15}  {'-----':>5}  {'------':>6}  {'------':>6}  {'------':>6}  {'--------':>8}")

    avg_conf_per_class = []
    for i, cls in enumerate(class_names_model):
        n    = int(support[i])
        p    = precision[i]
        r    = recall[i]
        f    = f1[i]
        mask = labels_known == i
        conf = float(np.mean(probs_known[mask, i])) if mask.sum() > 0 else 0.0
        avg_conf_per_class.append(conf)
        print(f"  {cls:<15}  {n:>5}  {p:>6.3f}  {r:>6.3f}  {f:>6.3f}  {conf:>8.3f}")

    # Unknown classes (not in model) — always misclassified
    if n_unknown > 0:
        unknown_classes = [c for c in folder_classes_sorted if c not in model_class_to_idx]
        for uc in unknown_classes:
            n_uc = int(np.sum(np.array([folder_classes_sorted[l] for l in labels]) == uc))
            # What does the model predict for these?
            uc_folder_idx = folder_classes_sorted.index(uc)
            uc_pred_mask = np.array([l == uc_folder_idx for l in labels])
            uc_top_preds = top_preds[uc_pred_mask]
            uc_probs_top = np.max(probs[uc_pred_mask], axis=1)
            most_common = np.bincount(uc_top_preds, minlength=n_model_classes)
            mc_idx = int(np.argmax(most_common))
            mc_name = class_names_model[mc_idx] if mc_idx < len(class_names_model) else "?"
            print(f"  {uc:<15}  {n_uc:>5}  {'N/A':>6}  {'N/A':>6}  {'N/A':>6}  "
                  f"{np.mean(uc_probs_top):>8.3f}  ← not in model, mostly → {mc_name} ({most_common[mc_idx]}x)")

    print(f"\n  Top confusion pairs (known classes):")
    off_diag = []
    for i in range(n_model_classes):
        for j in range(n_model_classes):
            if i != j and cm[i, j] > 0:
                off_diag.append((cm[i, j], class_names_model[i], class_names_model[j]))
    off_diag.sort(reverse=True)
    for count, true_cls, pred_cls in off_diag[:8]:
        print(f"    {true_cls} → {pred_cls}  ({count}x)")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\n  Threshold sweep (confidence ≥ threshold → route; else abstain):")
    # For threshold sweep, use ALL samples (unknown class samples are always wrong, low confidence)
    thresh_results = threshold_analysis(probs, labels_arr, class_names_model, model_class_to_idx)

    print(f"\n  {'Thr':>5}  {'RouteAcc':>9}  {'Abstention':>11}  {'OtherRej':>9}")
    print(f"  {'-----':>5}  {'---------':>9}  {'-----------':>11}  {'---------':>9}")
    for r in thresh_results:
        marker = " ← current" if abs(r["threshold"] - 0.60) < 0.01 else ""
        print(f"  {r['threshold']:>5.2f}  {r['routing_accuracy']:>9.3f}  "
              f"{r['abstention_rate']:>11.3f}  {r['other_rejection_rate']:>9.3f}{marker}")

    # Find the threshold where routing accuracy first hits 95%
    for r in thresh_results:
        if r["routing_accuracy"] >= 0.95:
            print(f"\n  First threshold reaching ≥95% routing accuracy: {r['threshold']:.2f}")
            print(f"    Abstention at that threshold: {r['abstention_rate']:.1%}")
            print(f"    Other-rejection at that threshold: {r['other_rejection_rate']:.1%}")
            break

    print(f"\n{'='*65}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = ckpt_path.parent

    # Confusion matrix
    cm_path = output_dir / "confusion_matrix_router.png"
    plot_confusion_matrix(cm, class_names_model, str(cm_path))

    # Threshold curves
    thrs = [r["threshold"] for r in thresh_results]
    accs = [r["routing_accuracy"] for r in thresh_results]
    abst = [r["abstention_rate"] for r in thresh_results]
    orej = [r["other_rejection_rate"] for r in thresh_results]
    curve_path = output_dir / "threshold_curves.png"
    plot_threshold_curves(thrs, accs, abst, orej, str(curve_path),
                          args.threshold or 0.60)

    # JSON
    per_class = {}
    for i, cls in enumerate(class_names_model):
        per_class[cls] = {
            "n": int(support[i]),
            "precision": round(float(precision[i]), 4),
            "recall": round(float(recall[i]), 4),
            "f1": round(float(f1[i]), 4),
            "avg_confidence": round(avg_conf_per_class[i], 4),
        }

    results_dict = {
        "checkpoint": str(ckpt_path),
        "model_classes": class_names_model,
        "val_classes": folder_classes_sorted,
        "missing_classes": [c for c in folder_classes_sorted if c not in model_class_to_idx],
        "accuracy_known_classes": round(float(overall_acc), 4),
        "effective_accuracy_all": round(float(effective_acc), 4),
        "val_samples_total": N,
        "val_samples_known": int(N - n_unknown),
        "per_domain": per_class,
        "threshold_sweep": thresh_results,
    }

    bench_path = output_dir / "benchmark_router.json"
    with open(bench_path, "w") as f:
        json.dump(results_dict, f, indent=2)
    print(f"  Saved: {bench_path}")


if __name__ == "__main__":
    main()
