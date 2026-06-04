"""
benchmark_ood.py — Measure OOD detection quality and find the optimal threshold.

The existing calibrate_energy_threshold.py only looks at in-distribution (ID)
energy scores and picks a percentile. That tells you "X% of real images pass"
but nothing about how many actual OOD inputs get caught.

This script adds OOD data to give the full picture:

  AUROC            — overall ID/OOD separation (1.0 = perfect, 0.5 = random)
  FPR@TPR95        — standard benchmark: what % of ID is falsely rejected when
                     95% of OOD is correctly caught
  Safety threshold — current p95 approach, but now with measured OOD recall
  Balanced threshold — best F1 on the OOD detection task

OOD data: use the other experts' val sets. Each expert's natural OOD is every
other domain — the domain router is the first line of defense, so the energy
check is catching router misroutes and completely off-domain inputs.

Example OOD sources per expert:
  berry_expert      →  psychedelics_dataset_split/val  highvalue_dataset_split/val
  psychedelics      →  berry_dataset_split/val         highvalue_dataset_split/val
  highvalue         →  berry_dataset_split/val         psychedelics_dataset_split/val
  medicinals        →  berry_dataset_split/val         psychedelics_dataset_split/val

Usage:
    CUDA_VISIBLE_DEVICES=1 python training/scripts/benchmark_ood.py \\
        --checkpoint runs/efficientnet/berry_expert/best.pt \\
        --id-dataset berry_dataset_split \\
        --ood-dataset psychedelics_dataset_split/val highvalue_dataset_split/val

Outputs (written next to the checkpoint):
    ood_benchmark.json   — metrics + all three threshold recommendations
    ood_benchmark.png    — energy distribution histogram + ROC curve
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
from PIL import ImageFile
import timm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve

ImageFile.LOAD_TRUNCATED_IMAGES = True


# ── Args ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Benchmark OOD detection quality")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--id-dataset", required=True,
                   help="Expert's own dataset root (uses val/ subdir)")
    p.add_argument("--ood-dataset", required=True, nargs="+",
                   help="One or more OOD dataset paths (val/ dirs or ImageFolder roots)")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--max-ood", type=int, default=5000,
                   help="Max OOD images to sample (avoids huge imbalance). Default 5000.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--workers", type=int, default=4)
    return p.parse_args()


# ── Energy ────────────────────────────────────────────────────────────────────

def compute_energy(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """E(x) = -T * log(sum(exp(logits / T))). Higher = more OOD."""
    scaled = logits / temperature
    # Subtract max per row for numerical stability, then add it back:
    # E(x) = -T * logsumexp(L/T);  logsumexp = max + log(sum(exp(L/T - max)))
    mx = scaled.max(axis=1, keepdims=True)
    shifted = scaled - mx
    return -temperature * (mx.squeeze(axis=1) + np.log(np.sum(np.exp(shifted), axis=1)))


# ── Inference ─────────────────────────────────────────────────────────────────

@torch.no_grad()
def collect_logits(model, loader, device) -> np.ndarray:
    model.eval()
    all_logits = []
    for images, _ in loader:
        images = images.to(device)
        logits = model(images).cpu().numpy()
        all_logits.append(logits)
    return np.concatenate(all_logits)


# ── Dataset helpers ───────────────────────────────────────────────────────────

def resolve_val_dir(path: str) -> str:
    """Accept either a dataset root (has val/ subdir) or a val/ dir directly."""
    if os.path.isdir(os.path.join(path, "val")):
        return os.path.join(path, "val")
    return path


def build_loader(val_dir: str, img_size: int, batch_size: int,
                 workers: int, max_samples: int = 0) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(val_dir, transform=transform)
    if max_samples and len(ds) > max_samples:
        indices = np.random.choice(len(ds), max_samples, replace=False)
        ds = torch.utils.data.Subset(ds, indices)
    return DataLoader(ds, batch_size=batch_size, shuffle=False,
                      num_workers=workers, pin_memory=True), len(ds)


# ── Threshold analysis ────────────────────────────────────────────────────────

def find_threshold_at_tpr(ood_energies: np.ndarray, tpr_target: float) -> float:
    """Threshold where TPR (OOD recall) = tpr_target. Uses OOD percentile."""
    # TPR = fraction of OOD with energy > threshold
    # TPR = tpr_target → threshold = (1 - tpr_target) percentile of OOD energies
    return float(np.percentile(ood_energies, (1 - tpr_target) * 100))


def compute_fpr_at_threshold(id_energies: np.ndarray, threshold: float) -> float:
    """Fraction of ID images falsely rejected (energy > threshold)."""
    return float(np.mean(id_energies > threshold))


def compute_tpr_at_threshold(ood_energies: np.ndarray, threshold: float) -> float:
    """Fraction of OOD images correctly rejected (energy > threshold)."""
    return float(np.mean(ood_energies > threshold))


def find_best_f1_threshold(id_energies: np.ndarray,
                           ood_energies: np.ndarray) -> tuple[float, float]:
    """Sweep thresholds to find the one maximising F1 for OOD detection."""
    all_energies = np.concatenate([id_energies, ood_energies])
    thresholds = np.percentile(all_energies, np.linspace(1, 99, 500))

    best_f1, best_t = 0.0, thresholds[0]
    for t in thresholds:
        tp = np.sum(ood_energies > t)
        fp = np.sum(id_energies > t)
        fn = np.sum(ood_energies <= t)
        if tp + fp == 0 or tp + fn == 0:
            continue
        precision = tp / (tp + fp)
        recall    = tp / (tp + fn)
        if precision + recall == 0:
            continue
        f1 = 2 * precision * recall / (precision + recall)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t), float(best_f1)


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_ood_benchmark(id_energies, ood_energies, thresholds_info,
                       auroc, model_name, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # ── Energy distribution histogram ────────────────────────────────────────
    bins = np.linspace(
        min(id_energies.min(), ood_energies.min()),
        max(id_energies.max(), ood_energies.max()),
        80
    )
    ax1.hist(id_energies,  bins=bins, alpha=0.6, color="steelblue",
             label=f"In-distribution (n={len(id_energies):,})", density=True)
    ax1.hist(ood_energies, bins=bins, alpha=0.6, color="tomato",
             label=f"OOD (n={len(ood_energies):,})", density=True)

    colors = {"safety (p95 ID)": "navy", "balanced (best F1)": "darkorange",
              "aggressive (TPR95)": "crimson"}
    for label, info in thresholds_info.items():
        ax1.axvline(info["threshold"], color=colors.get(label, "grey"),
                    linestyle="--", linewidth=1.5,
                    label=f"{label}: {info['threshold']:.3f}")

    ax1.set_xlabel("Energy score")
    ax1.set_ylabel("Density")
    ax1.set_title(f"{model_name}\nEnergy distributions: ID vs OOD")
    ax1.legend(fontsize=8)

    # ── ROC curve ─────────────────────────────────────────────────────────────
    labels  = np.concatenate([np.zeros(len(id_energies)), np.ones(len(ood_energies))])
    scores  = np.concatenate([id_energies, ood_energies])
    fpr_arr, tpr_arr, _ = roc_curve(labels, scores)

    ax2.plot(fpr_arr, tpr_arr, color="steelblue", lw=2,
             label=f"ROC curve (AUROC = {auroc:.4f})")
    ax2.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4)

    marker_colors = {"safety (p95 ID)": "navy", "balanced (best F1)": "darkorange",
                     "aggressive (TPR95)": "crimson"}
    for label, info in thresholds_info.items():
        ax2.scatter(info["fpr"], info["tpr"], s=80, zorder=5,
                    color=marker_colors.get(label, "grey"),
                    label=f"{label}  FPR={info['fpr']:.3f}  TPR={info['tpr']:.3f}")

    ax2.set_xlabel("False Positive Rate (ID rejected)")
    ax2.set_ylabel("True Positive Rate (OOD caught)")
    ax2.set_title(f"{model_name}\nROC curve — OOD detection")
    ax2.legend(fontsize=8)
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    np.random.seed(42)

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    ckpt_path = args.checkpoint if os.path.isabs(args.checkpoint) \
                else os.path.join(repo_root, args.checkpoint)

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    model_name  = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    img_size    = ckpt.get("img_size", 224)
    arch        = ckpt.get("arch", "tf_efficientnet_lite2")

    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)
    print(f"Model : {model_name}  ({num_classes} classes)  device={device}")

    # ── ID dataset (val split) ────────────────────────────────────────────────
    id_dir = resolve_val_dir(
        args.id_dataset if os.path.isabs(args.id_dataset)
        else os.path.join(repo_root, args.id_dataset)
    )
    id_loader, n_id = build_loader(id_dir, img_size, args.batch_size, args.workers)
    print(f"\nID  : {id_dir}  ({n_id:,} images)")

    print("Running ID inference ...")
    id_logits  = collect_logits(model, id_loader, device)
    id_energies = compute_energy(id_logits, args.temperature)

    # ── OOD datasets ──────────────────────────────────────────────────────────
    ood_logits_list = []
    ood_n_total = 0
    for ood_path in args.ood_dataset:
        ood_dir = resolve_val_dir(
            ood_path if os.path.isabs(ood_path)
            else os.path.join(repo_root, ood_path)
        )
        loader, n = build_loader(ood_dir, img_size, args.batch_size, args.workers,
                                 max_samples=args.max_ood // len(args.ood_dataset))
        print(f"OOD : {ood_dir}  ({n:,} images)")
        logits = collect_logits(model, loader, device)
        ood_logits_list.append(logits)
        ood_n_total += n

    ood_energies = compute_energy(np.concatenate(ood_logits_list), args.temperature)

    # ── AUROC ─────────────────────────────────────────────────────────────────
    labels = np.concatenate([np.zeros(len(id_energies)), np.ones(len(ood_energies))])
    scores = np.concatenate([id_energies, ood_energies])
    auroc  = float(roc_auc_score(labels, scores))

    # ── Threshold recommendations ─────────────────────────────────────────────
    #
    # 1. Safety (p95 ID) — current approach: 5% ID false rejection, measured OOD recall
    safety_t   = float(np.percentile(id_energies, 95))
    safety_fpr = compute_fpr_at_threshold(id_energies, safety_t)   # should be ~0.05
    safety_tpr = compute_tpr_at_threshold(ood_energies, safety_t)

    # 2. Balanced — best F1 on OOD detection task
    balanced_t, balanced_f1 = find_best_f1_threshold(id_energies, ood_energies)
    balanced_fpr = compute_fpr_at_threshold(id_energies, balanced_t)
    balanced_tpr = compute_tpr_at_threshold(ood_energies, balanced_t)

    # 3. Aggressive (TPR95) — catch 95% of OOD, show the FPR cost
    aggressive_t   = find_threshold_at_tpr(ood_energies, tpr_target=0.95)
    aggressive_fpr = compute_fpr_at_threshold(id_energies, aggressive_t)
    aggressive_tpr = compute_tpr_at_threshold(ood_energies, aggressive_t)

    thresholds_info = {
        "safety (p95 ID)":     {"threshold": safety_t,     "fpr": safety_fpr,     "tpr": safety_tpr},
        "balanced (best F1)":  {"threshold": balanced_t,   "fpr": balanced_fpr,   "tpr": balanced_tpr,  "f1": balanced_f1},
        "aggressive (TPR95)":  {"threshold": aggressive_t, "fpr": aggressive_fpr, "tpr": aggressive_tpr},
    }

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  {model_name} — OOD Benchmark")
    print(f"{'='*70}")
    print(f"  ID images  : {len(id_energies):,}")
    print(f"  OOD images : {len(ood_energies):,}")
    print(f"\n  AUROC      : {auroc:.4f}  {'(excellent)' if auroc > 0.90 else '(moderate)' if auroc > 0.75 else '(poor — ID/OOD overlap is high)'}")
    print(f"\n  {'Threshold':<26}  {'Value':>8}  {'FPR (ID rejected)':>18}  {'TPR (OOD caught)':>17}")
    print(f"  {'-'*26}  {'--------':>8}  {'------------------':>18}  {'-----------------':>17}")
    for label, info in thresholds_info.items():
        f1_str = f"  F1={info['f1']:.3f}" if "f1" in info else ""
        print(f"  {label:<26}  {info['threshold']:>8.4f}  {info['fpr']:>18.3f}  {info['tpr']:>17.3f}{f1_str}")

    print(f"\n  ID energy stats:  mean={id_energies.mean():.4f}  std={id_energies.std():.4f}  "
          f"p5={np.percentile(id_energies,5):.4f}  p95={np.percentile(id_energies,95):.4f}")
    print(f"  OOD energy stats: mean={ood_energies.mean():.4f}  std={ood_energies.std():.4f}  "
          f"p5={np.percentile(ood_energies,5):.4f}  p95={np.percentile(ood_energies,95):.4f}")

    # Recommendation
    print(f"\n  Recommendation:")
    if auroc > 0.90:
        rec = thresholds_info["balanced (best F1)"]
        print(f"    AUROC {auroc:.3f} — good separation. Use 'balanced' threshold")
        print(f"    ({balanced_t:.4f}): catches {balanced_tpr:.1%} of OOD at {balanced_fpr:.1%} ID false rejection.")
    elif auroc > 0.75:
        rec = thresholds_info["safety (p95 ID)"]
        print(f"    AUROC {auroc:.3f} — moderate separation. Energy scores partially")
        print(f"    overlap. 'Safety' threshold retains p95 ID behaviour but only catches")
        print(f"    {safety_tpr:.1%} of OOD. Consider adding more diverse OOD training data.")
    else:
        print(f"    AUROC {auroc:.3f} — poor separation. Energy-based OOD may not be reliable")
        print(f"    for this expert. Consider temperature scaling or a dedicated OOD head.")

    print(f"{'='*70}\n")

    # ── Save outputs ──────────────────────────────────────────────────────────
    output_dir = os.path.dirname(ckpt_path)

    # Plot
    plot_path = os.path.join(output_dir, "ood_benchmark.png")
    plot_ood_benchmark(id_energies, ood_energies, thresholds_info,
                       auroc, model_name, plot_path)

    # JSON
    results = {
        "model":           model_name,
        "auroc":           round(auroc, 4),
        "n_id":            int(len(id_energies)),
        "n_ood":           int(len(ood_energies)),
        "temperature":     args.temperature,
        "thresholds": {
            label: {
                "threshold": round(info["threshold"], 4),
                "fpr":       round(info["fpr"], 4),
                "tpr":       round(info["tpr"], 4),
                **( {"f1": round(info["f1"], 4)} if "f1" in info else {} ),
            }
            for label, info in thresholds_info.items()
        },
        "id_energy_stats": {
            "mean": round(float(id_energies.mean()), 4),
            "std":  round(float(id_energies.std()),  4),
            "p5":   round(float(np.percentile(id_energies, 5)),  4),
            "p95":  round(float(np.percentile(id_energies, 95)), 4),
        },
        "ood_energy_stats": {
            "mean": round(float(ood_energies.mean()), 4),
            "std":  round(float(ood_energies.std()),  4),
            "p5":   round(float(np.percentile(ood_energies, 5)),  4),
            "p95":  round(float(np.percentile(ood_energies, 95)), 4),
        },
    }

    json_path = os.path.join(output_dir, "ood_benchmark.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved: {json_path}")

    # Update the inference energy_calibration.json with the balanced threshold
    # (user can manually override to safety or aggressive if preferred)
    calib_path = os.path.join(output_dir, "energy_calibration.json")
    if os.path.exists(calib_path):
        with open(calib_path) as f:
            calib = json.load(f)
    else:
        calib = {"model": model_name, "temperature": args.temperature}

    calib["threshold_p95"]         = round(safety_t, 4)        # keep for reference
    calib["threshold_balanced"]    = round(balanced_t, 4)
    calib["threshold_aggressive"]  = round(aggressive_t, 4)
    calib["auroc"]                 = round(auroc, 4)
    # Default deployment threshold = balanced (can be changed to aggressive for safety-critical use)
    calib["threshold_p95"]         = round(balanced_t, 4)

    with open(calib_path, "w") as f:
        json.dump(calib, f, indent=2)

    # Mirror to inference/models/
    inference_path = os.path.join(repo_root, "inference", "models",
                                  f"{model_name}_energy.json")
    if os.path.exists(os.path.dirname(inference_path)):
        with open(inference_path, "w") as f:
            json.dump(calib, f, indent=2)
        print(f"  Updated: {inference_path}")

    print(f"\nDone. Review ood_benchmark.png to inspect ID/OOD energy overlap.")


if __name__ == "__main__":
    main()
