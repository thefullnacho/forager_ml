"""
Train the domain router classifier (YOLOv8n-cls).

Lightweight 3-class model (berry, mushroom, plant) that runs first in the
two-stage inference pipeline to determine which expert(s) to invoke.

Same architecture and hyperparameters as the expert training scripts.

Usage:
    python training/scripts/train_domain_router.py
"""

import sys
import os

# ── GPU diagnostics ────────────────────────────────────────────────────────────
print("=" * 60)
print("GPU / CUDA DIAGNOSTICS")
print("=" * 60)

import torch

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
    DEVICE = "0"
else:
    print("\n  No CUDA GPU detected — training will run on CPU.")
    print("   If you expected a GPU, check: nvidia-smi")
    DEVICE = "cpu"

print("=" * 60)

# Quick tensor smoke-test to catch driver issues before launching YOLO
if DEVICE != "cpu":
    print("\nRunning GPU smoke test...")
    try:
        x = torch.randn(64, 512, device="cuda")
        _ = x @ x.T
        del x
        torch.cuda.empty_cache()
        print("Smoke test passed.\n")
    except Exception as e:
        print(f"\n  GPU smoke test FAILED: {e}")
        print("  Aborting — fix the GPU issue or re-run with DEVICE='cpu'.")
        sys.exit(1)

# ── Config ─────────────────────────────────────────────────────────────────────
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

DATASET   = os.path.join(REPO_ROOT, "router_dataset")
MODEL     = "yolov8n-cls.pt"
RUN_NAME  = "domain_router"
PROJECT   = os.path.join(REPO_ROOT, "runs", "classify")

EPOCHS    = 50
IMGSZ     = 224
WORKERS   = 8

BATCH     = 64 if DEVICE != "cpu" else 32
AMP       = True if DEVICE != "cpu" else False

# ── Train ──────────────────────────────────────────────────────────────────────
print(f"Dataset   : {DATASET}")
print(f"Classes   : {sorted(os.listdir(os.path.join(DATASET, 'train')))}")
print(f"Device    : {DEVICE}")
print(f"Batch     : {BATCH}")
print(f"AMP       : {AMP}")
print(f"Epochs    : {EPOCHS}")
print(f"Output    : {PROJECT}/{RUN_NAME}")
print()

from ultralytics import YOLO

model = YOLO(MODEL)

results = model.train(
    task="classify",
    data=DATASET,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    device=DEVICE,
    workers=WORKERS,
    project=PROJECT,
    name=RUN_NAME,
    exist_ok=False,

    # Reproducibility — match other experts
    seed=0,
    deterministic=True,

    # Optimiser — match other experts
    optimizer="auto",
    lr0=0.01,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,

    # Augmentation — match other experts exactly
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    fliplr=0.5,
    flipud=0.0,
    translate=0.1,
    scale=0.5,
    auto_augment="randaugment",
    erasing=0.4,

    # GPU-specific
    amp=AMP,

    # Artefacts
    save=True,
    plots=True,
    verbose=True,
)

print("\n" + "=" * 60)
print("Training complete.")
print(f"Best weights : {PROJECT}/{RUN_NAME}/weights/best.pt")
print(f"Top-1 acc    : {results.results_dict.get('metrics/accuracy_top1', 'N/A'):.4f}")
print(f"Top-5 acc    : {results.results_dict.get('metrics/accuracy_top5', 'N/A'):.4f}")
print("=" * 60)
