"""
Rebuild train/val splits for all expert datasets from canonical flat directories.

Resolves:
  - Psychedelics split having more images than the flat source (stale split)
  - Ensures consistent 80/20 split ratio across all datasets
  - Incorporates any backfilled images (e.g. blueberry_highbush)
  - Validates image integrity (skips zero-byte / corrupt files)

For each dataset, reads from the flat directory (e.g. berry_dataset/) and
writes a clean split to the _split directory (e.g. berry_dataset_split/).

Usage:
    # Rebuild all splits
    python training/scripts/rebuild_dataset_splits.py

    # Rebuild a single dataset
    python training/scripts/rebuild_dataset_splits.py --only berry

    # Dry run — report what would happen without writing
    python training/scripts/rebuild_dataset_splits.py --dry-run
"""

import argparse
import os
import random
import shutil
import sys
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TRAIN_RATIO = 0.80
MIN_CLASS_SIZE = 50       # warn if any class has fewer than this
IMBALANCE_WARN = 10.0     # warn if max/min class ratio exceeds this
MIN_FILE_SIZE = 1024      # bytes — skip files smaller than 1KB (likely corrupt)

# ── Dataset definitions ──────────────────────────────────────────────────────
# Maps dataset name → (flat source dir, split output dir)

DATASETS = {
    "berry": (
        REPO_ROOT / "berry_dataset",
        REPO_ROOT / "berry_dataset_split",
    ),
    "psychedelics": (
        REPO_ROOT / "psychedelics_dataset",
        REPO_ROOT / "psychedelics_dataset_split",
    ),
    "high_value": (
        REPO_ROOT / "high_value_dataset",
        REPO_ROOT / "high_value_dataset_split",
    ),
}


def collect_valid_images(class_dir: Path) -> list[Path]:
    """
    Return image paths from a class directory, filtering out:
      - Non-image files
      - Zero-byte files
      - Files under MIN_FILE_SIZE (likely corrupt)
    """
    valid = []
    skipped = 0

    for f in class_dir.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() not in IMAGE_EXTS:
            continue
        if f.stat().st_size < MIN_FILE_SIZE:
            skipped += 1
            continue
        valid.append(f)

    if skipped > 0:
        print(f"    ⚠ {class_dir.name}: skipped {skipped} files < {MIN_FILE_SIZE} bytes")

    return sorted(valid)


def split_and_copy(
    images: list[Path],
    train_dir: Path,
    val_dir: Path,
    class_name: str,
):
    """Shuffle, split, and copy images to train/val directories."""
    shuffled = list(images)
    random.shuffle(shuffled)

    split_idx = int(len(shuffled) * TRAIN_RATIO)
    train_imgs = shuffled[:split_idx]
    val_imgs = shuffled[split_idx:]

    train_cls = train_dir / class_name
    val_cls = val_dir / class_name
    train_cls.mkdir(parents=True, exist_ok=True)
    val_cls.mkdir(parents=True, exist_ok=True)

    for src in train_imgs:
        shutil.copy2(src, train_cls / src.name)

    for src in val_imgs:
        shutil.copy2(src, val_cls / src.name)

    return len(train_imgs), len(val_imgs)


def rebuild_dataset(name: str, source_dir: Path, split_dir: Path, dry_run: bool):
    """Rebuild train/val split for one dataset."""
    print(f"\n{'='*60}")
    print(f"Dataset: {name}")
    print(f"  Source: {source_dir}")
    print(f"  Output: {split_dir}")
    print(f"{'='*60}")

    if not source_dir.is_dir():
        print(f"  ERROR: Source directory not found: {source_dir}")
        return False

    # Discover classes
    class_dirs = sorted([
        d for d in source_dir.iterdir()
        if d.is_dir()
    ])

    if not class_dirs:
        print(f"  ERROR: No class directories found in {source_dir}")
        return False

    # Collect and validate images per class
    class_data: dict[str, list[Path]] = {}
    total_images = 0

    for cls_dir in class_dirs:
        images = collect_valid_images(cls_dir)
        class_data[cls_dir.name] = images
        total_images += len(images)
        print(f"  {cls_dir.name:45s} {len(images):6d} images")

    print(f"  {'─'*55}")
    print(f"  {'TOTAL':45s} {total_images:6d} images")
    print(f"  Classes: {len(class_data)}")

    # Check for issues
    counts = [len(imgs) for imgs in class_data.values()]
    min_count = min(counts)
    max_count = max(counts)
    ratio = max_count / min_count if min_count > 0 else float("inf")

    issues = []

    if min_count < MIN_CLASS_SIZE:
        small_classes = [
            (name, len(imgs))
            for name, imgs in class_data.items()
            if len(imgs) < MIN_CLASS_SIZE
        ]
        for cls_name, cnt in small_classes:
            issues.append(f"CRITICAL: {cls_name} has only {cnt} images (< {MIN_CLASS_SIZE})")

    if ratio > IMBALANCE_WARN:
        issues.append(
            f"WARNING: Class imbalance ratio {ratio:.1f}x "
            f"(max={max_count}, min={min_count})"
        )

    empty_classes = [name for name, imgs in class_data.items() if len(imgs) == 0]
    if empty_classes:
        issues.append(f"CRITICAL: Empty classes: {empty_classes}")

    if issues:
        print(f"\n  Issues detected:")
        for issue in issues:
            print(f"    ⚠ {issue}")

    if dry_run:
        train_total = int(total_images * TRAIN_RATIO)
        val_total = total_images - train_total
        print(f"\n  [dry run] Would create: {train_total} train + {val_total} val")
        return len(issues) == 0

    # Clean and rebuild
    if split_dir.exists():
        print(f"\n  Removing existing split: {split_dir}")
        shutil.rmtree(split_dir)

    train_dir = split_dir / "train"
    val_dir = split_dir / "val"

    print(f"\n  Splitting ({TRAIN_RATIO:.0%} train / {1-TRAIN_RATIO:.0%} val) ...")

    train_total = 0
    val_total = 0

    for cls_name, images in sorted(class_data.items()):
        n_train, n_val = split_and_copy(images, train_dir, val_dir, cls_name)
        train_total += n_train
        val_total += n_val

    print(f"\n  ✓ Train: {train_total} images in {train_dir}")
    print(f"  ✓ Val  : {val_total} images in {val_dir}")
    print(f"  ✓ Total: {train_total + val_total}")

    return len(issues) == 0


# ── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Rebuild train/val dataset splits")
    p.add_argument("--only", choices=list(DATASETS.keys()),
                   help="Only rebuild this dataset")
    p.add_argument("--dry-run", action="store_true",
                   help="Report issues without writing")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Dataset Split Rebuilder")
    print(f"  Train ratio  : {TRAIN_RATIO:.0%}")
    print(f"  Min class    : {MIN_CLASS_SIZE} images")
    print(f"  Min file size: {MIN_FILE_SIZE} bytes")
    if args.dry_run:
        print(f"  Mode         : DRY RUN")
    print("=" * 60)

    targets = {args.only: DATASETS[args.only]} if args.only else DATASETS
    all_clean = True

    for name, (source, split) in targets.items():
        clean = rebuild_dataset(name, source, split, args.dry_run)
        if not clean:
            all_clean = False

    print(f"\n{'='*60}")
    if all_clean:
        print("All datasets clean.")
    else:
        print("Some datasets have issues — review warnings above.")
        print("Consider running backfill_inat_images.py for underrepresented classes.")
    print("=" * 60)


if __name__ == "__main__":
    main()
