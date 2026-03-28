"""
Build the domain router training dataset.

Samples images from existing expert datasets into four categories:
  berry     — all berry_dataset classes
  mushroom  — mushroom classes from high_value_dataset + all psychedelics_dataset (excl. ginseng)
  plant     — plant classes from high_value_dataset + inat plants + psychedelics ginseng
  other     — non-target nature images (animals, landscapes, insects, etc.)
              Acts as the first-pass OOD gate in the inference pipeline.

Creates router_dataset/train/ and router_dataset/val/ with an 80/20 split.
Target: ~15,000 images per class (~60,000 total).

The "other" class is sourced from iNaturalist animal/insect/landscape images
that have been pre-downloaded to inat_dataset/other/. If that directory doesn't
exist yet, run backfill_inat_images.py first:

    python training/scripts/backfill_inat_images.py \
        --taxon-id 47169 --output inat_dataset/other/animals --target 5000
    python training/scripts/backfill_inat_images.py \
        --taxon-id 47158 --output inat_dataset/other/insects --target 5000
    python training/scripts/backfill_inat_images.py \
        --taxon-id 48460 --output inat_dataset/other/lichens --target 3000
    python training/scripts/backfill_inat_images.py \
        --taxon-id 211194 --output inat_dataset/other/birds --target 2000

Note: iNat mushroom data (deadly/edible/medicinal) is intentionally excluded — those
bulk-scraped functional groupings contain too much label noise and drag accuracy down.

Usage:
    python training/scripts/build_router_dataset.py
"""

import os
import random
import shutil
from pathlib import Path

random.seed(42)

REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Source datasets ──────────────────────────────────────────────────────────

BERRY_DATASET        = REPO_ROOT / "berry_dataset"
HIGHVALUE_DATASET    = REPO_ROOT / "high_value_dataset"
PSYCHEDELICS_DATASET = REPO_ROOT / "psychedelics_dataset"
MEDICINALS_DATASET   = REPO_ROOT / "medicinals_dataset"
INAT_DATASET         = REPO_ROOT / "inat_dataset"

# ── Output ───────────────────────────────────────────────────────────────────

OUTPUT_DIR   = REPO_ROOT / "router_dataset"
TARGET_PER_CLASS = 15000
OTHER_DIR        = REPO_ROOT / "inat_dataset" / "other"
TRAIN_RATIO  = 0.80

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ── Domain mappings for highvalue_expert (mixed mushroom + plant classes) ────

HIGHVALUE_MUSHROOM_CLASSES = {
    "chanterelles_edible",
    "morels_edible",
    "chicken_of_the_woods",
    "lions_mane",
    "chaga_medicinal",
    "reishi_northeast",
    "high_value_toxics",
}

HIGHVALUE_PLANT_CLASSES = {
    "ginseng_american",
    "ramps_wild_leek",
    "ostrich_fern_fiddlehead",
    "saffron_crocus",
}

# ── Source definitions per router class ──────────────────────────────────────

ROUTER_CLASSES = {
    "berry": [],
    "mushroom": [],
    "plant": [],
    "other": [],
}


def collect_images(class_dir: Path) -> list[Path]:
    """Return all image paths in a single class directory."""
    return [
        p for p in class_dir.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS
    ]


def gather_sources():
    """Build the list of (source_dir, images) for each router class."""

    # ── Berry: all classes from berry_dataset ────────────────────────────────
    for cls_dir in sorted(BERRY_DATASET.iterdir()):
        if cls_dir.is_dir():
            imgs = collect_images(cls_dir)
            if imgs:
                ROUTER_CLASSES["berry"].append((cls_dir.name, imgs))

    # ── Mushroom sources ────────────────────────────────────────────────────
    # 1) Mushroom classes from highvalue_dataset
    for cls_dir in sorted(HIGHVALUE_DATASET.iterdir()):
        if cls_dir.is_dir() and cls_dir.name in HIGHVALUE_MUSHROOM_CLASSES:
            imgs = collect_images(cls_dir)
            if imgs:
                ROUTER_CLASSES["mushroom"].append((f"hv_{cls_dir.name}", imgs))

    # 2) Psychedelics_dataset classes (excluding ginseng — that's a plant)
    psych_plant_classes = {"panax_quinquefolius_ginseng_conservation"}
    for cls_dir in sorted(PSYCHEDELICS_DATASET.iterdir()):
        if cls_dir.is_dir() and cls_dir.name not in psych_plant_classes:
            imgs = collect_images(cls_dir)
            if imgs:
                ROUTER_CLASSES["mushroom"].append((f"psych_{cls_dir.name}", imgs))

    # Ginseng from psychedelics_dataset is a plant
    for cls_dir in sorted(PSYCHEDELICS_DATASET.iterdir()):
        if cls_dir.is_dir() and cls_dir.name in psych_plant_classes:
            imgs = collect_images(cls_dir)
            if imgs:
                ROUTER_CLASSES["plant"].append((f"psych_{cls_dir.name}", imgs))

    # NOTE: iNat mushroom dirs (deadly/edible/medicinal) intentionally excluded —
    # bulk-scraped functional groupings with too much label noise.

    # ── Plant sources ───────────────────────────────────────────────────────
    # 1) Plant classes from highvalue_dataset
    for cls_dir in sorted(HIGHVALUE_DATASET.iterdir()):
        if cls_dir.is_dir() and cls_dir.name in HIGHVALUE_PLANT_CLASSES:
            imgs = collect_images(cls_dir)
            if imgs:
                ROUTER_CLASSES["plant"].append((f"hv_{cls_dir.name}", imgs))

    # 2) Medicinals dataset (all classes are plants — safe and toxic lookalikes)
    if MEDICINALS_DATASET.is_dir():
        for cls_dir in sorted(MEDICINALS_DATASET.iterdir()):
            if cls_dir.is_dir():
                imgs = collect_images(cls_dir)
                if imgs:
                    ROUTER_CLASSES["plant"].append((f"med_{cls_dir.name}", imgs))
    else:
        print(f"  NOTE: {MEDICINALS_DATASET} not found — medicinals will be absent from plant class.")
        print(f"        Run: python data/acquisition/medicinals_pull_inat.py")

    # 3) iNaturalist plants
    inat_plants = INAT_DATASET / "plants"
    if inat_plants.is_dir():
        imgs = collect_images(inat_plants)
        if imgs:
            ROUTER_CLASSES["plant"].append(("inat_plants", imgs))

    # ── Other (OOD gate): non-target nature images ────────────────────────
    # Sourced from inat_dataset/other/ subdirectories (animals, insects,
    # lichens, birds, landscapes, etc.)
    if OTHER_DIR.is_dir():
        for sub_dir in sorted(OTHER_DIR.iterdir()):
            if sub_dir.is_dir():
                imgs = collect_images(sub_dir)
                if imgs:
                    ROUTER_CLASSES["other"].append((f"other_{sub_dir.name}", imgs))

        # Also grab any images directly in the other/ dir (flat layout)
        direct_imgs = collect_images(OTHER_DIR)
        if direct_imgs:
            ROUTER_CLASSES["other"].append(("other_direct", direct_imgs))
    else:
        print(f"\n  WARNING: {OTHER_DIR} not found.")
        print(f"  The 'other' class will be empty. Run backfill_inat_images.py first:")
        print(f"    python training/scripts/backfill_inat_images.py \\")
        print(f"        --taxon-id 47169 --output inat_dataset/other/animals --target 5000")
        print(f"    python training/scripts/backfill_inat_images.py \\")
        print(f"        --taxon-id 47158 --output inat_dataset/other/insects --target 5000")


def sample_evenly(sources: list[tuple[str, list[Path]]], target: int) -> list[Path]:
    """Sample target images evenly across source groups."""
    if not sources:
        return []

    per_source = max(1, target // len(sources))
    sampled = []

    for name, imgs in sources:
        n = min(per_source, len(imgs))
        sampled.extend(random.sample(imgs, n))

    random.shuffle(sampled)

    # If we overshot, trim; if under, that's fine
    return sampled[:target]


def copy_images(images: list[Path], dest_dir: Path):
    """Copy images into dest_dir, renaming to avoid collisions."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    for i, src in enumerate(images):
        ext = src.suffix.lower()
        dst = dest_dir / f"{i:05d}{ext}"
        shutil.copy2(src, dst)


def main():
    print("=" * 60)
    print("Building domain router dataset")
    print("=" * 60)

    # Verify source datasets exist
    for name, path in [
        ("berry_dataset", BERRY_DATASET),
        ("high_value_dataset", HIGHVALUE_DATASET),
        ("psychedelics_dataset", PSYCHEDELICS_DATASET),
    ]:
        if not path.is_dir():
            print(f"ERROR: {name} not found at {path}")
            return

    gather_sources()

    # Report source counts
    for cls, sources in ROUTER_CLASSES.items():
        total = sum(len(imgs) for _, imgs in sources)
        print(f"\n  {cls}: {len(sources)} source groups, {total} total images available")
        for name, imgs in sources:
            print(f"    {name}: {len(imgs)}")

    # Clean output directory
    if OUTPUT_DIR.exists():
        print(f"\nRemoving existing {OUTPUT_DIR} ...")
        shutil.rmtree(OUTPUT_DIR)

    # Sample and split
    print(f"\nSampling {TARGET_PER_CLASS} images per class ...")
    for cls, sources in ROUTER_CLASSES.items():
        images = sample_evenly(sources, TARGET_PER_CLASS)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_imgs = images[:split_idx]
        val_imgs   = images[split_idx:]

        train_dir = OUTPUT_DIR / "train" / cls
        val_dir   = OUTPUT_DIR / "val" / cls

        copy_images(train_imgs, train_dir)
        copy_images(val_imgs, val_dir)

        print(f"  {cls}: {len(train_imgs)} train + {len(val_imgs)} val = {len(images)} total")

    print(f"\nDataset written to: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
