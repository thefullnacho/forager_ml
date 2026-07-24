"""Replay a validation set through the ONNX pipeline and record every event.

Usage:
    python -m observability.batch_runner \
        --val-dir berry_dataset_split/val \
        --models-dir inference/onnx_staging \
        --model-version dev-$(git rev-parse --short HEAD) \
        --max-per-class 20

Expects an ImageFolder layout: <val-dir>/<class_name>/<image>.jpg, where
<class_name> is the ground-truth label. No location data is read or stored.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .db import connect, migrate
from .onnx_pipeline import OnnxPipeline
from .writer import EventWriter

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def iter_images(val_dir: Path, max_per_class: int) -> list[tuple[str, str]]:
    """Return (image_path, ground_truth_label) pairs from an ImageFolder tree."""
    pairs: list[tuple[str, str]] = []
    for class_dir in sorted(p for p in val_dir.iterdir() if p.is_dir()):
        imgs = sorted(p for p in class_dir.iterdir() if p.suffix.lower() in _IMG_EXTS)
        if max_per_class:
            imgs = imgs[:max_per_class]
        pairs.extend((str(p), class_dir.name) for p in imgs)
    return pairs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Replay a val set into the observability DB.")
    ap.add_argument("--val-dir", required=True)
    ap.add_argument("--models-dir", default="inference/onnx_staging", help="dir with *_logits.onnx")
    ap.add_argument("--meta-dir", default="inference/models", help="dir with *_classes.json / *_energy.json")
    ap.add_argument("--model-version", required=True)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = all")
    ap.add_argument("--dsn", default=None)
    args = ap.parse_args(argv)

    val_dir = Path(args.val_dir)
    pairs = iter_images(val_dir, args.max_per_class)
    if not pairs:
        print(f"No images under {val_dir}", file=sys.stderr)
        return 1

    pipeline = OnnxPipeline(args.models_dir, args.model_version, meta_dir=args.meta_dir)

    with connect(args.dsn) as conn:
        migrate(conn)
        writer = EventWriter(conn)
        n_toxic = 0
        for i, (path, truth) in enumerate(pairs, 1):
            outcome = pipeline.run(path, ground_truth=truth)
            writer.write(outcome)
            n_toxic += int(outcome.toxic_as_edible)
            if i % 25 == 0:
                print(f"  {i}/{len(pairs)}")
        writer.commit()

    print(f"Wrote {len(pairs)} runs (run_id={writer.run_id}). "
          f"toxic-as-edible: {n_toxic}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
