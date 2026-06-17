"""
Compile EfficientNet-Lite2 ONNX → HEF via Hailo DFC.

This script replaces the YOLO-specific convert_yolo_to_hef.py for the
EfficientNet-Lite2 architecture (SE-free, hailo8l-compatible). It handles:

  1. ONNX → HAR translation (Hailo Archive)
  2. int8 calibration using training images
  3. HAR → HEF compilation targeting hailo8l
  4. Class manifest generation

The ONNX model should already have the activation stripped (exported with
--no-activation) so the HEF outputs raw logits for energy scoring.

Run with the hailo_build interpreter (DFC venv):
    /home/alex/Downloads/hailo_build/bin/python \
        training/scripts/compile_efficientnet_hef.py \
        --onnx inference/onnx_staging/psychedelics_expert_logits.onnx \
        --name psychedelics_expert \
        --dataset psychedelics_dataset
"""

import argparse
import json
import os
import random
import sys

import numpy as np


# ── Config ───────────────────────────────────────────────────────────────────

HW_ARCH = "hailo8l"
CALIB_SAMPLES = 1200   # DFC needs 1024+ for optimization level > 0
IMG_SIZE = 224


def parse_args():
    p = argparse.ArgumentParser(description="Compile EfficientNet-Lite2 ONNX to HEF")
    p.add_argument("--onnx", required=True, help="Path to ONNX model")
    p.add_argument("--name", required=True, help="Model name (e.g. psychedelics_expert)")
    p.add_argument("--dataset", required=True,
                   help="Path to dataset dir for calibration images")
    p.add_argument("--output-dir", default=None,
                   help="Output dir for .hef and manifest (default: inference/models)")
    p.add_argument("--calib-samples", type=int, default=CALIB_SAMPLES)
    p.add_argument("--hw-arch", default=HW_ARCH, choices=["hailo8l", "hailo8"])
    return p.parse_args()


# ── Calibration data ─────────────────────────────────────────────────────────

def build_calib_data(dataset_dir: str, n: int = CALIB_SAMPLES) -> np.ndarray:
    """
    Sample n images evenly across all classes in dataset_dir.

    Returns np.ndarray (N, H, W, C) float32 in [0, 255] — NHWC format.
    The Hailo DFC calibration API expects NHWC with pixel values, not normalized.
    DFC handles normalization internally based on the ONNX preprocessing nodes.
    """
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    # Find class directories — handle both flat (dataset/) and split (dataset/train/) layouts
    if os.path.isdir(os.path.join(dataset_dir, "train")):
        scan_dir = os.path.join(dataset_dir, "train")
    else:
        scan_dir = dataset_dir

    class_dirs = sorted([
        os.path.join(scan_dir, d)
        for d in os.listdir(scan_dir)
        if os.path.isdir(os.path.join(scan_dir, d))
    ])

    if not class_dirs:
        print(f"ERROR: No class directories found in {scan_dir}")
        sys.exit(1)

    # Collect image paths, spread evenly
    all_paths = []
    per_class = max(1, n // len(class_dirs))
    for cls_dir in class_dirs:
        imgs = [
            os.path.join(cls_dir, f)
            for f in os.listdir(cls_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))
        ]
        all_paths.extend(random.sample(imgs, min(per_class, len(imgs))))

    random.shuffle(all_paths)
    all_paths = all_paths[:n]

    print(f"  Calibration: {len(all_paths)} images from {len(class_dirs)} classes")

    imgs_out = []
    for path in all_paths:
        try:
            img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            img = np.array(img, dtype=np.float32)   # (H, W, C) in [0, 255]
            imgs_out.append(img)
        except Exception:
            continue

    return np.stack(imgs_out, axis=0)   # (N, H, W, C)


# ── ONNX inspection ─────────────────────────────────────────────────────────

def get_onnx_io(onnx_path: str):
    import onnx
    model = onnx.load(onnx_path)
    inputs = [i.name for i in model.graph.input]
    outputs = [o.name for o in model.graph.output]
    shape = [d.dim_value for d in model.graph.input[0].type.tensor_type.shape.dim]
    return inputs, outputs, shape


# ── Compilation ──────────────────────────────────────────────────────────────

def compile_hef(args):
    from hailo_sdk_client import ClientRunner

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    onnx_path = args.onnx
    if not os.path.isabs(onnx_path):
        onnx_path = os.path.join(repo_root, onnx_path)

    dataset_dir = args.dataset
    if not os.path.isabs(dataset_dir):
        dataset_dir = os.path.join(repo_root, dataset_dir)

    output_dir = args.output_dir or os.path.join(repo_root, "inference", "models")
    os.makedirs(output_dir, exist_ok=True)

    hef_path = os.path.join(output_dir, f"{args.name}.hef")
    har_path = os.path.join(output_dir, f"{args.name}.har")

    if not os.path.exists(onnx_path):
        print(f"ERROR: ONNX not found: {onnx_path}")
        sys.exit(1)

    # Inspect ONNX
    inputs, outputs, shape = get_onnx_io(onnx_path)
    print(f"  ONNX input  : {inputs}  shape={shape}")
    print(f"  ONNX output : {outputs}")

    net_input_shapes = {inputs[0]: shape if len(shape) == 4 else [1, 3, IMG_SIZE, IMG_SIZE]}

    # ── Step 1: Parse ONNX → HAR ─────────────────────────────────────────────
    print(f"\n  [1/4] Parsing ONNX → HAR ...")
    runner = ClientRunner(hw_arch=args.hw_arch)

    runner.translate_onnx_model(
        onnx_path,
        args.name,
        start_node_names=[inputs[0]],
        end_node_names=outputs,       # raw logit output node — no activation
        net_input_shapes=net_input_shapes,
    )
    runner.save_har(har_path)
    print(f"  HAR saved: {har_path}")

    # ── Step 1b: Model script — normalization ────────────────────────────────
    # Tells DFC that the Pi will feed raw uint8 images [0,255] and the HEF
    # will apply ImageNet normalization internally. Without this, the DFC
    # can't correctly compute quantization shifts for the EfficientNet avgpool
    # (shift delta > 2 error).
    # Values are the standard ImageNet mean/std scaled to [0,255]:
    #   mean = [0.485, 0.456, 0.406] * 255 = [123.675, 116.28, 103.53]
    #   std  = [0.229, 0.224, 0.225] * 255 = [58.395,  57.12,  57.375]
    import tempfile
    # Model script syntax: layer_name = normalization([mean], [std])
    # ImageNet mean/std scaled to [0,255] domain:
    #   mean = [0.485, 0.456, 0.406] * 255 = [123.675, 116.28, 103.53]
    #   std  = [0.229, 0.224, 0.225] * 255 = [58.395,  57.12,  57.375]
    model_script_content = (
        "norm_layer = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])\n"
    )
    with tempfile.NamedTemporaryFile(mode="w", suffix=".alls", delete=False) as f:
        f.write(model_script_content)
        model_script_path = f.name
    runner.load_model_script(model_script_path)
    os.unlink(model_script_path)
    print(f"  Model script loaded (ImageNet normalization)")

    # ── Step 2: Build calibration data ────────────────────────────────────────
    print(f"\n  [2/4] Building calibration set ...")
    calib_data = build_calib_data(dataset_dir, args.calib_samples)

    # ── Step 3: int8 quantization ─────────────────────────────────────────────
    print(f"\n  [3/4] Optimizing (int8 quantization) ...")
    runner.optimize(calib_data)

    opt_har_path = har_path.replace(".har", "_optimized.har")
    runner.save_har(opt_har_path)
    print(f"  Optimized HAR: {opt_har_path}")

    # ── Step 4: Compile → HEF ─────────────────────────────────────────────────
    print(f"\n  [4/4] Compiling → HEF ...")
    hef_bytes = runner.compile()

    if hef_bytes is None:
        # Workaround: reload optimized HAR and compile from fresh runner
        print(f"  [retry] Reloading optimized HAR for compilation ...")
        runner2 = ClientRunner(har=opt_har_path, hw_arch=args.hw_arch)
        hef_bytes = runner2.compile()

    with open(hef_path, "wb") as f:
        f.write(hef_bytes)

    size_mb = os.path.getsize(hef_path) / 1e6
    print(f"\n  ✓ {hef_path}  ({size_mb:.1f} MB)")
    return hef_path


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 60)
    print(f"Compile EfficientNet-B2 → HEF")
    print(f"  Model   : {args.name}")
    print(f"  ONNX    : {args.onnx}")
    print(f"  Dataset : {args.dataset}")
    print(f"  HW arch : {args.hw_arch}")
    print("=" * 60)

    hef_path = compile_hef(args)

    # Write class manifest — load from training checkpoint if available
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = args.output_dir or os.path.join(repo_root, "inference", "models")

    # Try to load classes from training checkpoint
    ckpt_dir = os.path.join(repo_root, "runs", "efficientnet", args.name)
    ckpt_path = os.path.join(ckpt_dir, "best.pt")
    manifest_path = os.path.join(output_dir, f"{args.name}_classes.json")

    def _classes_from_dataset():
        # ImageFolder sorts classes alphabetically, so sorted(dirs) matches the
        # checkpoint's class order. This is torch-free — important because this
        # script runs under hailo_build, which has no torch.
        dataset_dir = args.dataset
        if not os.path.isabs(dataset_dir):
            dataset_dir = os.path.join(repo_root, dataset_dir)
        scan_dir = os.path.join(dataset_dir, "train") if os.path.isdir(os.path.join(dataset_dir, "train")) else dataset_dir
        return sorted(d for d in os.listdir(scan_dir) if os.path.isdir(os.path.join(scan_dir, d)))

    if os.path.exists(manifest_path):
        classes = json.load(open(manifest_path))["classes"]
    else:
        # Prefer the dataset (torch-free). Fall back to the checkpoint only if
        # torch is importable in this env.
        classes = _classes_from_dataset()
        if not classes and os.path.exists(ckpt_path):
            try:
                import torch
                classes = torch.load(ckpt_path, map_location="cpu", weights_only=True)["classes"]
            except Exception as e:
                print(f"  WARNING: could not load classes from checkpoint: {e}")

    manifest = {"model": args.name, "classes": classes}
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"  ✓ Manifest: {manifest_path}")

    print("\n" + "=" * 60)
    print("Compilation complete.")
    print(f"\nNext step — calibrate energy thresholds:")
    print(f"  python training/scripts/calibrate_energy_threshold.py \\")
    print(f"      --checkpoint runs/efficientnet/{args.name}/best.pt \\")
    print(f"      --dataset {args.dataset}")
    print("=" * 60)


if __name__ == "__main__":
    main()
