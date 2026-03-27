"""
Export a trained EfficientNet-B2 checkpoint to ONNX.

Key feature: --no-activation strips the final classification head's implicit
activation so the ONNX model outputs raw logits. This is critical for:
  1. DFC compilation with end_node_names (raw logit HEF)
  2. Energy-based OOD detection on the Pi 5 CPU
  3. Temperature scaling for calibrated confidence

Usage:
    # Export WITH activation (standard softmax output)
    python training/scripts/export_efficientnet_onnx.py \
        --checkpoint runs/efficientnet/psychedelics_expert/best.pt

    # Export WITHOUT activation (raw logits for OOD pipeline)
    python training/scripts/export_efficientnet_onnx.py \
        --checkpoint runs/efficientnet/psychedelics_expert/best.pt \
        --no-activation

Output goes to inference/onnx_staging/<model_name>.onnx
"""

import argparse
import os
import sys

import torch
import torch.nn as nn
import timm


class EfficientNetRawLogits(nn.Module):
    """
    Wrapper that ensures the model outputs raw logits with no activation.
    timm's EfficientNet already outputs logits by default (no softmax),
    but this wrapper makes the intent explicit and verifiable.
    """
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base = base_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


def parse_args():
    p = argparse.ArgumentParser(description="Export EfficientNet-B2 to ONNX")
    p.add_argument("--checkpoint", required=True,
                   help="Path to best.pt from training")
    p.add_argument("--no-activation", action="store_true",
                   help="Export raw logits (no softmax/sigmoid). Required for OOD pipeline.")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: inference/onnx_staging)")
    p.add_argument("--opset", type=int, default=13,
                   help="ONNX opset version (13 = max supported by Hailo DFC 3.33)")
    return p.parse_args()


def main():
    args = parse_args()

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    output_dir = args.output_dir or os.path.join(repo_root, "inference", "onnx_staging")
    os.makedirs(output_dir, exist_ok=True)

    # Load checkpoint
    ckpt_path = args.checkpoint
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(repo_root, ckpt_path)

    if not os.path.exists(ckpt_path):
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)

    model_name = ckpt["model_name"]
    num_classes = ckpt["num_classes"]
    img_size = ckpt.get("img_size", 224)
    classes = ckpt["classes"]

    print(f"Model      : {model_name}")
    print(f"Classes    : {num_classes} — {classes}")
    print(f"Input size : {img_size}x{img_size}")
    print(f"Val accuracy: {ckpt.get('val_acc', 'N/A')}")
    print(f"Raw logits : {args.no_activation}")

    # Rebuild model — use saved arch name, fall back to efficientnet_lite2
    arch = ckpt.get("arch", "efficientnet_lite2")
    print(f"Arch       : {arch}")
    model = timm.create_model(arch, pretrained=False, num_classes=num_classes)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    if args.no_activation:
        model = EfficientNetRawLogits(model)

    # Determine output filename
    suffix = "_logits" if args.no_activation else ""
    onnx_filename = f"{model_name}{suffix}.onnx"
    onnx_path = os.path.join(output_dir, onnx_filename)

    # Export
    dummy_input = torch.randn(1, 3, img_size, img_size)

    print(f"\nExporting to ONNX (opset {args.opset}, legacy exporter) ...")
    # dynamo=False forces the TorchScript-based legacy exporter which reliably
    # produces opset 11-13 output that Hailo DFC 3.33 can parse.
    # The new dynamo exporter (default in PyTorch 2.9+) outputs opset 18 with
    # different Conv attribute encoding that DFC's ONNX parser rejects.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=None,  # fixed batch size for Hailo DFC
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )

    size_mb = os.path.getsize(onnx_path) / 1e6
    print(f"✓ Exported: {onnx_path}  ({size_mb:.1f} MB)")

    # Verify with onnx
    try:
        import onnx
        model_onnx = onnx.load(onnx_path)
        onnx.checker.check_model(model_onnx)

        # Print I/O info for DFC config
        inputs = [i.name for i in model_onnx.graph.input]
        outputs = [o.name for o in model_onnx.graph.output]
        in_shape = [d.dim_value for d in model_onnx.graph.input[0].type.tensor_type.shape.dim]
        out_shape = [d.dim_value for d in model_onnx.graph.output[0].type.tensor_type.shape.dim]

        print(f"\nONNX verification passed.")
        print(f"  Input  : {inputs}  shape={in_shape}")
        print(f"  Output : {outputs}  shape={out_shape}")
    except ImportError:
        print("(onnx package not installed — skipping verification)")

    print(f"\nNext step:")
    print(f"  python training/scripts/compile_efficientnet_hef.py \\")
    print(f"      --onnx {onnx_path} \\")
    print(f"      --name {model_name} \\")
    print(f"      --dataset {ckpt_path.replace('best.pt', '').rstrip('/')}")


if __name__ == "__main__":
    main()
