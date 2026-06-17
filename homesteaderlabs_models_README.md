# HomesteaderLabs — Open Model Releases

Edge AI models for field identification of wild plants, mushrooms, and forageables. Trained on iNaturalist research-grade observations and compiled for the Hailo 8L AI accelerator. Designed for the Forager handheld device and available for independent research and deployment.

---

## System Architecture

The Forager inference stack uses a two-stage pipeline:

```
Camera frame (224×224 RGB)
        │
        ▼
┌─────────────────────┐
│   Domain Router     │  EfficientNet Lite2 — 4 classes
│   (mushroom /       │  Confidence threshold: 0.74
│    berry / plant /  │  OOD gate: energy score
│    other)           │
└─────────────────────┘
        │
        ├── mushroom ──► [ Mycologist Expert ] ─┐
        │                [ High-Value Expert  ] ─┴─ deadly-vetoes-safe
        │
        ├── berry ─────► [ Berry Expert ]
        │
        ├── plant ─────► [ High-Value Expert  ] ─┐
        │                [ Medicinals Expert  ] ─┴─ deadly-vetoes-safe
        │
        └── other ─────► abstain
```

Each expert is an independent `tf_efficientnet_lite2` classifier (timm) trained on a domain-specific iNaturalist dataset and compiled to HEF via the Hailo Dataflow Compiler. All models output raw logits — post-processing (softmax, energy-based OOD rejection, confidence thresholding) runs on the Pi 5 CPU.

When a domain is served by two experts, the result is resolved **deadly-vetoes-safe**: a DEADLY verdict from either expert beats a non-deadly one even if the non-deadly verdict is more confident. Max-confidence is used only to break ties within the same safety tier. This prevents a confident edible call (e.g. "ramps") from overriding a cautious deadly call (e.g. a toxic lookalike) — the lily-of-the-valley failure mode.

---

## Hardware Target

| Component | Spec |
|---|---|
| SBC | Raspberry Pi 5 (4GB) |
| AI accelerator | Hailo 8L M.2 AI HAT (4 TOPS, int8) |
| Camera | Raspberry Pi Camera Module 3 (IMX708, CAM0) |
| Display | Waveshare 3.7" eInk (480×280, 4-gray) |
| Inference runtime | HailoRT 4.x |

Models can also run on CPU via ONNX Runtime using the provided `.onnx` files.

---

## Available Models

| Model | Domain | Classes | Status | Model Card |
|---|---|---|---|---|
| **mycologist** | Psilocybin mushrooms + deadly lookalikes | 14 | Released | [MODEL_CARD.md](mycologist/MODEL_CARD.md) |
| berry_expert | Wild berries + toxic lookalikes | 11 | In progress | — |
| highvalue_expert | Chanterelles, morels, lion's mane, ginseng, ramps | 11 | In progress | — |
| medicinals_expert | Wild medicinal plants + toxic lookalikes | 21 | In progress | — |
| domain_router | Domain classification (mushroom/berry/plant/other) | 4 | In progress | — |

---

## Using the Models

### HEF (Hailo 8L, recommended)

```python
import hailo_platform as hp

hef = hp.HEF("mycologist/psychedelics_expert.hef")
# See HailoRT documentation for VDevice setup and inference loop
```

### ONNX (CPU inference)

```python
import onnxruntime as ort
import numpy as np

sess = ort.InferenceSession("mycologist/psychedelics_expert_logits.onnx")
# Input: float32 [1, 3, 224, 224], ImageNet-normalized
# Output: raw logits [1, 14] — apply softmax for probabilities
logits = sess.run(["output"], {"input": img_tensor})[0]
probs = np.exp(logits) / np.exp(logits).sum()
```

### Recompiling the HEF (Hailo DFC)

Each model release includes a `*_hailo8l.yaml` documenting all DFC compilation parameters (input shape, normalization, calibration settings, opset). See the model directory for the compilation reference and `training/scripts/compile_efficientnet_hef.py` in the [forager_ml repo](https://github.com/TheFullNacho/forager_ml) for the full compilation pipeline.

### Reproducing the Dataset

Acquisition scripts for all iNaturalist datasets are in `data/acquisition/` in the forager_ml repo. Each script uses verified taxon IDs and `quality_grade=research` filtering. Training images are not redistributed.

---

## Safety Notice

**These models are for informational and research purposes only.**

Wild mushroom and plant identification carries fatal risk. These models do not replace expert mycological or botanical identification. No model output should be acted on — including consumption decisions — without independent verification by a qualified expert. Amatoxin poisoning (Amanita phalloides, Galerina marginata, Conocybe filaris) is lethal and has no reliable field antidote.

HomesteaderLabs and contributors to this repository accept no liability for decisions made based on model output.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

Training data sourced from iNaturalist observers under CC-BY and CC0 licenses. Model weights are an original derivative work and are released under Apache 2.0 independently of the training data license.

---

## Citation

If you use these models in research, please cite:

```
HomesteaderLabs (2026). Forager Field Identification Models.
https://github.com/homesteaderlabs/models
Training data: iNaturalist (https://www.inaturalist.org)
```

[HomesteaderLabs.com](https://homesteaderlabs.com)
