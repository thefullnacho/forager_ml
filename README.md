# Forager ML

Real-time plant and fungi identification pipeline for edge deployment on a **Hailo 8L NPU** (Raspberry Pi 5). Three specialist YOLO classifiers run in parallel, results converge through a voting layer, and the output is pushed to an eInk display with optional voice trigger and TTS.

---

## Hardware

### Dev Machine

| Component | Spec |
|-----------|------|
| Primary GPU | NVIDIA RTX 5080 (Blackwell, `sm_120`), 16 GB VRAM |
| Secondary GPU | NVIDIA RTX 4060 Ti 16 GB (Ada Lovelace, `sm_89`) — used for Hailo DFC compilation |
| CPU | AMD Ryzen 9 9950X3D |
| PSU | 1000W Gold (handles both GPUs comfortably ~800W combined load) |
| PyTorch stack | PyTorch 2.9.1+cu128 · CUDA 12.8 · cuDNN 91002 |

> **Blackwell (RTX 5080) + PyTorch:** Works perfectly for YOLO training via `forager_stable` env.
>
> **Blackwell + TF 2.18 (Hailo DFC):** Does NOT work — TF 2.18 only supports up to `sm_90`. The DFC falls back to optimization level 0 with a CPU-only TF session, capping calibration at 64 images regardless of how many you provide. The **RTX 4060 Ti** (`sm_89`) is used specifically for DFC compilation to get optimization level 2.

### Edge Device

| Component | Spec |
|-----------|------|
| Board | Raspberry Pi 5 |
| NPU | Hailo 8L M.2 HAT |
| Camera | Raspberry Pi Camera Module 3 (IMX708) — must use **CAM0 port** on Pi 5 |
| Display | Waveshare 3.7" eInk HAT · 480×280 · 4-gray mode (epd3in7 driver) |
| Audio | USB microphone + speaker (OpenAI Whisper tiny.en + pyttsx3/espeak TTS) |
| Pi hostname | `forager-dev` · `192.168.4.73` |

> **Pi 5 camera note:** Camera Module 3 must be connected to **CAM0** (the port closer to the USB-C power jack). CAM1 will not be detected.

---

## Model Architecture

Three expert classifiers, all identical architecture:

| Model | Dataset | Classes | Top-1 Acc | Top-5 Acc |
|-------|---------|---------|-----------|-----------|
| `berry_expert` | `berry_dataset` (38,672 images) | 11 | 94.2% | 99.6% |
| `highvalue_expert` | `high_value_dataset` (36,688 images) | 12 | 96.6% | 99.8% |
| `psychedelics_expert` | `psychedelics_dataset` (~33,000 images) | 12 | 80.91% | 99.59% |

**Architecture:** YOLOv8n-cls · 224×224 input · 50 epochs · batch 64 · RandAugment · AMP

> **highvalue_expert has 12 classes, not 11.** It includes both `reishi_mushroom` (Ganoderma lucidum) and `reishi_northeast` (Ganoderma tsugae) as separate classes. This caused a class manifest mismatch on first deployment — `reishi_mushroom` was missing from the JSON. The class manifest and SPECIES_METADATA in `convergence.py` must both list 12 classes.

### Berry Expert Classes
`bittersweet_nightshade_toxic` · `blackberry_common` · `blueberry_highbush` · `blueberry_wild` · `canada_moonseed_deadly` · `elderberry_american` · `poison_ivy` · `pokeweed_toxic` · `staghorn_sumac` · `virginia_creeper_toxic` · `wild_grape_riverbank`

### High-Value Expert Classes (12)
`chaga_medicinal` · `chanterelles_edible` · `chicken_of_the_woods` · `ginseng_american` · `high_value_toxics` · `lions_mane` · `morels_edible` · `ostrich_fern_fiddlehead` · `ramps_wild_leek` · `reishi_mushroom` · `reishi_northeast` · `saffron_crocus`

### Psychedelics Expert Classes
`amanita_muscaria_toxic` · `amanita_phalloides_deadly` · `conocybe_filaris_deadly` · `galerina_marginata_toxic` · `gymnopilus_junonius` · `other_mushroom` · `panax_quinquefolius_ginseng_conservation` · `psilocybe_azurescens` · `psilocybe_caerulipes` · `psilocybe_cubensis` · `psilocybe_cyanescens` · `psilocybe_semilanceata`

---

## Directory Structure

```
forager_ml/
├── data/
│   ├── acquisition/          # iNaturalist scrapers (per category)
│   └── calibration/          # Calibration set builders for Hailo quantisation
│
├── training/
│   ├── configs/              # YAML/JSON training and hardware configs
│   └── scripts/
│       ├── train_psychedelics_expert.py   # GPU training script (Blackwell-ready)
│       └── train_forager_lite*.py         # Legacy EfficientNetB0 trainers (v4-v7)
│
├── inference/
│   ├── convert_yolo_to_hef.py  # .pt -> .onnx -> .hef full conversion pipeline
│   ├── main.py                 # Raspberry Pi entry point
│   ├── models/                 # Compiled .hef files + _classes.json manifests
│   ├── onnx_staging/           # Intermediate .onnx files (temp, safe to delete)
│   └── pipeline/
│       ├── loader.py           # HailoModelLoader — loads all HEFs into one VDevice
│       ├── runner.py           # AsyncRunner — parallel inference via ThreadPoolExecutor
│       ├── convergence.py      # Voting layer, species metadata, ForagerResult
│       ├── camera.py           # picamera2 capture → 224×224 numpy array
│       ├── display.py          # Waveshare 3.7" eInk renderer
│       └── voice.py            # Whisper trigger + pyttsx3 TTS
│
├── optimization/
│   ├── hailo/                # Hailo DFC compilation scripts (legacy EfficientNet)
│   ├── quantization/         # TFLite / ONNX / SavedModel export
│   └── sony_imx500/          # Sony IMX500 pipeline (deprecated)
│
├── utils/
│   ├── model_ops/            # Model surgery: strip layers, fix names, flatten
│   └── visualization/        # Confusion matrix generation
│
├── runs/classify/
│   ├── berry_expert/         # Trained YOLO weights + training plots
│   ├── highvalue_expert/     # Trained YOLO weights + training plots
│   └── psychedelics_expert/  # Trained YOLO weights + training plots
│
├── berry_dataset/            # 38,672 images · 11 classes
├── high_value_dataset/       # 36,688 images · 12 classes
└── psychedelics_dataset/     # ~33,000 images · 12 classes
```

---

## Python Environments

Two environments are required — they cannot be merged due to dependency conflicts.

| Environment | Path | Purpose |
|-------------|------|---------|
| `forager_stable` | `/home/alex/miniconda3/envs/forager_stable/` | Training — ultralytics, PyTorch, OpenCV |
| `hailo_build` | `~/Downloads/hailo_build/` | Hailo DFC compilation (`hailo_sdk_client`, TF 2.18) |

> The conversion script (`convert_yolo_to_hef.py`) is always run under `hailo_build`. It automatically calls `forager_stable`'s Python via subprocess for the ONNX export step — no manual environment switching needed.

---

## Pipeline: Dev Machine

### 1. Train the Expert Models

```bash
# berry_expert and highvalue_expert are already trained.
# Train psychedelics expert (~20-30 min on RTX 5080):
/home/alex/miniconda3/envs/forager_stable/bin/python \
    training/scripts/train_psychedelics_expert.py

# Weights land at:
#   runs/classify/psychedelics_expert/weights/best.pt
```

> **RTX 5080 smoke test:** PyTorch 2.9.1+cu128 supports Blackwell natively. If a smoke test fails, check the matrix multiply shape — `x = torch.randn(64, 512); x @ x.T` not `x @ x.view(64,-1).T`.

### 2. Convert .pt → .hef

**Requires the RTX 4060 Ti to be installed** so TF 2.18 inside `hailo_build` gets a compatible GPU (sm_89 ≤ sm_90 limit). Without a compatible GPU, DFC falls back to optimization level 0.

```bash
# With both GPUs installed, hide the RTX 5080 (Blackwell) from TF,
# expose only the 4060 Ti. Check GPU indices first:
nvidia-smi -L

# Then run with only the 4060 Ti visible (adjust index as needed):
CUDA_VISIBLE_DEVICES=1 /home/alex/Downloads/hailo_build/bin/python \
    inference/convert_yolo_to_hef.py
```

To force a full recompile (e.g. after fixing class manifests or changing calibration):
```bash
rm inference/models/*.hef inference/models/*.har
CUDA_VISIBLE_DEVICES=1 /home/alex/Downloads/hailo_build/bin/python \
    inference/convert_yolo_to_hef.py
```

Output in `inference/models/`:
```
berry_expert.hef              berry_expert_classes.json
highvalue_expert.hef          highvalue_expert_classes.json
psychedelics_expert.hef       psychedelics_expert_classes.json
```

Watch for `Optimization level: 2` in DFC output — if you see `Reducing optimization level to 0`, the wrong GPU is visible.

**Calibration details:**
- 1200 images sampled evenly across classes per model
- PIL-based loading (not OpenCV — hailo_build doesn't have cv2)
- Format: NHWC float32 [0, 255] — matches YOLO ONNX's expected input range
- `ImageFile.LOAD_TRUNCATED_IMAGES = True` to skip corrupt dataset images

### 3. Copy to Raspberry Pi

```bash
scp inference/models/*.hef inference/models/*_classes.json \
    pi@192.168.4.73:~/forager/models/
```

---

## Pipeline: Raspberry Pi

### Running Inference

```bash
# SSH in
ssh pi@192.168.4.73

# Full pipeline (requires display, mic, speaker wired up):
python inference/main.py

# Development / SSH testing flags:
python inference/main.py --no-voice --no-display --no-tts
```

### Pipeline Components

| File | Role |
|------|------|
| `pipeline/loader.py` | `HailoModelLoader` — loads all .hef files into one VDevice with `ROUND_ROBIN` scheduler |
| `pipeline/runner.py` | `AsyncRunner` — parallel inference via `ThreadPoolExecutor(max_workers=3)` (HRT_4 pattern) |
| `pipeline/convergence.py` | Voting layer, species metadata, `ForagerResult` |
| `pipeline/camera.py` | picamera2 → 672×672 RGB capture → resized to 224×224 |
| `pipeline/display.py` | Waveshare 3.7" epd3in7 4-gray renderer |
| `pipeline/voice.py` | Whisper tiny.en trigger words + pyttsx3/espeak TTS |

### HailoRT API Notes

The inference pipeline uses HailoRT Python API. Key details that differ from older docs:

- `InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)` — called at inference time inside `_infer_single()`, not pre-created in loader
- `OutputVStreamParams.make(...)` — same pattern
- Input tensor: `np.expand_dims(image.astype(np.float32), axis=0)` — shape `(1, 224, 224, 3)` NHWC, values [0, 255]
- The YOLO ONNX model includes `/255` normalization internally; the HEF inherits this — do **not** normalize before sending

### Convergence Logic

```
For each model's top prediction:
  1. Domain gate  — species must be in that model's trained class list
  2. Confidence gate — discard if confidence < 0.75
  3. Vote aggregation — surviving votes collected per candidate species
  4. Agreement boost — multiply by 1.20 if 2+ models agree (cap at 0.99)
  5. Abstention — return UNKNOWN if no species survives all gates
```

### Quantization / Confidence Notes

Int8 quantization at optimization level 0 (no GPU) flattens the softmax output, making all class probabilities nearly uniform (~8-9% for 12 classes). At optimization level 2 (with 4060 Ti), proper confidence separation is expected.

As a workaround for level-0 HEFs, `runner.py` applies probability sharpening:
```python
def _sharpen(probs, alpha=4.0):
    sharpened = np.power(probs, alpha)
    return sharpened / sharpened.sum()
```
This can be removed or tuned once level-2 HEFs are compiled.

### eInk Output Format

```
SPECIES NAME
Scientific name
Confidence: XX%
─────────────────
⚠ Lookalike: [name]
  Key diff: [detail]
─────────────────
SAFE / CAUTION / DEADLY / UNKNOWN
```

### Voice Trigger Words

`scan` · `capture` · `identify` · `go` · `forager`

---

## Known Issues & Gotchas

| Issue | Root Cause | Fix |
|-------|-----------|-----|
| DFC optimization level forced to 0 | TF 2.18 does not support Blackwell (`sm_120`) | Install RTX 4060 Ti; use `CUDA_VISIBLE_DEVICES` to hide 5080 |
| highvalue_expert class count mismatch (12 vs 11) | `reishi_mushroom` missing from manifest | Added to `_classes.json` and `SPECIES_METADATA` |
| Pi camera not detected | Camera Module 3 in wrong port (CAM1) | Move FPC cable to **CAM0** |
| `create_input_vstreams_params` AttributeError | HailoRT API version difference | Use `InputVStreamParams.make()` / `OutputVStreamParams.make()` at inference time |
| Calibration fails with `'images'` key error | Old API expected dict input | Pass bare numpy array (NHWC) to `runner.optimize()` |
| `optimization_level` kwarg rejected | Not in SDK 3.33.0 signature | Call `runner.optimize(calib_data)` with no extra kwargs |
| Truncated image OSError during calibration | Corrupt files in dataset | `ImageFile.LOAD_TRUNCATED_IMAGES = True` + try/except skip |
| `optimize_full_precision()` fails | "Model requires quantized weights" | Use `runner.optimize()` not `optimize_full_precision()` |
| Voice/TTS imports crash on Pi if not installed | Top-level imports fail | Wrapped in `try/except` with `_WHISPER_AVAILABLE` / `_TTS_AVAILABLE` flags |

---

## Legacy Hardware (Archived)

The project originally targeted a **Google Coral TPU** (required driver rebuilds — abandoned) and then a **Sony IMX500** (full quantisation pipeline built but not deployed). Both are superseded by the Hailo 8L.

Root-level `.tflite`, `.keras`, `.h5`, and Sony `.har` files are artifacts from these iterations and can be archived or deleted.

---

## Datasets

All images sourced from **iNaturalist** via scrapers in `data/acquisition/` with `quality_grade` filtering.

| Dataset | Images | Classes | Scraper |
|---------|--------|---------|---------|
| `berry_dataset` | 38,672 | 11 | `berry_pull_inat.py` |
| `high_value_dataset` | 36,688 | 12 | `high_value_pull_inat.py` |
| `psychedelics_dataset` | ~33,000 | 12 | `psychedelic_pull_inat.py` |

---

*Part of the [Homesteader Labs](https://github.com/thefullnacho) ecosystem.*
