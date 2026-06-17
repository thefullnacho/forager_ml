# Forager ML

Real-time plant and fungi identification for edge deployment on a **Hailo 8L NPU** (Raspberry Pi 5). A domain router plus four expert classifiers — all `tf_efficientnet_lite2` (~4.9M params each) — run a two-stage pipeline: the router picks the domain, the relevant expert(s) classify, and results are resolved **deadly-vetoes-safe** before being pushed to an eInk display with optional voice trigger and TTS. All models output raw logits so softmax, energy-based OOD rejection, and confidence gating run on the Pi 5 CPU.

> **Architecture note:** earlier iterations used YOLOv8n-cls classifiers and a max-confidence voting layer. Both are gone. The shipped stack is EfficientNet-Lite2 with a router and deadly-vetoes-safe resolution (see [Convergence Logic](#convergence-logic)). Some `runs/classify/` artifacts and `convert_yolo_to_hef.py` remain from the YOLO era and are legacy.

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

> **Blackwell (RTX 5080) + PyTorch:** Works perfectly for EfficientNet training via `forager_stable` env.
>
> **Blackwell + TF 2.18 (Hailo DFC):** Does NOT work — TF 2.18 only supports up to `sm_90`. DFC compilation must run on the **RTX 4060 Ti** (`sm_89`) via `CUDA_VISIBLE_DEVICES=1` to get optimization level 2. See [Known Issues](#known-issues--gotchas) for the CUDA-library path gotcha.

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

A domain router and four expert classifiers, all identical architecture (`tf_efficientnet_lite2`, 224×224 input, raw-logit output):

| Model | Domain(s) | Classes | Val Top-1 | Toxic-as-edible FAR |
|-------|-----------|---------|-----------|---------------------|
| `domain_router` | berry / mushroom / plant / other | 4 | — | — |
| `berry_expert` | berry | 11 | 92.1% | 0.0 |
| `highvalue_expert` | mushroom, plant | 11 | 97.4% | 0.0 |
| `medicinals_expert` | plant | 21 | 95.7% | 0.0 |
| `psychedelics_expert` | mushroom | 14 | 91.3% | 0.0 |

Accuracies are `overall_accuracy` from `runs/efficientnet/<name>/benchmark.json`. **Toxic-as-edible FAR** is the false-accept rate of a toxic/deadly class being predicted as an edible — 0.0 across all experts on the validation sets.

**Architecture:** `tf_efficientnet_lite2` (timm) · ~4.9M params each · 224×224 · 50 epochs · RandAugment · AMP · raw logits (no softmax/sigmoid head) for energy-based OOD.

> **Class manifests are authoritative.** `inference/models/<name>_classes.json` (and the matching checkpoint) define the deployed class set; `SPECIES_METADATA` in `convergence.py` must be a superset. `highvalue_expert` ships **11 classes** with `reishi_northeast` (*Ganoderma tsugae*); `convergence.py` also carries an unused `reishi_mushroom` (*Ganoderma lucidum*) entry for forward-compat. After any recompile, regenerate the manifest from the checkpoint so class order/index stays aligned.

### Router Classes (4)
`berry` · `mushroom` · `plant` · `other`

### Berry Expert Classes (11)
`bittersweet_nightshade_toxic` · `blackberry_common` · `blueberry_highbush` · `blueberry_wild` · `canada_moonseed_deadly` · `elderberry_american` · `poison_ivy` · `pokeweed_toxic` · `staghorn_sumac` · `virginia_creeper_toxic` · `wild_grape_riverbank`

### High-Value Expert Classes (11)
`chaga_medicinal` · `chanterelles_edible` · `chicken_of_the_woods` · `ginseng_american` · `high_value_toxics` · `lions_mane` · `morels_edible` · `ostrich_fern_fiddlehead` · `ramps_wild_leek` · `reishi_northeast` · `saffron_crocus`

### Medicinals Expert Classes (21)
`boneset` · `burdock` · `catnip` · `coltsfoot` · `echinacea` · `foxglove_toxic` · `goldenrod` · `motherwort` · `mullein` · `plantain_broadleaf` · `poison_hemlock_deadly` · `red_clover` · `st_johns_wort` · `stinging_nettle` · `valerian` · `water_hemlock_deadly` · `white_snakeroot_toxic` · `wild_bergamot` · `wild_carrot` · `wood_nettle` · `yarrow`

### Psychedelics Expert Classes (14)
`amanita_muscaria_toxic` · `amanita_phalloides_deadly` · `conocybe_filaris_deadly` · `galerina_marginata_toxic` · `gymnopilus_junonius` · `other_mushroom` · `panaeolus_cinctulus` · `panax_quinquefolius_ginseng_conservation` · `psilocybe_azurescens` · `psilocybe_caerulipes` · `psilocybe_cubensis` · `psilocybe_cyanescens` · `psilocybe_ovoideocystidiata` · `psilocybe_semilanceata`

---

## Directory Structure

```
forager_ml/
├── data/
│   └── acquisition/          # iNaturalist scrapers (per category)
│
├── training/
│   └── scripts/
│       ├── train_efficientnet_specialist.py  # Train an expert (timm, Blackwell-ready)
│       ├── train_domain_router.py            # Train the 4-class domain router
│       ├── export_efficientnet_onnx.py       # best.pt -> ONNX (raw logits)
│       ├── compile_efficientnet_hef.py       # ONNX -> HEF via Hailo DFC
│       ├── calibrate_energy_threshold.py     # Energy-based OOD thresholds
│       ├── benchmark_expert.py               # Per-expert accuracy + toxic-as-edible FAR
│       ├── benchmark_router.py               # Router accuracy (EfficientNet or legacy YOLO)
│       ├── benchmark_ood.py                  # OOD detector AUROC
│       ├── build_router_dataset.py           # Assemble the router dataset
│       └── rebuild_dataset_splits.py         # (Re)create train/val splits
│
├── inference/
│   ├── main.py                 # Raspberry Pi entry point
│   ├── convert_yolo_to_hef.py  # LEGACY (YOLO era) — superseded by compile_efficientnet_hef.py
│   ├── models/                 # Deployed .hef + _classes.json + _energy.json
│   ├── onnx_staging/           # Intermediate .onnx files (temp, safe to delete)
│   └── pipeline/
│       ├── loader.py           # HailoModelLoader — loads all HEFs into one VDevice
│       ├── runner.py           # AsyncRunner — router→expert(s); energy OOD gate
│       ├── convergence.py      # Deadly-vetoes-safe resolution + species metadata
│       ├── camera.py           # picamera2 capture → 224×224 numpy array
│       ├── display.py          # Waveshare 3.7" eInk renderer
│       └── voice.py            # Whisper trigger + pyttsx3 TTS
│
├── runs/efficientnet/         # Trained checkpoints (best.pt), benchmarks, calibration
│   ├── domain_router_v2/
│   ├── berry_expert/
│   ├── highvalue_expert/
│   ├── medicinals_expert/
│   └── psychedelics_expert/
├── runs/classify/             # LEGACY YOLO runs (archivable)
│
├── berry_dataset/  high_value_dataset/  medicinals_dataset/  psychedelics_dataset/
├── *_dataset_split/           # ImageFolder train/ + val/ splits used for training
└── router_dataset/            # 4-class router dataset (train/ + val/)
```

---

## Python Environments

Two environments are required — they cannot be merged due to dependency conflicts.

| Environment | Path | Purpose |
|-------------|------|---------|
| `forager_stable` | `/home/alex/miniconda3/envs/forager_stable/` | Training, ONNX export, energy calibration — timm, PyTorch, OpenCV |
| `hailo_build` | `~/Downloads/hailo_build/` | Hailo DFC compilation (`hailo_sdk_client`, TF 2.18) |

---

## Pipeline: Dev Machine

### 1. Train

```bash
PY=/home/alex/miniconda3/envs/forager_stable/bin/python

# An expert (~20-30 min on RTX 5080):
$PY training/scripts/train_efficientnet_specialist.py \
    --dataset psychedelics_dataset_split --name psychedelics_expert --epochs 50

# The domain router:
$PY training/scripts/train_domain_router.py --name domain_router

# Weights land at: runs/efficientnet/<name>/best.pt
```

### 2. Export ONNX (raw logits)

```bash
$PY training/scripts/export_efficientnet_onnx.py \
    --checkpoint runs/efficientnet/psychedelics_expert/best.pt --no-activation
# -> inference/onnx_staging/<name>_logits.onnx
```

`--no-activation` strips the head activation so the ONNX (and HEF) output raw logits — required for energy-based OOD.

### 3. Compile .onnx → .hef (Hailo DFC)

**Must run on the RTX 4060 Ti** (`sm_89`) so TF 2.18 gets a supported GPU and DFC reaches **optimization level 2**. The bundled CUDA-12 libs must be on `LD_LIBRARY_PATH` (the system `/usr/local/cuda` is CUDA 13, which TF 2.18 can't use):

```bash
SP=/home/alex/Downloads/hailo_build/lib/python3.10/site-packages
NVLIBS=$(ls -d $SP/nvidia/*/lib | tr '\n' ':')

LD_LIBRARY_PATH="$NVLIBS$LD_LIBRARY_PATH" CUDA_VISIBLE_DEVICES=1 \
    /home/alex/Downloads/hailo_build/bin/python \
    training/scripts/compile_efficientnet_hef.py \
        --onnx inference/onnx_staging/psychedelics_expert_logits.onnx \
        --name psychedelics_expert \
        --dataset psychedelics_dataset_split
```

Watch for `Using default optimization level of 2` in the DFC output. If you see `Reducing optimization level to 0 ... no available GPU`, TF can't see the 4060 Ti — check `LD_LIBRARY_PATH` and `CUDA_VISIBLE_DEVICES` (see [Known Issues](#known-issues--gotchas)).

> **Recompiling an existing model:** delete the old `inference/models/<name>.hef`, `.har`, `_optimized.har`, **and `_classes.json`** first. The compile script reuses an existing `_classes.json` if present, so a stale manifest will survive a recompile and silently keep the old class set.

Output in `inference/models/`: `<name>.hef`, `<name>_classes.json`.

**Calibration details:**
- ~1200 images sampled evenly across classes (DFC needs 1024+ for optimization level > 0)
- PIL-based loading (hailo_build has no cv2)
- Format: NHWC float32 [0, 255]; the HEF applies ImageNet normalization internally via a DFC model script (`normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])`)
- `ImageFile.LOAD_TRUNCATED_IMAGES = True` to skip corrupt dataset images

### 4. Calibrate energy thresholds

```bash
$PY training/scripts/calibrate_energy_threshold.py \
    --checkpoint runs/efficientnet/psychedelics_expert/best.pt \
    --dataset psychedelics_dataset_split
# -> writes runs/efficientnet/<name>/energy_calibration.json
#    and copies inference/models/<name>_energy.json (threshold_p95 + temperature)
```

Re-run this whenever a model's classes change — the threshold is class-set specific. The inference temperature is read from this JSON and must match the calibration temperature.

### 5. Copy to Raspberry Pi

```bash
scp inference/models/*.hef inference/models/*_classes.json inference/models/*_energy.json \
    pi@192.168.4.73:~/forager/models/
```

---

## Pipeline: Raspberry Pi

### Running Inference

```bash
ssh pi@192.168.4.73

# Full pipeline (display, mic, speaker wired up):
python inference/main.py

# Development / SSH testing flags:
python inference/main.py --no-voice --no-display --no-tts
```

### Pipeline Components

| File | Role |
|------|------|
| `pipeline/loader.py` | `HailoModelLoader` — loads all .hef into one VDevice (`ROUND_ROBIN`); reads `_energy.json` threshold + temperature |
| `pipeline/runner.py` | `AsyncRunner` — router → expert(s); energy OOD gate; returns ALL surviving predictions |
| `pipeline/convergence.py` | Deadly-vetoes-safe resolution, species metadata, `ForagerResult` |
| `pipeline/camera.py` | picamera2 → 672×672 RGB capture → resized to 224×224 |
| `pipeline/display.py` | Waveshare 3.7" epd3in7 4-gray renderer |
| `pipeline/voice.py` | Whisper tiny.en trigger words + pyttsx3/espeak TTS |

### HailoRT API Notes

- `InputVStreamParams.make(network_group, quantized=False, format_type=FormatType.FLOAT32)` — called at inference time inside `_infer_single()`, not pre-created in loader
- `OutputVStreamParams.make(...)` — same pattern
- Input tensor: `np.expand_dims(image.astype(np.float32), axis=0)` — shape `(1, 224, 224, 3)` NHWC, values [0, 255]
- The HEF applies ImageNet normalization internally (baked in at compile via the DFC model script) — feed raw [0, 255] pixels; do **not** normalize before sending

### Convergence Logic

```
Stage 1 — Router
  • energy OOD gate (reject if energy > threshold_p95) → "other"/abstain
  • confidence gate (reject if top < 0.74) → abstain
  • else pick domain: berry | mushroom | plant

Stage 2 — Expert(s) for that domain (runner.py DOMAIN_EXPERTS)
  • berry    → berry_expert
  • mushroom → highvalue_expert + psychedelics_expert
  • plant    → highvalue_expert + medicinals_expert
  • each expert: energy OOD gate before softmax; OOD-rejected experts drop out

Resolution (convergence.resolve) — DEADLY-VETOES-SAFE
  • a DEADLY verdict beats a non-deadly one even if less confident
  • within a safety tier, highest confidence wins
  • no surviving prediction → UNKNOWN
```

**Why deadly-vetoes-safe and not max-confidence:** with max-confidence voting, a confident "ramps" (SAFE, 0.90) out-votes a cautious "deadly lookalike" (DEADLY, 0.55) — the lily-of-the-valley failure mode. In this domain a false reassurance is the only error that truly matters, so any deadly verdict wins.

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
| DFC optimization forced to level 0 ("no available GPU") | TF 2.18 can't load CUDA libs: system CUDA is 13, TF 2.18 needs CUDA 12; and Blackwell (`sm_120`) is unsupported | Run on the 4060 Ti with `CUDA_VISIBLE_DEVICES=1` **and** prepend `hailo_build`'s bundled `site-packages/nvidia/*/lib` to `LD_LIBRARY_PATH` |
| Recompile keeps old class set | `compile_efficientnet_hef.py` reuses an existing `_classes.json` | Delete `inference/models/<name>_classes.json` (and `.hef`/`.har`) before recompiling |
| Expert returns UNKNOWN for a valid class | Class in manifest but missing from `SPECIES_METADATA` | `SPECIES_METADATA` must be a superset of every deployed manifest |
| Pi camera not detected | Camera Module 3 in wrong port (CAM1) | Move FPC cable to **CAM0** |
| `create_input_vstreams_params` AttributeError | HailoRT API version difference | Use `InputVStreamParams.make()` / `OutputVStreamParams.make()` at inference time |
| Calibration fails with `'images'` key error | Old API expected dict input | Pass bare numpy array (NHWC) to `runner.optimize()` |
| `optimization_level` kwarg rejected | Not in SDK 3.33.0 signature | Call `runner.optimize(calib_data)` with no extra kwargs |
| Truncated image OSError during calibration | Corrupt files in dataset | `ImageFile.LOAD_TRUNCATED_IMAGES = True` + try/except skip |
| `optimize_full_precision()` fails | "Model requires quantized weights" | Use `runner.optimize()` not `optimize_full_precision()` |
| Voice/TTS imports crash on Pi if not installed | Top-level imports fail | Wrapped in `try/except` with `_WHISPER_AVAILABLE` / `_TTS_AVAILABLE` flags |

---

## Legacy Hardware & Code (Archived)

The project originally targeted a **Google Coral TPU** (driver rebuilds — abandoned) then a **Sony IMX500** (quantisation pipeline built but not deployed), and used **YOLOv8n-cls** classifiers before EfficientNet-Lite2. Superseded artifacts: `runs/classify/`, `inference/convert_yolo_to_hef.py`, the YOLO fallback in `benchmark_router.py`, and any root-level `.tflite`/`.keras`/`.h5`/Sony `.har` files — all archivable/deletable.

---

## Datasets

All images sourced from **iNaturalist** via scrapers in `data/acquisition/` with `quality_grade=research` filtering. Counts below are the train+val splits actually used for training.

| Dataset | Images (split) | Classes |
|---------|----------------|---------|
| `berry_dataset_split` | 39,009 | 11 |
| `high_value_dataset_split` | 33,524 | 11 |
| `psychedelics_dataset_split` | 34,631 | 14 |
| `medicinals_dataset_split` | 76,000 | 21 |
| `router_dataset` | 57,417 | 4 |

Training images are not redistributed. Acquisition scripts use verified taxon IDs.

---

*Part of the [Homesteader Labs](https://github.com/thefullnacho) ecosystem.*
