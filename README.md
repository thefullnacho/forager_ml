# Forager ML

The complete Machine Learning pipeline for **Forager**: Dataset acquisition, model training, and edge optimization for Hailo NPU and Sony IMX500.

This repository contains the source code for building the "brain" of the Forager plant identification system. It handles everything from scraping training data to exporting optimized models for specialized hardware.

## 🚀 Key Features

- **Data Acquisition**: Custom scripts for pulling and filtering images from iNaturalist.
- **Model Training**: Multiple architectures including MobileNetV3 and YOLO-based classifiers.
- **Optimization**: Workflows for Sony IMX500 and Hailo-8 acceleration.
- **Evaluation**: Tools for generating confusion matrices and performance metrics.

## 🛠 Project Structure

The project is organized into functional modules:

- **`data/`**: Tools for gathering and preparing data.
  - `acquisition/`: Scrapers for iNaturalist (`pull_inat.py`, etc.).
  - `calibration/`: Generation of `.npy` calibration sets for quantization.
- **`training/`**: The core training logic.
  - `configs/`: Master configuration files and class mappings (`config.json`, `final_map.json`).
  - `scripts/`: Various training iterations (MobileNet, Production, Master).
- **`optimization/`**: Hardware-specific compilation and quantization.
  - `hailo/`: Hailo-8/8L compilation tools.
  - `sony_imx500/`: Sony IMX500 compatibility and quantization scripts.
  - `quantization/`: General TFLite, ONNX, and SavedModel export tools.
- **`utils/`**: Helper utilities.
  - `visualization/`: Confusion matrix and performance plotting.
  - `model_ops/`: Low-level manipulation (fixing outputs, stripping layers, flattening models).
- **`scripts/`**: Miscellaneous utility scripts.

## 📦 Hardware Targets

- **Hailo-8/8L**: Optimized `.hef` and `.har` files.
- **Sony IMX500**: Hardware-compatible Keras/TFLite implementations.
- **Standard Edge**: Quantized TFLite for general mobile/IoT usage.

## 📝 Usage

To ensure paths to datasets and configurations resolve correctly, it is recommended to run scripts from the project root:

```bash
# Example: Running a training script
python training/scripts/train_forager_lite.py

# Example: Generating a confusion matrix
python utils/visualization/confusion_matrix.py
```

## 🛠 Setup

This project typically uses a dedicated Python environment (e.g., `inat_env`).
```bash
source inat_env/bin/activate
# Install dependencies as needed (TensorFlow, Hailo SDK, etc.)
```

---
*Part of the [Homesteader Labs](https://github.com/thefullnacho) ecosystem.*