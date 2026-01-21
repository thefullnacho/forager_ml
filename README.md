# Forager ML

The complete Machine Learning pipeline for **Forager**: Dataset acquisition, model training, and edge optimization for Hailo NPU and Sony IMX500.

This repository contains the source code for building the "brain" of the Forager plant identification system. It handles everything from scraping training data to exporting optimized models for specialized hardware.

## 🚀 Key Features

- **Data Acquisition**: Custom scripts for pulling and filtering images from iNaturalist (`pull_inat.py`).
- **Model Training**: Multiple architectures including MobileNetV3 and YOLO-based classifiers.
- **Optimization**: Workflows for Sony IMX500 and Hailo-8 acceleration.
- **Evaluation**: Tools for generating confusion matrices and performance metrics.

## 🛠 Project Structure

- `train_*.py`: Various training scripts for different model versions and experiments.
- `quantize_*.py`: Optimization scripts for converting models to TFLite, ONNX, and HEF.
- `rebuild_*.py`: Utilities for adjusting model architectures (stripping layers, fixing output names).
- `config.json`: Master configuration for class mappings and training parameters.

## 📦 Hardware Targets

- **Hailo-8/8L**: Optimized `.hef` and `.har` files.
- **Sony IMX500**: Hardware-compatible Keras/TFLite implementations.
- **Standard Edge**: Quantized TFLite for general mobile/IoT usage.

## 📝 Setup

This project uses a dedicated Python environment.
```bash
# Example setup (adjust based on your local env)
python -m venv inat_env
source inat_env/bin/activate
pip install -r requirements.txt  # If available
```

---
*Part of the [Homesteader Labs](https://github.com/thefullnacho) ecosystem.*
