# Dog Breed Classifier with Mixed Detection

A deep learning system for dog breed classification that handles real-world photo conditions, detects out-of-distribution inputs, and provides calibrated confidence scores.

## Features

- **Multi-stage Pipeline**: Dog detection → Breed classification → Confidence calibration
- **Robustness**: Handles blur, poor lighting, partial views, and multiple dogs
- **Unknown Detection**: Rejects non-dog images and uncertain predictions
- **Calibrated Confidence**: Temperature-scaled probabilities for reliable uncertainty estimates
- **Interpretability**: Grad-CAM visualizations showing model attention

## Architecture

1. **Stage 1 - Dog Detection**: YOLOv5 for localizing dogs in images
2. **Stage 2 - Breed Classification**: EfficientNet-B3 with custom classifier head
3. **Stage 3 - Calibration**: Temperature scaling for confidence calibration

## Project Structure

```
.
├── configs/
│   └── config.yaml              # Hyperparameters and settings
├── data/
│   ├── raw/                     # Original datasets
│   ├── processed/               # Preprocessed data
│   └── robustness_test/         # Robustness evaluation sets
├── src/
│   ├── data/
│   │   ├── dataset.py           # PyTorch Dataset classes
│   │   └── augmentation.py      # Data augmentation pipeline
│   ├── models/
│   │   ├── detector.py          # Dog detection (YOLOv5)
│   │   ├── classifier.py        # Breed classifier (EfficientNet)
│   │   └── calibration.py       # Temperature scaling
│   ├── training/
│   │   ├── train.py             # Training loop
│   │   ├── evaluate.py          # Evaluation functions
│   │   └── utils.py             # Utilities (seed, logging)
│   └── visualization/
│       ├── gradcam.py           # Grad-CAM implementation
│       └── plots.py             # Visualization utilities
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_evaluation_analysis.ipynb
├── checkpoints/                 # Saved models
├── logs/                        # Training logs
├── results/                     # Evaluation results
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Download Dataset

Download the Stanford Dogs Dataset:
```bash
# Create data directory
mkdir -p data/raw

# Download Stanford Dogs (manual download required)
# Visit: http://vision.stanford.edu/aditya86/ImageNetDogs/
# Extract to data/raw/stanford_dogs/
```

### 3. Prepare Data

```bash
# Create train/val/test splits
python scripts/prepare_data.py --data_dir data/raw/stanford_dogs --output_dir data/processed

# Create robustness test sets
python scripts/create_robustness_tests.py --input_dir data/processed --output_dir data/robustness_test
```

## Usage

### Training

#### Phase 1: Train with frozen backbone
```bash
python scripts/train_phase1.py --config configs/config.yaml
```

#### Phase 2: Fine-tune last blocks
```bash
python scripts/train_phase2.py --config configs/config.yaml --checkpoint checkpoints/phase1_best.pth
```

#### Phase 3: Full fine-tuning
```bash
python scripts/train_phase3.py --config configs/config.yaml --checkpoint checkpoints/phase2_best.pth
```

### Calibration

```bash
python scripts/calibrate.py --model checkpoints/phase3_best.pth --val_data data/processed/val.csv
```

### Evaluation

```bash
# Standard evaluation
python scripts/evaluate.py --model checkpoints/calibrated_model.pth --test_data data/processed/test.csv

# Robustness evaluation
python scripts/evaluate_robustness.py --model checkpoints/calibrated_model.pth --robustness_dir data/robustness_test

# OOD detection evaluation
python scripts/evaluate_ood.py --model checkpoints/calibrated_model.pth --ood_dir data/ood_samples
```

### Inference on Single Image

```bash
python scripts/predict.py --model checkpoints/calibrated_model.pth --image path/to/dog.jpg
```

## Performance Targets

- **Top-1 Accuracy**: ≥ 70%
- **Top-5 Accuracy**: ≥ 85%
- **Expected Calibration Error (ECE)**: ≤ 0.15
- **OOD Detection AUROC**: ≥ 0.80
- **Robustness Accuracy**: ≥ 60% on challenging conditions
- **Inference Time**: ≤ 5 seconds per image
- **Model Size**: ≤ 500 MB

## Configuration

Edit `configs/config.yaml` to modify:
- Model architecture and hyperparameters
- Training phases and learning rates
- Data augmentation settings
- Detection and calibration parameters
- Paths and logging options

## Reproducibility

All experiments use fixed random seeds (default: 42) for reproducibility:
```python
from src.training.utils import set_seed
set_seed(42)
```

## Citation

If you use this code, please cite:
- Stanford Dogs Dataset: Khosla et al., 2011
- EfficientNet: Tan & Le, 2019
- YOLOv5: Ultralytics, 2020

## License

This project is for academic use only (APS360 course project).

## Acknowledgments

- Stanford Dogs Dataset
- PyTorch and timm libraries
- YOLOv5 by Ultralytics

