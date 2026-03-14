# Setup Guide: Installing Dependencies

## 🚨 Current Issue

You're getting `ModuleNotFoundError: No module named 'torch'` because PyTorch and other dependencies aren't installed yet.

## ✅ Quick Fix

### Option 1: Install All Dependencies (Recommended)

```bash
pip install -r requirements.txt
```

This installs:
- PyTorch (deep learning framework)
- torchvision (image processing)
- numpy, pandas (data processing)
- matplotlib, seaborn (visualization)
- And other required packages

### Option 2: Install PyTorch Only (Minimal)

If you just want to test quickly:

```bash
# For CPU only (slower training)
pip install torch torchvision

# For GPU (CUDA 11.8) - much faster
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For GPU (CUDA 12.1) - latest
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## 🔍 Check Your Setup

### 1. Check if you have GPU available:

```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

### 2. Verify installation:

```bash
python -c "import torch; import torchvision; print('PyTorch version:', torch.__version__); print('torchvision version:', torchvision.__version__)"
```

## 📦 Full Installation Steps

### Step 1: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 3: Verify Installation

```bash
python -c "import torch; print('✓ PyTorch installed')"
python -c "import torchvision; print('✓ torchvision installed')"
python -c "import numpy; print('✓ numpy installed')"
python -c "import pandas; print('✓ pandas installed')"
python -c "import matplotlib; print('✓ matplotlib installed')"
```

## ⚡ GPU vs CPU

### If You Have NVIDIA GPU:

1. Check CUDA version:
```bash
nvidia-smi
```

2. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

3. Verify GPU is detected:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

### If You Don't Have GPU (CPU Only):

Training will be MUCH slower (days instead of hours), but it will work:

```bash
pip install torch torchvision
```

## 🎯 After Installation

Once dependencies are installed, you can:

### 1. Test the mock demo (no training needed):
```bash
python scripts/mock_demo.py --image test.jpg --scenario high_confidence
```

### 2. Start training:
```bash
python scripts/train_baseline.py
```

## 🔧 Troubleshooting

### "pip: command not found"
Try `pip3` instead of `pip`:
```bash
pip3 install -r requirements.txt
```

### "Permission denied"
Use `--user` flag:
```bash
pip install --user -r requirements.txt
```

### "CUDA out of memory" (during training)
Reduce batch size in `configs/config.yaml`:
```yaml
batch_size: 8  # or even 4
```

### "Slow training on CPU"
- Consider using Google Colab (free GPU)
- Or reduce dataset size for testing
- Or use a smaller model

## 📊 Disk Space Requirements

- PyTorch: ~2-3 GB
- Other dependencies: ~500 MB
- Dataset: Already have (~5 GB)
- Model checkpoints: ~500 MB
- **Total**: ~8-9 GB

## 🚀 Quick Start After Installation

```bash
# 1. Verify installation
python -c "import torch; print('PyTorch ready!')"

# 2. Test mock demo
python scripts/mock_demo.py --image test.jpg --scenario high_confidence

# 3. Start training (if you have time and GPU)
python scripts/train_baseline.py
```

## 💡 Alternative: Google Colab

If you don't have a GPU or don't want to install locally:

1. Upload your code to Google Drive
2. Open Google Colab (free GPU)
3. Mount Drive and run training there
4. Download trained models back

## ⏱️ Expected Installation Time

- Installing dependencies: 5-10 minutes
- Verifying installation: 1 minute
- **Total**: ~10 minutes

## 📝 Summary

**Minimum to run training**:
```bash
pip install torch torchvision numpy pandas matplotlib tqdm PyYAML
```

**Recommended (all dependencies)**:
```bash
pip install -r requirements.txt
```

**Then verify**:
```bash
python -c "import torch; print('Ready to train!')"
```

---

**After installation, run**: `python scripts/train_baseline.py`
