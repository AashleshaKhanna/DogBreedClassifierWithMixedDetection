@echo off
echo ============================================================
echo Installing Dependencies for Dog Breed Classifier
echo ============================================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python not found! Please install Python first.
    pause
    exit /b 1
)
echo.

echo Installing PyTorch and dependencies...
echo This may take 5-10 minutes...
echo.

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib seaborn tqdm PyYAML Pillow scikit-learn

echo.
echo ============================================================
echo Verifying Installation
echo ============================================================
echo.

python -c "import torch; print('✓ PyTorch version:', torch.__version__)"
python -c "import torchvision; print('✓ torchvision version:', torchvision.__version__)"
python -c "import torch; print('✓ CUDA available:', torch.cuda.is_available())"
python -c "import numpy; print('✓ numpy installed')"
python -c "import pandas; print('✓ pandas installed')"
python -c "import matplotlib; print('✓ matplotlib installed')"

echo.
echo ============================================================
echo Installation Complete!
echo ============================================================
echo.
echo You can now:
echo   1. Test mock demo: python scripts\mock_demo.py --image test.jpg --scenario high_confidence
echo   2. Start training: python scripts\train_baseline.py
echo.
pause
