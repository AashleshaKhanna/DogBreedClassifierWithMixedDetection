"""
Quick setup verification script.
Run this before training to ensure everything is ready.
"""

import sys
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed."""
    print("="*60)
    print("CHECKING DEPENDENCIES")
    print("="*60)
    
    required = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'yaml': 'PyYAML',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'tqdm': 'tqdm',
        'sklearn': 'scikit-learn'
    }
    
    missing = []
    for module, name in required.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - MISSING!")
            missing.append(name)
    
    if missing:
        print(f"\n⚠️  Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True


def check_pytorch():
    """Check PyTorch installation and GPU availability."""
    print("\n" + "="*60)
    print("CHECKING PYTORCH")
    print("="*60)
    
    try:
        import torch
        print(f"✓ PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print("  → Training will use GPU (fast!)")
        else:
            print("⚠️  CUDA not available")
            print("  → Training will use CPU (slower but works)")
        
        return True
    except Exception as e:
        print(f"✗ PyTorch check failed: {e}")
        return False


def check_datasets():
    """Check if dataset files exist."""
    print("\n" + "="*60)
    print("CHECKING DATASETS")
    print("="*60)
    
    required_files = [
        'data/processed/train_combined.csv',
        'data/processed/val_combined.csv',
        'data/processed/test_combined.csv',
        'data/processed/breed_names_combined.txt'
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path} - MISSING!")
            all_exist = False
    
    if all_exist:
        print("\n✓ All dataset files found!")
    else:
        print("\n⚠️  Some dataset files are missing!")
        print("Make sure you've run the dataset preparation scripts.")
    
    return all_exist


def check_new_test_data():
    """Check new test data structure."""
    print("\n" + "="*60)
    print("CHECKING NEW TEST DATA")
    print("="*60)
    
    new_test_dir = Path('data/new_test')
    
    if not new_test_dir.exists():
        print("✗ data/new_test/ directory not found!")
        return False
    
    # Check known breeds
    known_dir = new_test_dir / 'known_breeds'
    if known_dir.exists():
        breeds = list(known_dir.iterdir())
        total_images = sum(len(list(breed_dir.glob('*'))) for breed_dir in breeds if breed_dir.is_dir())
        print(f"✓ Known breeds: {len(breeds)} breeds, {total_images} images")
    else:
        print("✗ known_breeds/ directory not found!")
    
    # Check unknown breeds
    unknown_dir = new_test_dir / 'unknown_breeds'
    if unknown_dir.exists():
        unknown_images = len(list(unknown_dir.glob('*.*')))
        print(f"✓ Unknown breeds: {unknown_images} images")
    else:
        print("✗ unknown_breeds/ directory not found!")
    
    # Check OOD
    ood_dir = new_test_dir / 'ood'
    if ood_dir.exists():
        ood_images = len(list(ood_dir.glob('*.*')))
        print(f"✓ OOD samples: {ood_images} images")
    else:
        print("✗ ood/ directory not found!")
    
    print("\n✓ New test data structure looks good!")
    return True


def check_config():
    """Check configuration file."""
    print("\n" + "="*60)
    print("CHECKING CONFIGURATION")
    print("="*60)
    
    config_path = Path('configs/config.yaml')
    if config_path.exists():
        print(f"✓ {config_path}")
        
        try:
            import yaml
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            
            print(f"  Batch size: {config['training']['batch_size']}")
            print(f"  Image size: {config['data']['image_size']}")
            print(f"  Num classes: {config['model']['num_classes']}")
            print("\n✓ Configuration loaded successfully!")
            return True
        except Exception as e:
            print(f"⚠️  Error loading config: {e}")
            return False
    else:
        print(f"✗ {config_path} - MISSING!")
        return False


def check_trained_models():
    """Check if models are already trained."""
    print("\n" + "="*60)
    print("CHECKING TRAINED MODELS")
    print("="*60)
    
    baseline_checkpoint = Path('checkpoints/baseline/baseline_best.pth')
    primary_checkpoint = Path('checkpoints/primary/primary_best.pth')
    
    baseline_exists = baseline_checkpoint.exists()
    primary_exists = primary_checkpoint.exists()
    
    if baseline_exists:
        print(f"✓ Baseline model found: {baseline_checkpoint}")
    else:
        print(f"✗ Baseline model not found")
        print("  → Need to train: python scripts/train_baseline.py")
    
    if primary_exists:
        print(f"✓ Primary model found: {primary_checkpoint}")
    else:
        print(f"✗ Primary model not found")
        print("  → Need to train: python scripts/train_primary.py")
    
    if baseline_exists and primary_exists:
        print("\n✓ Both models are trained! Ready to evaluate.")
        return True
    else:
        print("\n⚠️  Models need to be trained before evaluation.")
        return False


def main():
    """Run all checks."""
    print("\n" + "="*70)
    print(" "*15 + "SETUP VERIFICATION")
    print("="*70)
    
    checks = {
        'Dependencies': check_dependencies(),
        'PyTorch': check_pytorch(),
        'Datasets': check_datasets(),
        'New Test Data': check_new_test_data(),
        'Configuration': check_config(),
        'Trained Models': check_trained_models()
    }
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for check_name, passed in checks.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{check_name:.<40} {status}")
    
    all_passed = all(checks.values())
    
    print("\n" + "="*70)
    if all_passed:
        print("✓ ALL CHECKS PASSED!")
        print("\nYou're ready to evaluate your models!")
        print("\nNext steps:")
        print("1. python scripts/evaluate_models.py")
        print("2. python scripts/generate_learning_curves.py")
        print("3. python scripts/evaluate_new_data.py")
    else:
        print("⚠️  SOME CHECKS FAILED")
        print("\nYou need to:")
        
        if not checks['Dependencies']:
            print("1. Install dependencies: pip install -r requirements.txt")
        
        if not checks['Trained Models']:
            print("2. Train models:")
            print("   python scripts/train_baseline.py")
            print("   python scripts/train_primary.py")
        
        print("\nSee IMMEDIATE_ACTION_PLAN.md for detailed instructions.")
    
    print("="*70)


if __name__ == '__main__':
    main()
