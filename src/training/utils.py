"""
Training utilities: seed setting, logging, etc.
"""

import torch
import numpy as np
import random
import os
from pathlib import Path
import json
from datetime import datetime


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    # Make cudnn deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variable for Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"Random seed set to {seed}")


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count trainable parameters in model.
    
    Args:
        model: PyTorch model
    
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model_size_mb(model: torch.nn.Module) -> float:
    """
    Get model size in MB.
    
    Args:
        model: PyTorch model
    
    Returns:
        Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def save_config(config: dict, save_dir: Path):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_dir: Directory to save config
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    config_path = save_dir / 'config.json'
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Config saved to {config_path}")


def load_config(config_path: Path) -> dict:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
    
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    return config


class AverageMeter:
    """Computes and stores the average and current value."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Logger:
    """Simple logger for training metrics."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = self.log_dir / f'training_log_{timestamp}.txt'
        
        # Initialize log file
        with open(self.log_file, 'w') as f:
            f.write(f"Training Log - {timestamp}\n")
            f.write("=" * 60 + "\n\n")
    
    def log(self, message: str, print_msg: bool = True):
        """
        Log message to file and optionally print.
        
        Args:
            message: Message to log
            print_msg: Whether to print to console
        """
        if print_msg:
            print(message)
        
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_metrics(self, epoch: int, metrics: dict):
        """
        Log metrics for an epoch.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        message = f"Epoch {epoch}: "
        message += " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.log(message)


def get_device() -> str:
    """
    Get available device (cuda/cpu).
    
    Returns:
        Device string
    """
    if torch.cuda.is_available():
        device = 'cuda'
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    else:
        device = 'cpu'
        print("Using CPU")
    
    return device


def print_model_info(model: torch.nn.Module):
    """
    Print model information.
    
    Args:
        model: PyTorch model
    """
    num_params = count_parameters(model)
    model_size = get_model_size_mb(model)
    
    print("\nModel Information:")
    print(f"  Trainable parameters: {num_params:,}")
    print(f"  Model size: {model_size:.2f} MB")
    print(f"  Architecture: {model.__class__.__name__}")


if __name__ == '__main__':
    # Test utilities
    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
