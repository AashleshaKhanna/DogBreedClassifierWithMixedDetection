"""
Training script for dog breed classifier.
Implements 3-phase training strategy with proper validation and checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import time
from typing import Dict, Optional
from pathlib import Path
import json


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int
) -> Dict[str, float]:
    """
    Train for one epoch.
    
    Args:
        model: Classification model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    running_loss = 0.0
    correct = 0
    total = 0
    
    start_time = time.time()
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        # Print progress
        if (batch_idx + 1) % 50 == 0:
            print(f"Epoch {epoch} [{batch_idx + 1}/{len(train_loader)}] "
                  f"Loss: {running_loss / (batch_idx + 1):.4f} "
                  f"Acc: {100. * correct / total:.2f}%")
    
    epoch_time = time.time() - start_time
    
    metrics = {
        'loss': running_loss / len(train_loader),
        'accuracy': 100. * correct / total,
        'time': epoch_time
    }
    
    return metrics


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: str
) -> Dict[str, float]:
    """
    Validate model.
    
    Args:
        model: Classification model
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to run on
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    running_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            # Top-1 accuracy
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5 accuracy
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
    
    metrics = {
        'loss': running_loss / len(val_loader),
        'accuracy': 100. * correct / total,
        'top5_accuracy': 100. * top5_correct / total
    }
    
    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    filename: str
):
    """Save model checkpoint."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    
    filepath = checkpoint_dir / filename
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[optim.Optimizer],
    checkpoint_path: Path,
    device: str
) -> int:
    """
    Load model checkpoint.
    
    Returns:
        Starting epoch number
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint['epoch']
    metrics = checkpoint.get('metrics', {})
    
    print(f"Checkpoint loaded from epoch {epoch}")
    print(f"Metrics: {metrics}")
    
    return epoch


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: optim.lr_scheduler._LRScheduler,
    num_epochs: int,
    device: str,
    checkpoint_dir: Path,
    early_stopping_patience: int = 5
) -> Dict[str, list]:
    """
    Full training loop with validation and checkpointing.
    
    Args:
        model: Classification model
        train_loader: Training data loader
        val_loader: Validation data loader
        criterion: Loss function
        optimizer: Optimizer
        scheduler: Learning rate scheduler
        num_epochs: Number of epochs to train
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Patience for early stopping
    
    Returns:
        Dictionary with training history
    """
    best_val_acc = 0.0
    best_val_loss = float('inf')
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'val_top5_acc': [],
        'lr': []
    }
    
    for epoch in range(1, num_epochs + 1):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{num_epochs}")
        print(f"{'='*60}")
        
        # Training
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validation
        val_metrics = validate(model, val_loader, criterion, device)
        
        # Update history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_top5_acc'].append(val_metrics['top5_accuracy'])
        history['lr'].append(optimizer.param_groups[0]['lr'])
        
        # Print epoch summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"Train Loss: {train_metrics['loss']:.4f} | Train Acc: {train_metrics['accuracy']:.2f}%")
        print(f"Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}% | "
              f"Val Top-5: {val_metrics['top5_accuracy']:.2f}%")
        print(f"Time: {train_metrics['time']:.2f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Learning rate scheduling
        scheduler.step(val_metrics['accuracy'])
        
        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir, 'best_model.pth'
            )
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Save best loss model
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir, 'best_loss_model.pth'
            )
        
        # Periodic checkpoint
        if epoch % 5 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics,
                checkpoint_dir, f'checkpoint_epoch_{epoch}.pth'
            )
        
        # Early stopping
        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {epoch} epochs")
            break
    
    # Save final model
    save_checkpoint(
        model, optimizer, epoch, val_metrics,
        checkpoint_dir, 'final_model.pth'
    )
    
    # Save training history
    history_path = checkpoint_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best validation loss: {best_val_loss:.4f}")
    
    return history


if __name__ == '__main__':
    print("Training module loaded successfully")
