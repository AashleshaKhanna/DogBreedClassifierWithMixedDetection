"""
Training script for baseline ResNet-50 model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
import argparse
from tqdm import tqdm

from src.models.baseline import create_baseline_model
from src.data.dataset import DogBreedDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.utils import set_seed, get_device, AverageMeter
from src.training.train import save_checkpoint


def train_baseline(config_path: str, output_dir: str):
    """Train baseline ResNet-50 model."""
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(config['seed'])
    
    # Device
    device = get_device()
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Training Baseline ResNet-50 Model")
    print("="*60)
    
    # Load datasets
    print("\nLoading datasets...")
    train_dataset = DogBreedDataset(
        image_dir='src/data',
        annotations_file=config['data']['train_annotations'],
        transform=get_train_transforms(config['data']['image_size']),
        use_combined=True
    )
    
    val_dataset = DogBreedDataset(
        image_dir='src/data',
        annotations_file=config['data']['val_annotations'],
        transform=get_val_transforms(config['data']['image_size']),
        breed_to_idx=train_dataset.breed_to_idx,
        use_combined=True
    )
    
    # Weighted sampling for class imbalance
    sample_weights = train_dataset.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        sampler=sampler,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Number of classes: {train_dataset.num_classes}")
    
    # Create model
    print("\nCreating baseline model...")
    model = create_baseline_model(
        num_classes=train_dataset.num_classes,
        pretrained=True,
        device=device
    )
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['phase1']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Training loop
    print("\nStarting training...")
    print("Phase 1: Frozen backbone (5 epochs)")
    
    model.freeze_backbone()
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Phase 1: Frozen backbone
    for epoch in range(1, 6):
        model.train()
        train_loss = AverageMeter()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/5')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss.avg, 'acc': 100. * train_correct / train_total})
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss.update(loss.item(), images.size(0))
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc}, 
                          output_dir, 'baseline_best.pth')
    
    # Phase 2: Full fine-tuning
    print("\nPhase 2: Full fine-tuning (15 epochs)")
    model.unfreeze_all()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['phase2']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    for epoch in range(6, 21):
        model.train()
        train_loss = AverageMeter()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch}/20')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss.avg, 'acc': 100. * train_correct / train_total})
        
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = AverageMeter()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss.update(loss.item(), images.size(0))
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_acc = 100. * val_correct / val_total
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss.avg)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch}: Train Loss: {train_loss.avg:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss.avg:.4f}, Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, {'val_acc': val_acc}, 
                          output_dir, 'baseline_best.pth')
    
    # Save final model
    save_checkpoint(model, optimizer, 20, {'val_acc': val_acc}, 
                   output_dir, 'baseline_final.pth')
    
    # Save history
    import json
    with open(output_dir / 'baseline_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train baseline ResNet-50 model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='checkpoints/baseline',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    train_baseline(args.config, args.output)
