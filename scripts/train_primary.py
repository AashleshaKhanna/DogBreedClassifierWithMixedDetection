"""
Training script for primary EfficientNet-B3 model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
import yaml
import argparse
from tqdm import tqdm

from src.models.classifier import create_model, LabelSmoothingCrossEntropy
from src.data.dataset import DogBreedDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.utils import set_seed, get_device, AverageMeter
from src.training.train import save_checkpoint


def train_primary(config_path: str, output_dir: str):
    """Train primary EfficientNet-B3 model."""
    
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
    print("Training Primary EfficientNet-B3 Model")
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
    
    # Weighted sampling
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
    print("\nCreating primary model...")
    model = create_model(
        num_classes=train_dataset.num_classes,
        pretrained=True,
        device=device
    )
    
    # Loss and optimizer
    criterion = LabelSmoothingCrossEntropy(epsilon=config['training']['label_smoothing'])
    
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_top5_acc': []}
    
    # Phase 1: Frozen backbone
    print("\nPhase 1: Frozen backbone (10 epochs)")
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase1']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    for epoch in range(1, 11):
        # Training
        model.train()
        train_loss = AverageMeter()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Phase 1 - Epoch {epoch}/10')
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
        val_metrics = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_top5_acc'].append(val_metrics['top5_acc'])
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_metrics['acc']:.2f}%, "
              f"Val Top-5: {val_metrics['top5_acc']:.2f}%")
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, epoch, val_metrics, 
                          output_dir, 'primary_best.pth')
    
    # Phase 2: Fine-tune last blocks
    print("\nPhase 2: Fine-tune last 2 blocks (20 epochs)")
    model.unfreeze_backbone(num_blocks=2)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase2']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    for epoch in range(11, 31):
        # Training
        model.train()
        train_loss = AverageMeter()
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Phase 2 - Epoch {epoch}/30')
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
        val_metrics = validate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss.avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_top5_acc'].append(val_metrics['top5_acc'])
        
        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_metrics['acc']:.2f}%, "
              f"Val Top-5: {val_metrics['top5_acc']:.2f}%")
        
        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, epoch, val_metrics, 
                          output_dir, 'primary_best.pth')
    
    # Save final model and history
    save_checkpoint(model, optimizer, 30, val_metrics, 
                   output_dir, 'primary_final.pth')
    
    import json
    with open(output_dir / 'primary_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_dir}")


def validate(model, val_loader, criterion, device):
    """Validation function."""
    model.eval()
    val_loss = AverageMeter()
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            val_loss.update(loss.item(), images.size(0))
            
            # Top-1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
    
    return {
        'loss': val_loss.avg,
        'acc': 100. * correct / total,
        'top5_acc': 100. * top5_correct / total
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train primary EfficientNet-B3 model')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='checkpoints/primary',
                       help='Output directory for checkpoints')
    
    args = parser.parse_args()
    
    train_primary(args.config, args.output)
