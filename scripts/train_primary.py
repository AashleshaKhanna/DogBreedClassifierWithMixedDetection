"""
Training script for primary EfficientNet-B3 model.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch.cuda.amp import GradScaler, autocast
import yaml
import argparse
import json
from tqdm import tqdm

from src.models.classifier import create_model, LabelSmoothingCrossEntropy
from src.data.dataset import DogBreedDataset
from src.data.augmentation import get_train_transforms, get_val_transforms
from src.training.utils import set_seed, get_device, AverageMeter
from src.training.train import save_checkpoint


def run_epoch(model, loader, criterion, optimizer, scaler, use_amp, device, desc):
    """Single training epoch with optional AMP."""
    model.train()
    loss_meter = AverageMeter()
    correct = total = 0

    pbar = tqdm(loader, desc=desc)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        if use_amp:
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        loss_meter.update(loss.item(), images.size(0))
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss_meter.avg, 'acc': 100. * correct / total})

    return loss_meter.avg, 100. * correct / total


def validate(model, val_loader, criterion, device):
    """Validation function."""
    model.eval()
    val_loss = AverageMeter()
    correct = top5_correct = total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss.update(loss.item(), images.size(0))
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()

    return {
        'loss': val_loss.avg,
        'acc': 100. * correct / total,
        'top5_acc': 100. * top5_correct / total
    }


def train_primary(config_path: str, output_dir: str):
    """Train primary EfficientNet-B3 model."""

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    set_seed(config['seed'])
    device = get_device()
    use_amp = config.get('mixed_precision', False) and device == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp:
        print("Mixed precision (AMP) enabled")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*60)
    print("Training Primary EfficientNet-B3 Model")
    print("="*60)

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

    sampler = WeightedRandomSampler(
        weights=train_dataset.get_sample_weights(),
        num_samples=len(train_dataset),
        replacement=True
    )

    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'],
                              sampler=sampler, num_workers=config['data']['num_workers'],
                              pin_memory=config['data']['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'],
                            shuffle=False, num_workers=config['data']['num_workers'],
                            pin_memory=config['data']['pin_memory'])

    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Classes: {train_dataset.num_classes}")

    print("\nCreating primary model...")
    model = create_model(num_classes=train_dataset.num_classes, pretrained=True, device=device)
    criterion = LabelSmoothingCrossEntropy(epsilon=config['training']['label_smoothing'])

    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_top5_acc': []}

    # ── Phase 1: Frozen backbone ──────────────────────────────────────────
    print("\nPhase 1: Frozen backbone (10 epochs)")
    model.freeze_backbone()
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase1']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    for epoch in range(1, 11):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp, device,
            f'Phase 1 - Epoch {epoch}/10'
        )
        val_metrics = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_top5_acc'].append(val_metrics['top5_acc'])

        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_metrics['acc']:.2f}%, "
              f"Val Top-5: {val_metrics['top5_acc']:.2f}%")

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, 'primary_best.pth')

    # ── Phase 2: Fine-tune last 2 blocks ─────────────────────────────────
    print("\nPhase 2: Fine-tune last 2 blocks (20 epochs)")
    model.unfreeze_backbone(num_blocks=2)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config['training']['phase2']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )

    for epoch in range(11, 31):
        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, optimizer, scaler, use_amp, device,
            f'Phase 2 - Epoch {epoch}/30'
        )
        val_metrics = validate(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['acc'])
        history['val_top5_acc'].append(val_metrics['top5_acc'])

        print(f"Epoch {epoch}: Train Acc: {train_acc:.2f}%, Val Acc: {val_metrics['acc']:.2f}%, "
              f"Val Top-5: {val_metrics['top5_acc']:.2f}%")

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            save_checkpoint(model, optimizer, epoch, val_metrics, output_dir, 'primary_best.pth')

    save_checkpoint(model, optimizer, 30, val_metrics, output_dir, 'primary_final.pth')

    with open(output_dir / 'primary_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete! Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train primary EfficientNet-B3 model')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--output', type=str, default='checkpoints/primary')
    args = parser.parse_args()
    train_primary(args.config, args.output)
