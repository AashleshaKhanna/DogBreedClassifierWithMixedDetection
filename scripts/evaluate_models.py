"""
Comprehensive evaluation script for trained models on test set.
Run this after training baseline and primary models.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm
import argparse

from src.models.baseline import create_baseline_model
from src.models.classifier import create_model
from src.data.dataset import DogBreedDataset
from src.data.augmentation import get_val_transforms
from src.training.utils import get_device


def load_model(model_type, checkpoint_path, num_classes, device):
    """Load trained model from checkpoint."""
    if model_type == 'baseline':
        model = create_baseline_model(num_classes=num_classes, pretrained=False, device=device)
    else:  # primary
        model = create_model(num_classes=num_classes, pretrained=False, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_model(model, dataloader, device, class_names):
    """Evaluate model and return detailed metrics."""
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    correct = 0
    top5_correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            
            # Top-1
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Top-5
            _, top5_pred = outputs.topk(5, 1, True, True)
            top5_correct += top5_pred.eq(labels.view(-1, 1).expand_as(top5_pred)).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # Calculate metrics
    accuracy = 100. * correct / total
    top5_accuracy = 100. * top5_correct / total
    
    # Per-class accuracy
    cm = confusion_matrix(all_labels, all_preds)
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Average confidence
    all_probs = np.array(all_probs)
    max_probs = all_probs.max(axis=1)
    avg_confidence = max_probs.mean()
    
    results = {
        'accuracy': accuracy,
        'top5_accuracy': top5_accuracy,
        'avg_confidence': avg_confidence,
        'total_samples': total,
        'correct_predictions': correct,
        'confusion_matrix': cm.tolist(),
        'per_class_accuracy': {class_names[i]: float(per_class_acc[i]) 
                               for i in range(len(class_names))},
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs.tolist()
    }
    
    return results


def plot_confusion_matrix(cm, class_names, save_path, title='Confusion Matrix'):
    """Plot confusion matrix heatmap."""
    plt.figure(figsize=(20, 18))
    
    # Only show top 30 classes for readability
    if len(class_names) > 30:
        # Get top 30 most confused classes
        class_totals = cm.sum(axis=1)
        top_indices = np.argsort(class_totals)[-30:]
        cm_subset = cm[top_indices][:, top_indices]
        class_names_subset = [class_names[i] for i in top_indices]
    else:
        cm_subset = cm
        class_names_subset = class_names
    
    sns.heatmap(cm_subset, annot=False, fmt='d', cmap='Blues',
                xticklabels=class_names_subset,
                yticklabels=class_names_subset)
    plt.title(title, fontsize=16)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confusion matrix to {save_path}")


def analyze_failures(results, class_names, top_n=10):
    """Analyze failure cases."""
    preds = np.array(results['predictions'])
    labels = np.array(results['labels'])
    probs = np.array(results['probabilities'])
    
    # Find misclassified samples
    misclassified = preds != labels
    misclassified_indices = np.where(misclassified)[0]
    
    print(f"\n{'='*60}")
    print(f"FAILURE ANALYSIS")
    print(f"{'='*60}")
    print(f"Total misclassified: {len(misclassified_indices)} / {len(labels)}")
    print(f"Error rate: {100 * len(misclassified_indices) / len(labels):.2f}%")
    
    # Most confused pairs
    confusion_pairs = {}
    for idx in misclassified_indices:
        true_label = labels[idx]
        pred_label = preds[idx]
        pair = (class_names[true_label], class_names[pred_label])
        confusion_pairs[pair] = confusion_pairs.get(pair, 0) + 1
    
    print(f"\nTop {top_n} most confused breed pairs:")
    sorted_pairs = sorted(confusion_pairs.items(), key=lambda x: x[1], reverse=True)
    for i, ((true_breed, pred_breed), count) in enumerate(sorted_pairs[:top_n], 1):
        print(f"{i}. {true_breed} → {pred_breed}: {count} times")
    
    # Low confidence errors
    misclass_probs = probs[misclassified].max(axis=1)
    low_conf_threshold = 0.5
    low_conf_errors = (misclass_probs < low_conf_threshold).sum()
    print(f"\nLow confidence errors (< {low_conf_threshold}): {low_conf_errors}")
    print(f"High confidence errors (≥ {low_conf_threshold}): {len(misclassified_indices) - low_conf_errors}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--baseline-checkpoint', type=str, 
                       default='checkpoints/baseline/baseline_best.pth')
    parser.add_argument('--primary-checkpoint', type=str,
                       default='checkpoints/primary/primary_best.pth')
    parser.add_argument('--output-dir', type=str, default='results')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("MODEL EVALUATION ON TEST SET")
    print("="*60)
    
    # Load test dataset
    print("\nLoading test dataset...")
    test_dataset = DogBreedDataset(
        image_dir='src/data',
        annotations_file=config['data']['test_annotations'],
        transform=get_val_transforms(config['data']['image_size']),
        use_combined=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['data']['num_workers'],
        pin_memory=config['data']['pin_memory']
    )
    
    class_names = [breed for breed, _ in sorted(test_dataset.breed_to_idx.items(), 
                                                  key=lambda x: x[1])]
    
    print(f"Test samples: {len(test_dataset)}")
    print(f"Number of classes: {len(class_names)}")
    
    # Evaluate baseline model
    if Path(args.baseline_checkpoint).exists():
        print("\n" + "="*60)
        print("EVALUATING BASELINE MODEL (ResNet-50)")
        print("="*60)
        
        baseline_model = load_model('baseline', args.baseline_checkpoint, 
                                    len(class_names), device)
        baseline_results = evaluate_model(baseline_model, test_loader, device, class_names)
        
        print(f"\nBaseline Results:")
        print(f"  Test Accuracy: {baseline_results['accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {baseline_results['top5_accuracy']:.2f}%")
        print(f"  Avg Confidence: {baseline_results['avg_confidence']:.4f}")
        
        # Save results
        with open(output_dir / 'baseline_test_results.json', 'w') as f:
            # Remove large arrays before saving
            save_results = {k: v for k, v in baseline_results.items() 
                           if k not in ['predictions', 'labels', 'probabilities']}
            json.dump(save_results, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(baseline_results['confusion_matrix']),
            class_names,
            output_dir / 'baseline_confusion_matrix.png',
            'Baseline Model - Confusion Matrix (Top 30 Classes)'
        )
        
        # Failure analysis
        analyze_failures(baseline_results, class_names)
    else:
        print(f"\nBaseline checkpoint not found: {args.baseline_checkpoint}")
        print("Skipping baseline evaluation.")
    
    # Evaluate primary model
    if Path(args.primary_checkpoint).exists():
        print("\n" + "="*60)
        print("EVALUATING PRIMARY MODEL (EfficientNet-B3)")
        print("="*60)
        
        primary_model = load_model('primary', args.primary_checkpoint,
                                   len(class_names), device)
        primary_results = evaluate_model(primary_model, test_loader, device, class_names)
        
        print(f"\nPrimary Results:")
        print(f"  Test Accuracy: {primary_results['accuracy']:.2f}%")
        print(f"  Top-5 Accuracy: {primary_results['top5_accuracy']:.2f}%")
        print(f"  Avg Confidence: {primary_results['avg_confidence']:.4f}")
        
        # Save results
        with open(output_dir / 'primary_test_results.json', 'w') as f:
            save_results = {k: v for k, v in primary_results.items()
                           if k not in ['predictions', 'labels', 'probabilities']}
            json.dump(save_results, f, indent=2)
        
        # Plot confusion matrix
        plot_confusion_matrix(
            np.array(primary_results['confusion_matrix']),
            class_names,
            output_dir / 'primary_confusion_matrix.png',
            'Primary Model - Confusion Matrix (Top 30 Classes)'
        )
        
        # Failure analysis
        analyze_failures(primary_results, class_names)
    else:
        print(f"\nPrimary checkpoint not found: {args.primary_checkpoint}")
        print("Skipping primary evaluation.")
    
    # Comparison
    if Path(args.baseline_checkpoint).exists() and Path(args.primary_checkpoint).exists():
        print("\n" + "="*60)
        print("MODEL COMPARISON")
        print("="*60)
        
        improvement = primary_results['accuracy'] - baseline_results['accuracy']
        print(f"\nAccuracy Improvement: {improvement:+.2f}%")
        print(f"  Baseline: {baseline_results['accuracy']:.2f}%")
        print(f"  Primary:  {primary_results['accuracy']:.2f}%")
        
        top5_improvement = primary_results['top5_accuracy'] - baseline_results['top5_accuracy']
        print(f"\nTop-5 Accuracy Improvement: {top5_improvement:+.2f}%")
        print(f"  Baseline: {baseline_results['top5_accuracy']:.2f}%")
        print(f"  Primary:  {primary_results['top5_accuracy']:.2f}%")
        
        # Save comparison
        comparison = {
            'baseline': {
                'accuracy': baseline_results['accuracy'],
                'top5_accuracy': baseline_results['top5_accuracy'],
                'avg_confidence': baseline_results['avg_confidence']
            },
            'primary': {
                'accuracy': primary_results['accuracy'],
                'top5_accuracy': primary_results['top5_accuracy'],
                'avg_confidence': primary_results['avg_confidence']
            },
            'improvement': {
                'accuracy': improvement,
                'top5_accuracy': top5_improvement
            }
        }
        
        with open(output_dir / 'model_comparison.json', 'w') as f:
            json.dump(comparison, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
