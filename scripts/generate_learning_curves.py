"""
Generate learning curves and visualizations from training history.
Run this after training to create figures for the final report.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import argparse


def plot_learning_curves(history, title, save_path):
    """Plot training and validation loss/accuracy curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss curves
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title(f'{title} - Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    if 'val_top5_acc' in history:
        ax2.plot(epochs, history['val_top5_acc'], 'g--', label='Val Top-5 Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'{title} - Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved learning curves to {save_path}")


def plot_comparison(baseline_history, primary_history, save_path):
    """Plot comparison of baseline vs primary model."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    baseline_epochs = range(1, len(baseline_history['val_acc']) + 1)
    primary_epochs = range(1, len(primary_history['val_acc']) + 1)
    
    # Validation accuracy comparison
    ax1.plot(baseline_epochs, baseline_history['val_acc'], 'b-', 
             label='Baseline (ResNet-50)', linewidth=2)
    ax1.plot(primary_epochs, primary_history['val_acc'], 'r-',
             label='Primary (EfficientNet-B3)', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Validation Accuracy (%)', fontsize=12)
    ax1.set_title('Model Comparison - Validation Accuracy', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # Validation loss comparison
    ax2.plot(baseline_epochs, baseline_history['val_loss'], 'b-',
             label='Baseline (ResNet-50)', linewidth=2)
    ax2.plot(primary_epochs, primary_history['val_loss'], 'r-',
             label='Primary (EfficientNet-B3)', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Model Comparison - Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved comparison plot to {save_path}")


def print_summary(history, model_name):
    """Print training summary statistics."""
    print(f"\n{'='*60}")
    print(f"{model_name} Training Summary")
    print(f"{'='*60}")
    
    print(f"Total epochs: {len(history['train_loss'])}")
    print(f"\nFinal metrics:")
    print(f"  Train Loss: {history['train_loss'][-1]:.4f}")
    print(f"  Train Accuracy: {history['train_acc'][-1]:.2f}%")
    print(f"  Val Loss: {history['val_loss'][-1]:.4f}")
    print(f"  Val Accuracy: {history['val_acc'][-1]:.2f}%")
    
    if 'val_top5_acc' in history:
        print(f"  Val Top-5 Accuracy: {history['val_top5_acc'][-1]:.2f}%")
    
    print(f"\nBest validation accuracy: {max(history['val_acc']):.2f}% (Epoch {np.argmax(history['val_acc']) + 1})")
    print(f"Best validation loss: {min(history['val_loss']):.4f} (Epoch {np.argmin(history['val_loss']) + 1})")
    
    # Check for overfitting
    final_train_acc = history['train_acc'][-1]
    final_val_acc = history['val_acc'][-1]
    gap = final_train_acc - final_val_acc
    
    print(f"\nTrain-Val gap: {gap:.2f}%")
    if gap > 10:
        print("  ⚠️  Significant overfitting detected")
    elif gap > 5:
        print("  ⚠️  Moderate overfitting")
    else:
        print("  ✓ Good generalization")


def main():
    parser = argparse.ArgumentParser(description='Generate learning curves from training history')
    parser.add_argument('--baseline-history', type=str,
                       default='checkpoints/baseline/baseline_history.json')
    parser.add_argument('--primary-history', type=str,
                       default='checkpoints/primary/primary_history.json')
    parser.add_argument('--output-dir', type=str, default='results/figures')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("GENERATING LEARNING CURVES")
    print("="*60)
    
    # Load and plot baseline history
    baseline_path = Path(args.baseline_history)
    if baseline_path.exists():
        with open(baseline_path, 'r') as f:
            baseline_history = json.load(f)
        
        print_summary(baseline_history, "Baseline (ResNet-50)")
        plot_learning_curves(
            baseline_history,
            'Baseline Model (ResNet-50)',
            output_dir / 'baseline_learning_curves.png'
        )
    else:
        print(f"\nBaseline history not found: {baseline_path}")
        baseline_history = None
    
    # Load and plot primary history
    primary_path = Path(args.primary_history)
    if primary_path.exists():
        with open(primary_path, 'r') as f:
            primary_history = json.load(f)
        
        print_summary(primary_history, "Primary (EfficientNet-B3)")
        plot_learning_curves(
            primary_history,
            'Primary Model (EfficientNet-B3)',
            output_dir / 'primary_learning_curves.png'
        )
    else:
        print(f"\nPrimary history not found: {primary_path}")
        primary_history = None
    
    # Plot comparison if both exist
    if baseline_history and primary_history:
        plot_comparison(
            baseline_history,
            primary_history,
            output_dir / 'model_comparison_curves.png'
        )
        
        print(f"\n{'='*60}")
        print("COMPARISON")
        print(f"{'='*60}")
        
        baseline_best = max(baseline_history['val_acc'])
        primary_best = max(primary_history['val_acc'])
        improvement = primary_best - baseline_best
        
        print(f"\nBest validation accuracy:")
        print(f"  Baseline: {baseline_best:.2f}%")
        print(f"  Primary:  {primary_best:.2f}%")
        print(f"  Improvement: {improvement:+.2f}%")
    
    print(f"\n{'='*60}")
    print(f"Figures saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
