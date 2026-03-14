"""
Visualization utilities for evaluation and analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[Path] = None
):
    """
    Plot training history (loss and accuracy curves).
    
    Args:
        history: Dictionary with training metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train Acc', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val Acc', linewidth=2)
    axes[1].plot(epochs, history['val_top5_acc'], 'g--', label='Val Top-5 Acc', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy (%)', fontsize=12)
    axes[1].set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[2].plot(epochs, history['lr'], 'purple', linewidth=2)
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('Learning Rate', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].set_yscale('log')
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[Path] = None,
    figsize: tuple = (12, 10)
):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        normalize: Whether to normalize by row
        save_path: Optional path to save figure
        figsize: Figure size
    """
    if normalize:
        cm = cm.astype('float') / (cm.sum(axis=1, keepdims=True) + 1e-8)
    
    plt.figure(figsize=figsize)
    sns.heatmap(
        cm,
        annot=False,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''),
              fontsize=14, fontweight='bold')
    plt.xticks(rotation=90, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = 15,
    save_path: Optional[Path] = None
):
    """
    Plot reliability diagram for calibration assessment.
    
    Args:
        confidences: Predicted confidences
        accuracies: Actual accuracies (0 or 1)
        n_bins: Number of bins
        save_path: Optional path to save figure
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    bin_confidences = []
    bin_accuracies = []
    bin_counts = []
    
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidences.append(confidences[mask].mean())
            bin_accuracies.append(accuracies[mask].mean())
            bin_counts.append(mask.sum())
        else:
            bin_confidences.append(0)
            bin_accuracies.append(0)
            bin_counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Reliability diagram
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration', linewidth=2)
    ax1.bar(
        bin_confidences,
        bin_accuracies,
        width=1.0/n_bins,
        alpha=0.7,
        edgecolor='black',
        label='Model'
    )
    ax1.set_xlabel('Confidence', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.set_title('Reliability Diagram', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    
    # Confidence histogram
    ax2.hist(confidences, bins=n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_per_class_accuracy(
    per_class_acc: Dict[str, float],
    save_path: Optional[Path] = None,
    top_n: int = 20
):
    """
    Plot per-class accuracy bar chart.
    
    Args:
        per_class_acc: Dictionary mapping class names to accuracies
        save_path: Optional path to save figure
        top_n: Number of classes to show (best and worst)
    """
    # Sort by accuracy
    sorted_classes = sorted(per_class_acc.items(), key=lambda x: x[1])
    
    # Get worst and best
    worst_classes = sorted_classes[:top_n]
    best_classes = sorted_classes[-top_n:]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Worst performing
    classes_worst, accs_worst = zip(*worst_classes)
    ax1.barh(range(len(classes_worst)), accs_worst, color='salmon', edgecolor='black')
    ax1.set_yticks(range(len(classes_worst)))
    ax1.set_yticklabels(classes_worst, fontsize=9)
    ax1.set_xlabel('Accuracy (%)', fontsize=12)
    ax1.set_title(f'Worst {top_n} Classes', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='x')
    
    # Best performing
    classes_best, accs_best = zip(*best_classes)
    ax2.barh(range(len(classes_best)), accs_best, color='lightgreen', edgecolor='black')
    ax2.set_yticks(range(len(classes_best)))
    ax2.set_yticklabels(classes_best, fontsize=9)
    ax2.set_xlabel('Accuracy (%)', fontsize=12)
    ax2.set_title(f'Best {top_n} Classes', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


def plot_robustness_results(
    results: Dict[str, float],
    save_path: Optional[Path] = None
):
    """
    Plot robustness evaluation results.
    
    Args:
        results: Dictionary with robustness metrics
        save_path: Optional path to save figure
    """
    categories = []
    accuracies = []
    
    for key, value in results.items():
        if 'accuracy' in key and 'top5' not in key:
            category = key.replace('_accuracy', '')
            categories.append(category)
            accuracies.append(value)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(categories, accuracies, color='steelblue', edgecolor='black', alpha=0.8)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.,
            height,
            f'{acc:.1f}%',
            ha='center',
            va='bottom',
            fontsize=11,
            fontweight='bold'
        )
    
    plt.xlabel('Robustness Category', fontsize=12)
    plt.ylabel('Top-1 Accuracy (%)', fontsize=12)
    plt.title('Robustness Evaluation Results', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    plt.ylim([0, 100])
    
    # Add target line
    plt.axhline(y=60, color='red', linestyle='--', linewidth=2, label='Target (60%)')
    plt.legend(fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    print("Visualization module loaded successfully")
