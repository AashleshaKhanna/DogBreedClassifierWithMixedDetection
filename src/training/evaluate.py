"""
Evaluation functions for model performance assessment.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score
from typing import Dict, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_accuracy(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    top_k: int = 5
) -> Dict[str, float]:
    """
    Evaluate model accuracy (Top-1 and Top-K).
    
    Args:
        model: Classification model
        data_loader: Data loader
        device: Device to run on
        top_k: K for Top-K accuracy
    
    Returns:
        Dictionary with accuracy metrics
    """
    model.eval()
    
    correct_top1 = 0
    correct_topk = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            # Top-1
            _, pred_top1 = outputs.max(1)
            correct_top1 += pred_top1.eq(labels).sum().item()
            
            # Top-K
            _, pred_topk = outputs.topk(top_k, 1, True, True)
            correct_topk += pred_topk.eq(labels.view(-1, 1).expand_as(pred_topk)).sum().item()
            
            total += labels.size(0)
    
    return {
        'top1_accuracy': 100. * correct_top1 / total,
        f'top{top_k}_accuracy': 100. * correct_topk / total,
        'total_samples': total
    }


def evaluate_per_class(
    model: nn.Module,
    data_loader: DataLoader,
    device: str,
    class_names: List[str]
) -> Dict[str, float]:
    """
    Evaluate per-class accuracy.
    
    Args:
        model: Classification model
        data_loader: Data loader
        device: Device to run on
        class_names: List of class names
    
    Returns:
        Dictionary mapping class names to accuracies
    """
    model.eval()
    
    class_correct = np.zeros(len(class_names))
    class_total = np.zeros(len(class_names))
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = outputs.max(1)
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == label:
                    class_correct[label] += 1
    
    per_class_acc = {}
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            acc = 100. * class_correct[i] / class_total[i]
            per_class_acc[class_name] = acc
        else:
            per_class_acc[class_name] = 0.0
    
    return per_class_acc


def get_predictions(
    model: nn.Module,
    data_loader: DataLoader,
    device: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get model predictions and probabilities.
    
    Args:
        model: Classification model
        data_loader: Data loader
        device: Device to run on
    
    Returns:
        predictions: Predicted class indices
        probabilities: Predicted probabilities
        labels: True labels
    """
    model.eval()
    
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    predictions = np.concatenate(all_preds)
    probabilities = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    
    return predictions, probabilities, labels


def compute_confusion_matrix(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
) -> np.ndarray:
    """Compute confusion matrix."""
    cm = confusion_matrix(labels, predictions)
    return cm


def evaluate_ood_detection(
    model: nn.Module,
    in_dist_loader: DataLoader,
    ood_loader: DataLoader,
    device: str
) -> Dict[str, float]:
    """
    Evaluate out-of-distribution detection performance.
    
    Args:
        model: Classification model
        in_dist_loader: In-distribution data loader
        ood_loader: Out-of-distribution data loader
        device: Device to run on
    
    Returns:
        Dictionary with OOD detection metrics
    """
    model.eval()
    
    # Get max confidences for in-distribution
    in_dist_confidences = []
    with torch.no_grad():
        for images, _ in in_dist_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = probs.max(1)
            in_dist_confidences.extend(max_probs.cpu().numpy())
    
    # Get max confidences for OOD
    ood_confidences = []
    with torch.no_grad():
        for images in ood_loader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            max_probs, _ = probs.max(1)
            ood_confidences.extend(max_probs.cpu().numpy())
    
    # Create labels (1 = in-dist, 0 = OOD)
    y_true = np.concatenate([
        np.ones(len(in_dist_confidences)),
        np.zeros(len(ood_confidences))
    ])
    
    # Scores (higher confidence = more likely in-dist)
    y_scores = np.concatenate([in_dist_confidences, ood_confidences])
    
    # Compute AUROC
    auroc = roc_auc_score(y_true, y_scores)
    
    # Compute precision/recall at different thresholds
    thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]
    threshold_metrics = {}
    
    for thresh in thresholds:
        y_pred = (y_scores >= thresh).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tn = ((y_pred == 0) & (y_true == 0)).sum()
        fn = ((y_pred == 0) & (y_true == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        threshold_metrics[f'precision@{thresh}'] = precision
        threshold_metrics[f'recall@{thresh}'] = recall
    
    return {
        'auroc': auroc,
        **threshold_metrics,
        'in_dist_mean_conf': np.mean(in_dist_confidences),
        'ood_mean_conf': np.mean(ood_confidences)
    }


def evaluate_robustness(
    model: nn.Module,
    robustness_loaders: Dict[str, DataLoader],
    device: str
) -> Dict[str, float]:
    """
    Evaluate model on robustness test sets.
    
    Args:
        model: Classification model
        robustness_loaders: Dict mapping category names to data loaders
        device: Device to run on
    
    Returns:
        Dictionary with accuracy per robustness category
    """
    results = {}
    
    for category, loader in robustness_loaders.items():
        metrics = evaluate_accuracy(model, loader, device)
        results[f'{category}_accuracy'] = metrics['top1_accuracy']
        results[f'{category}_top5_accuracy'] = metrics['top5_accuracy']
    
    return results


def print_classification_report(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str]
):
    """Print detailed classification report."""
    report = classification_report(
        labels,
        predictions,
        target_names=class_names,
        digits=3
    )
    print("\nClassification Report:")
    print(report)


if __name__ == '__main__':
    print("Evaluation module loaded successfully")
