"""
Confidence calibration using temperature scaling.
Stage 3 of the pipeline: Calibrate prediction confidences.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Tuple


class TemperatureScaling(nn.Module):
    """
    Temperature scaling for model calibration.
    
    Applies a learned temperature parameter to logits:
        p_calibrated = softmax(logits / T)
    """
    
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)
    
    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        """
        Apply temperature scaling to logits.
        
        Args:
            logits: Model logits (B, num_classes)
        
        Returns:
            Calibrated probabilities (B, num_classes)
        """
        return torch.softmax(logits / self.temperature, dim=1)
    
    def get_temperature(self) -> float:
        """Get current temperature value."""
        return self.temperature.item()


def train_temperature_scaling(
    model: nn.Module,
    val_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    max_iter: int = 50,
    lr: float = 0.01
) -> TemperatureScaling:
    """
    Train temperature scaling on validation set.
    
    Args:
        model: Trained classification model
        val_loader: Validation data loader
        device: Device to run on
        max_iter: Maximum optimization iterations
        lr: Learning rate for temperature optimization
    
    Returns:
        Trained TemperatureScaling module
    """
    model.eval()
    
    # Collect logits and labels
    all_logits = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            logits = model(images)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    
    all_logits = torch.cat(all_logits, dim=0).to(device)
    all_labels = torch.cat(all_labels, dim=0).to(device)
    
    # Initialize temperature scaling
    temp_model = TemperatureScaling().to(device)
    
    # Optimize temperature
    optimizer = optim.LBFGS([temp_model.temperature], lr=lr, max_iter=max_iter)
    criterion = nn.CrossEntropyLoss()
    
    def eval_loss():
        optimizer.zero_grad()
        calibrated_probs = temp_model(all_logits)
        loss = criterion(torch.log(calibrated_probs + 1e-8), all_labels)
        loss.backward()
        return loss
    
    optimizer.step(eval_loss)
    
    print(f"Optimal temperature: {temp_model.get_temperature():.4f}")
    
    return temp_model


def compute_ece(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the difference between predicted confidence and actual accuracy.
    
    Args:
        probs: Predicted probabilities (N, num_classes)
        labels: True labels (N,)
        n_bins: Number of bins for calibration
    
    Returns:
        ece: Expected Calibration Error
    """
    confidences = np.max(probs, axis=1)
    predictions = np.argmax(probs, axis=1)
    accuracies = (predictions == labels).astype(float)
    
    # Create bins
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    ece = 0.0
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidence = confidences[mask].mean()
            bin_accuracy = accuracies[mask].mean()
            bin_weight = mask.sum() / len(confidences)
            ece += bin_weight * abs(bin_confidence - bin_accuracy)
    
    return ece


def evaluate_calibration(
    model: nn.Module,
    temp_model: TemperatureScaling,
    data_loader: DataLoader,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> Tuple[float, float]:
    """
    Evaluate calibration before and after temperature scaling.
    
    Args:
        model: Classification model
        temp_model: Temperature scaling module
        data_loader: Data loader for evaluation
        device: Device to run on
    
    Returns:
        ece_before: ECE before calibration
        ece_after: ECE after calibration
    """
    model.eval()
    temp_model.eval()
    
    all_probs_before = []
    all_probs_after = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in data_loader:
            images = images.to(device)
            logits = model(images)
            
            # Before calibration
            probs_before = torch.softmax(logits, dim=1)
            all_probs_before.append(probs_before.cpu().numpy())
            
            # After calibration
            probs_after = temp_model(logits)
            all_probs_after.append(probs_after.cpu().numpy())
            
            all_labels.append(labels.numpy())
    
    all_probs_before = np.concatenate(all_probs_before, axis=0)
    all_probs_after = np.concatenate(all_probs_after, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    ece_before = compute_ece(all_probs_before, all_labels)
    ece_after = compute_ece(all_probs_after, all_labels)
    
    return ece_before, ece_after


class CalibratedClassifier(nn.Module):
    """Wrapper combining classifier and temperature scaling."""
    
    def __init__(
        self,
        classifier: nn.Module,
        temp_model: TemperatureScaling
    ):
        super().__init__()
        self.classifier = classifier
        self.temp_model = temp_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with calibration.
        
        Args:
            x: Input images (B, 3, H, W)
        
        Returns:
            Calibrated probabilities (B, num_classes)
        """
        logits = self.classifier(x)
        probs = self.temp_model(logits)
        return probs
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get uncalibrated logits."""
        return self.classifier(x)


if __name__ == '__main__':
    # Test temperature scaling
    print("Testing temperature scaling...")
    
    # Create dummy data
    logits = torch.randn(100, 10)
    labels = torch.randint(0, 10, (100,))
    
    # Create temperature model
    temp_model = TemperatureScaling()
    
    # Get calibrated probabilities
    probs = temp_model(logits)
    
    print(f"Temperature: {temp_model.get_temperature():.4f}")
    print(f"Probs shape: {probs.shape}")
    print(f"Probs sum: {probs.sum(dim=1).mean():.4f} (should be ~1.0)")
