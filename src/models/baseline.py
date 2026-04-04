"""
Baseline model: ResNet-50 for dog breed classification.
Simple single-stage classifier for comparison with multi-stage pipeline.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights


class BaselineResNet50(nn.Module):
    """
    Baseline ResNet-50 classifier.
    Standard transfer learning approach without robustness enhancements.
    """
    
    def __init__(self, num_classes: int, pretrained: bool = True):
        """
        Initialize baseline ResNet-50.
        
        Args:
            num_classes: Number of dog breeds
            pretrained: Use ImageNet pretrained weights
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load pretrained ResNet-50 (using new weights API)
        if pretrained:
            try:
                weights = ResNet50_Weights.IMAGENET1K_V1
                self.resnet = models.resnet50(weights=weights)
                print("Loaded pretrained ResNet-50 weights")
            except Exception as e:
                print(f"Warning: Could not load pretrained weights ({e})")
                print("Training from scratch (no pretrained weights)")
                self.resnet = models.resnet50(weights=None)
        else:
            self.resnet = models.resnet50(weights=None)
        
        # Replace final layer
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)
        
        # Initialize new layer
        nn.init.kaiming_normal_(self.resnet.fc.weight, mode='fan_out', nonlinearity='relu')
        nn.init.constant_(self.resnet.fc.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
        
        Returns:
            logits: Class logits (B, num_classes)
        """
        return self.resnet(x)
    
    def freeze_backbone(self):
        """Freeze all layers except final classifier."""
        for param in self.resnet.parameters():
            param.requires_grad = False
        
        # Unfreeze final layer
        for param in self.resnet.fc.parameters():
            param.requires_grad = True
    
    def unfreeze_all(self):
        """Unfreeze all layers for fine-tuning."""
        for param in self.resnet.parameters():
            param.requires_grad = True


def create_baseline_model(
    num_classes: int,
    pretrained: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BaselineResNet50:
    """
    Create and initialize baseline model.
    
    Args:
        num_classes: Number of dog breeds
        pretrained: Use pretrained weights
        device: Device to place model on
    
    Returns:
        model: Initialized BaselineResNet50
    """
    model = BaselineResNet50(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


if __name__ == '__main__':
    # Test baseline model
    model = create_baseline_model(num_classes=93)
    
    # Test input
    x = torch.randn(4, 3, 224, 224)
    if torch.cuda.is_available():
        x = x.cuda()
    
    # Forward pass
    with torch.no_grad():
        logits = model(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {logits.shape}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print("Baseline model test passed!")
