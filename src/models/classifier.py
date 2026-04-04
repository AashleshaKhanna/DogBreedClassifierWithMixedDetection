"""
Dog breed classification model using EfficientNet-B3.
Stage 2 of the pipeline: Classify dog breed from detected region.
"""

import torch
import torch.nn as nn
import timm
from typing import Optional


class BreedClassifier(nn.Module):
    """EfficientNet-B3 based breed classifier."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        dropout_rates: tuple = (0.3, 0.2)
    ):
        """
        Initialize breed classifier.
        
        Args:
            num_classes: Number of dog breeds
            pretrained: Use ImageNet pretrained weights
            dropout_rates: Dropout rates for classifier head (2 layers)
        """
        super().__init__()
        
        self.num_classes = num_classes
        
        # Load EfficientNet-B3 backbone
        try:
            self.backbone = timm.create_model(
                'efficientnet_b3',
                pretrained=pretrained,
                num_classes=0,  # Remove classification head
                global_pool=''  # Remove global pooling
            )
            if pretrained:
                print("Loaded pretrained EfficientNet-B3 weights")
        except Exception as e:
            print(f"Warning: Could not load pretrained weights ({e})")
            print("Training from scratch (no pretrained weights)")
            self.backbone = timm.create_model(
                'efficientnet_b3',
                pretrained=False,
                num_classes=0,
                global_pool=''
            )
        
        # Get feature dimension
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy_input)
            feature_dim = features.shape[1]
        
        # Custom classification head
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout_rates[0]),
            nn.Linear(feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rates[1]),
            nn.Linear(512, num_classes)
        )
        
        # Initialize classifier weights
        self._init_classifier()
    
    def _init_classifier(self):
        """Initialize classifier head weights."""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
        
        Returns:
            logits: Class logits (B, num_classes)
        """
        # Extract features
        features = self.backbone(x)
        
        # Global pooling
        pooled = self.global_pool(features)
        
        # Classification
        logits = self.classifier(pooled)
        
        return logits
    
    def freeze_backbone(self):
        """Freeze backbone for transfer learning."""
        for param in self.backbone.parameters():
            param.requires_grad = False
    
    def unfreeze_backbone(self, num_blocks: Optional[int] = None):
        """
        Unfreeze backbone for fine-tuning.
        
        Args:
            num_blocks: Number of blocks to unfreeze from end (None = all)
        """
        if num_blocks is None:
            # Unfreeze all
            for param in self.backbone.parameters():
                param.requires_grad = True
        else:
            # Unfreeze last N blocks
            # EfficientNet has blocks named 'blocks.X'
            blocks = [name for name, _ in self.backbone.named_children() if 'blocks' in name]
            
            for name, module in self.backbone.named_children():
                if 'blocks' in name:
                    # Get block index
                    try:
                        block_idx = int(name.split('.')[-1])
                        total_blocks = len(blocks)
                        if block_idx >= total_blocks - num_blocks:
                            for param in module.parameters():
                                param.requires_grad = True
                    except:
                        pass
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features before classification.
        
        Args:
            x: Input tensor (B, 3, 224, 224)
        
        Returns:
            features: Feature tensor (B, feature_dim)
        """
        features = self.backbone(x)
        pooled = self.global_pool(features)
        return pooled.flatten(1)


class LabelSmoothingCrossEntropy(nn.Module):
    """Cross entropy loss with label smoothing."""
    
    def __init__(self, epsilon: float = 0.1):
        """
        Args:
            epsilon: Smoothing parameter (0 = no smoothing)
        """
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute label smoothing loss.
        
        Args:
            logits: Model predictions (B, num_classes)
            targets: Ground truth labels (B,)
        
        Returns:
            loss: Scalar loss value
        """
        num_classes = logits.size(-1)
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        
        # One-hot encode targets
        targets_one_hot = torch.zeros_like(log_probs).scatter_(
            1, targets.unsqueeze(1), 1
        )
        
        # Apply label smoothing
        targets_smooth = (1 - self.epsilon) * targets_one_hot + \
                        self.epsilon / num_classes
        
        # Compute loss
        loss = -(targets_smooth * log_probs).sum(dim=-1).mean()
        
        return loss


def create_model(
    num_classes: int,
    pretrained: bool = True,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
) -> BreedClassifier:
    """
    Create and initialize breed classifier.
    
    Args:
        num_classes: Number of dog breeds
        pretrained: Use pretrained weights
        device: Device to place model on
    
    Returns:
        model: Initialized BreedClassifier
    """
    model = BreedClassifier(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)
    return model


def test_model():
    """Test model forward pass."""
    model = create_model(num_classes=50)
    
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


if __name__ == '__main__':
    test_model()
