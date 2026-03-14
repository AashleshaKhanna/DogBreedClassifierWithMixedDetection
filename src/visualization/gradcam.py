"""
Grad-CAM implementation for model interpretability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from typing import Optional, Tuple


class GradCAM:
    """Gradient-weighted Class Activation Mapping."""
    
    def __init__(self, model: nn.Module, target_layer: str):
        """
        Initialize Grad-CAM.
        
        Args:
            model: Classification model
            target_layer: Name of target layer for CAM
        """
        self.model = model
        self.model.eval()
        
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        for name, module in self.model.named_modules():
            if name == self.target_layer:
                module.register_forward_hook(forward_hook)
                module.register_full_backward_hook(backward_hook)
                break
    
    def generate(
        self,
        input_image: torch.Tensor,
        target_class: Optional[int] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.
        
        Args:
            input_image: Input tensor (1, 3, H, W)
            target_class: Target class index (None = predicted class)
        
        Returns:
            heatmap: Grad-CAM heatmap (H, W)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(input_image)
        
        # Get target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)
        
        # Compute weights (global average pooling of gradients)
        weights = gradients.mean(dim=(1, 2))  # (C,)
        
        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # Apply ReLU
        cam = F.relu(cam)
        
        # Normalize
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam
    
    def visualize(
        self,
        input_image: torch.Tensor,
        original_image: Image.Image,
        target_class: Optional[int] = None,
        alpha: float = 0.5
    ) -> Image.Image:
        """
        Generate Grad-CAM visualization overlaid on original image.
        
        Args:
            input_image: Preprocessed input tensor (1, 3, H, W)
            original_image: Original PIL Image
            target_class: Target class (None = predicted)
            alpha: Overlay transparency
        
        Returns:
            Visualization as PIL Image
        """
        # Generate heatmap
        heatmap = self.generate(input_image, target_class)
        
        # Resize heatmap to match original image
        heatmap_resized = Image.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = heatmap_resized.resize(
            original_image.size,
            Image.BILINEAR
        )
        heatmap_resized = np.array(heatmap_resized) / 255.0
        
        # Apply colormap
        cmap = plt.get_cmap('jet')
        heatmap_colored = cmap(heatmap_resized)[:, :, :3]  # RGB
        heatmap_colored = (heatmap_colored * 255).astype(np.uint8)
        
        # Convert original image to array
        original_array = np.array(original_image.convert('RGB'))
        
        # Overlay
        overlay = (alpha * heatmap_colored + (1 - alpha) * original_array).astype(np.uint8)
        
        return Image.fromarray(overlay)


def visualize_gradcam_grid(
    model: nn.Module,
    images: list,
    preprocessed_images: torch.Tensor,
    class_names: list,
    target_layer: str,
    save_path: Optional[str] = None
):
    """
    Create grid of Grad-CAM visualizations.
    
    Args:
        model: Classification model
        images: List of original PIL Images
        preprocessed_images: Preprocessed tensor (B, 3, H, W)
        class_names: List of class names
        target_layer: Target layer for Grad-CAM
        save_path: Optional path to save figure
    """
    gradcam = GradCAM(model, target_layer)
    
    n_images = len(images)
    fig, axes = plt.subplots(2, n_images, figsize=(4 * n_images, 8))
    
    if n_images == 1:
        axes = axes.reshape(2, 1)
    
    for i, (img, prep_img) in enumerate(zip(images, preprocessed_images)):
        # Get prediction
        with torch.no_grad():
            output = model(prep_img.unsqueeze(0))
            pred_class = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1)[0, pred_class].item()
        
        # Original image
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original\n{class_names[pred_class]}\nConf: {confidence:.3f}')
        axes[0, i].axis('off')
        
        # Grad-CAM
        cam_viz = gradcam.visualize(prep_img.unsqueeze(0), img, pred_class)
        axes[1, i].imshow(cam_viz)
        axes[1, i].set_title('Grad-CAM')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()


if __name__ == '__main__':
    print("Grad-CAM module loaded successfully")
