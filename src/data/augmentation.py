"""
Data augmentation pipeline for dog breed classification.
"""

from torchvision import transforms


def get_train_transforms(image_size: int = 224):
    """
    Training augmentation pipeline.
    Simulates real-world photo conditions: lighting, angles, partial occlusion.
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=0, scale=(0.9, 1.1), translate=(0.1, 0.1)),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.15)),
    ])


def get_val_transforms(image_size: int = 224):
    """
    Validation/test transform pipeline (no augmentation).
    """
    return transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def get_inference_transforms(image_size: int = 224):
    """Alias for val transforms — used during inference."""
    return get_val_transforms(image_size)
