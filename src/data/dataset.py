"""
PyTorch Dataset classes for dog breed classification.
Supports both single-source and combined multi-source datasets.
"""

import os
from pathlib import Path
from typing import Optional, Callable, Tuple, List
from PIL import Image
import torch
from torch.utils.data import Dataset
import pandas as pd


class DogBreedDataset(Dataset):
    """Dataset class for dog breed classification with multi-source support."""
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        transform: Optional[Callable] = None,
        breed_to_idx: Optional[dict] = None,
        use_combined: bool = False
    ):
        """
        Args:
            image_dir: Base directory containing images
                      For combined dataset: parent of kaggle/ and stanford/ folders
                      For single dataset: directory with breed subdirectories
            annotations_file: CSV file with columns:
                            - Single source: ['image_path', 'breed']
                            - Combined: ['image_path', 'breed', 'source']
            transform: Optional transform to apply to images
            breed_to_idx: Optional mapping from breed names to indices
            use_combined: If True, handles multi-source paths (kaggle/stanford prefixes)
        """
        self.image_dir = Path(image_dir)
        self.annotations = pd.read_csv(annotations_file)
        self.transform = transform
        self.use_combined = use_combined
        
        # Create breed to index mapping
        if breed_to_idx is None:
            unique_breeds = sorted(self.annotations['breed'].unique())
            self.breed_to_idx = {breed: idx for idx, breed in enumerate(unique_breeds)}
        else:
            self.breed_to_idx = breed_to_idx
        
        self.idx_to_breed = {idx: breed for breed, idx in self.breed_to_idx.items()}
        self.num_classes = len(self.breed_to_idx)
        
        print(f"Dataset initialized: {len(self)} images, {self.num_classes} classes")
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Returns:
            image: Transformed image tensor
            label: Breed class index
        """
        row = self.annotations.iloc[idx]
        img_rel_path = row['image_path']
        breed = row['breed']
        
        # Handle combined dataset paths
        if self.use_combined and 'source' in row:
            source = row['source']
            if source == 'kaggle':
                # Path format: kaggle/breed_name/image.jpg
                # Actual location: src/data/Kaggle_dataset/train/breed_name/image.jpg
                parts = Path(img_rel_path).parts
                breed_folder = parts[1]
                img_name = parts[2]
                
                # Try train/val/test directories
                for split in ['train', 'val', 'test']:
                    img_path = Path('src/data/Kaggle_dataset') / split / breed_folder / img_name
                    if img_path.exists():
                        break
            else:  # stanford
                # Path format: stanford/n02088094-Afghan_hound/image.jpg
                # Actual location: src/data/Stanford_dataset/images/Images/n02088094-Afghan_hound/image.jpg
                parts = Path(img_rel_path).parts
                breed_folder = parts[1]
                img_name = parts[2]
                img_path = Path('src/data/Stanford_dataset/images/Images') / breed_folder / img_name
        else:
            # Single source dataset
            img_path = self.image_dir / img_rel_path
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (224, 224), color='gray')
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # Get label
        label = self.breed_to_idx[breed]
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute class weights for handling imbalance."""
        breed_counts = self.annotations['breed'].value_counts()
        weights = []
        for breed in sorted(self.breed_to_idx.keys()):
            count = breed_counts.get(breed, 1)
            weights.append(1.0 / count)
        
        weights = torch.tensor(weights, dtype=torch.float32)
        weights = weights / weights.sum() * len(weights)  # Normalize
        return weights
    
    def get_sample_weights(self) -> List[float]:
        """Get per-sample weights for WeightedRandomSampler."""
        breed_counts = self.annotations['breed'].value_counts().to_dict()
        sample_weights = []
        
        for idx in range(len(self.annotations)):
            breed = self.annotations.iloc[idx]['breed']
            weight = 1.0 / breed_counts[breed]
            sample_weights.append(weight)
        
        return sample_weights


class RobustnessTestDataset(Dataset):
    """Dataset for robustness evaluation (blur, lighting, etc.)."""
    
    def __init__(
        self,
        image_dir: str,
        annotations_file: str,
        category: str,
        transform: Optional[Callable] = None,
        breed_to_idx: Optional[dict] = None
    ):
        """
        Args:
            image_dir: Directory containing images
            annotations_file: CSV with columns ['image_path', 'breed', 'category']
            category: Robustness category to filter (e.g., 'blur', 'partial', 'lighting')
            transform: Optional transform to apply
            breed_to_idx: Mapping from breed names to indices
        """
        self.image_dir = image_dir
        df = pd.read_csv(annotations_file)
        self.annotations = df[df['category'] == category].reset_index(drop=True)
        self.transform = transform
        self.breed_to_idx = breed_to_idx
        self.category = category
    
    def __len__(self) -> int:
        return len(self.annotations)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = os.path.join(self.image_dir, self.annotations.iloc[idx]['image_path'])
        breed = self.annotations.iloc[idx]['breed']
        
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        label = self.breed_to_idx[breed] if breed in self.breed_to_idx else -1
        
        return image, label


class OODDataset(Dataset):
    """Dataset for out-of-distribution samples (non-dogs)."""
    
    def __init__(
        self,
        image_dir: str,
        image_list: List[str],
        transform: Optional[Callable] = None
    ):
        """
        Args:
            image_dir: Directory containing OOD images
            image_list: List of image filenames
            transform: Optional transform to apply
        """
        self.image_dir = image_dir
        self.image_list = image_list
        self.transform = transform
    
    def __len__(self) -> int:
        return len(self.image_list)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        img_path = os.path.join(self.image_dir, self.image_list[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image
