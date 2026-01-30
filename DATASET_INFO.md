# Dog Breed Classification Dataset

## Dataset Overview

**Source**: Kaggle Dog Breed Classification Dataset  
**Location**: `src/data/Dog Breed Classification/`  
**Total Breeds**: 93  
**Total Images**: 16,080

## Dataset Statistics

### Split Distribution
- **Train**: 12,782 images (79.5%)
- **Validation**: 1,524 images (9.5%)
- **Test**: 1,774 images (11.0%)

### Images per Breed
- Average: ~173 images per breed
- All breeds have sufficient samples (>100 images)

## Dataset Structure

```
src/data/Dog Breed Classification/
├── train/
│   ├── afghan_hound/
│   ├── african_hunting_dog/
│   ├── airedale/
│   └── ... (93 breeds total)
├── val/
│   └── ... (same 93 breeds)
└── test/
    └── ... (same 93 breeds)
```

## Annotation Files

CSV files have been created for easy loading with PyTorch:

- `data/processed/train.csv` - Training set annotations
- `data/processed/val.csv` - Validation set annotations  
- `data/processed/test.csv` - Test set annotations

**CSV Format**:
```csv
image_path,breed
afghan_hound/afghan_hound0.jpg,afghan_hound
```

The `image_path` is relative to the split directory (train/val/test).

## Breed List

93 dog breeds including:
- afghan_hound
- african_hunting_dog
- airedale
- basenji
- beagle
- border_collie
- german_shepherd
- golden_retriever
- labrador_retriever
- rottweiler
- ... and 83 more

Full list available in: `data/processed/breed_names.txt`

## Breed to Index Mapping

Mapping file: `data/processed/breed_mapping.txt`

Format:
```
index,breed_name
0,afghan_hound
1,african_hunting_dog
...
```

## Usage with PyTorch

```python
from src.data.dataset import DogBreedDataset
from src.data.augmentation import get_train_transforms

# Load dataset
dataset = DogBreedDataset(
    image_dir='src/data/Dog Breed Classification/train',
    annotations_file='data/processed/train.csv',
    transform=get_train_transforms()
)

# Create DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=32, shuffle=True)
```

## Configuration

The config file has been updated with:
- `num_classes: 93`
- `data_dir: src/data/Dog Breed Classification/train`
- Annotation file paths

## Notes

- Dataset is already split into train/val/test
- All images are in JPG format
- Images vary in size and aspect ratio
- Some breeds may be visually similar (e.g., different terrier types)
- Dataset includes some wild canids (dhole, dingo, african_hunting_dog)

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Explore data: Check `notebooks/01_data_exploration.ipynb`
3. Start training: `python scripts/train_phase1.py`
