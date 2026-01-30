"""
Combine Kaggle and Stanford dog breed datasets.
This satisfies AC-1.2: "Training dataset must combine at least 2 different data sources"

Strategy:
1. Analyze both datasets and find overlapping breeds
2. Normalize breed names (Stanford uses ImageNet IDs, Kaggle uses clean names)
3. Combine images from both sources for overlapping breeds
4. Create unified train/val/test splits
5. Generate annotation files for the combined dataset
"""

import os
from pathlib import Path
import re
from collections import defaultdict


def normalize_breed_name(name):
    """
    Normalize breed names to match between datasets.
    Stanford format: n02088094-Afghan_hound
    Kaggle format: afghan_hound
    """
    # Remove ImageNet ID prefix if present
    if name.startswith('n'):
        name = re.sub(r'^n\d+-', '', name)
    
    # Convert to lowercase and replace spaces/hyphens with underscores
    name = name.lower()
    name = name.replace(' ', '_').replace('-', '_')
    
    # Handle special cases
    name_mapping = {
        'entlebucher': 'entlebucher',
        'great_dane': 'great_dane',
        'saint_bernard': 'saint_bernard',
        'mexican_hairless': 'mexican_hairless',
        'african_hunting_dog': 'african_hunting_dog',
        'boston_bull': 'boston_bull',
        'brabancon_griffon': 'brabancon_griffon',
    }
    
    return name_mapping.get(name, name)


def get_image_files(directory):
    """Get all image files in a directory."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
    return image_files


def analyze_datasets(kaggle_dir, stanford_dir):
    """Analyze both datasets and find overlapping breeds."""
    
    print("="*60)
    print("Analyzing Datasets")
    print("="*60)
    
    # Kaggle dataset
    kaggle_train = kaggle_dir / 'train'
    kaggle_breeds = {}
    
    print("\nKaggle Dataset:")
    for breed_dir in sorted(kaggle_train.iterdir()):
        if breed_dir.is_dir():
            breed_name = normalize_breed_name(breed_dir.name)
            num_images = len(get_image_files(breed_dir))
            kaggle_breeds[breed_name] = {
                'original_name': breed_dir.name,
                'train_count': num_images,
                'train_dir': breed_dir
            }
    
    print(f"  Breeds: {len(kaggle_breeds)}")
    print(f"  Total train images: {sum(b['train_count'] for b in kaggle_breeds.values())}")
    
    # Add val and test counts
    kaggle_val = kaggle_dir / 'val'
    kaggle_test = kaggle_dir / 'test'
    
    for breed_dir in kaggle_val.iterdir():
        if breed_dir.is_dir():
            breed_name = normalize_breed_name(breed_dir.name)
            if breed_name in kaggle_breeds:
                kaggle_breeds[breed_name]['val_count'] = len(get_image_files(breed_dir))
                kaggle_breeds[breed_name]['val_dir'] = breed_dir
    
    for breed_dir in kaggle_test.iterdir():
        if breed_dir.is_dir():
            breed_name = normalize_breed_name(breed_dir.name)
            if breed_name in kaggle_breeds:
                kaggle_breeds[breed_name]['test_count'] = len(get_image_files(breed_dir))
                kaggle_breeds[breed_name]['test_dir'] = breed_dir
    
    # Stanford dataset
    stanford_images = stanford_dir / 'images' / 'Images'
    stanford_breeds = {}
    
    print("\nStanford Dataset:")
    for breed_dir in sorted(stanford_images.iterdir()):
        if breed_dir.is_dir():
            breed_name = normalize_breed_name(breed_dir.name)
            num_images = len(get_image_files(breed_dir))
            stanford_breeds[breed_name] = {
                'original_name': breed_dir.name,
                'count': num_images,
                'dir': breed_dir
            }
    
    print(f"  Breeds: {len(stanford_breeds)}")
    print(f"  Total images: {sum(b['count'] for b in stanford_breeds.values())}")
    
    # Find overlapping breeds
    kaggle_set = set(kaggle_breeds.keys())
    stanford_set = set(stanford_breeds.keys())
    overlapping = kaggle_set & stanford_set
    
    print(f"\nOverlapping Breeds: {len(overlapping)}")
    print(f"Kaggle only: {len(kaggle_set - stanford_set)}")
    print(f"Stanford only: {len(stanford_set - kaggle_set)}")
    
    return kaggle_breeds, stanford_breeds, overlapping


def create_combined_annotations(kaggle_breeds, stanford_breeds, overlapping, output_dir):
    """Create combined annotation files."""
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*60)
    print("Creating Combined Annotations")
    print("="*60)
    
    # We'll use Kaggle's existing splits and augment with Stanford data
    # Strategy: Add Stanford images to training set for overlapping breeds
    
    train_lines = ['image_path,breed,source\n']
    val_lines = ['image_path,breed,source\n']
    test_lines = ['image_path,breed,source\n']
    
    all_breeds = set()
    
    # Add Kaggle data (keep existing splits)
    print("\nAdding Kaggle dataset...")
    for breed_name, breed_info in kaggle_breeds.items():
        all_breeds.add(breed_name)
        
        # Train
        if 'train_dir' in breed_info:
            for img_path in get_image_files(breed_info['train_dir']):
                rel_path = f"kaggle/{breed_info['original_name']}/{img_path.name}"
                train_lines.append(f"{rel_path},{breed_name},kaggle\n")
        
        # Val
        if 'val_dir' in breed_info:
            for img_path in get_image_files(breed_info['val_dir']):
                rel_path = f"kaggle/{breed_info['original_name']}/{img_path.name}"
                val_lines.append(f"{rel_path},{breed_name},kaggle\n")
        
        # Test
        if 'test_dir' in breed_info:
            for img_path in get_image_files(breed_info['test_dir']):
                rel_path = f"kaggle/{breed_info['original_name']}/{img_path.name}"
                test_lines.append(f"{rel_path},{breed_name},kaggle\n")
    
    # Add Stanford data for overlapping breeds (to training set)
    print("Adding Stanford dataset (overlapping breeds to training)...")
    stanford_added = 0
    for breed_name in overlapping:
        if breed_name in stanford_breeds:
            all_breeds.add(breed_name)
            breed_info = stanford_breeds[breed_name]
            
            for img_path in get_image_files(breed_info['dir']):
                rel_path = f"stanford/{breed_info['original_name']}/{img_path.name}"
                train_lines.append(f"{rel_path},{breed_name},stanford\n")
                stanford_added += 1
    
    print(f"  Added {stanford_added} Stanford images to training set")
    
    # Write CSV files
    train_csv = output_dir / 'train_combined.csv'
    val_csv = output_dir / 'val_combined.csv'
    test_csv = output_dir / 'test_combined.csv'
    
    with open(train_csv, 'w') as f:
        f.writelines(train_lines)
    
    with open(val_csv, 'w') as f:
        f.writelines(val_lines)
    
    with open(test_csv, 'w') as f:
        f.writelines(test_lines)
    
    print(f"\nCreated annotation files:")
    print(f"  Train: {train_csv} ({len(train_lines)-1} images)")
    print(f"  Val:   {val_csv} ({len(val_lines)-1} images)")
    print(f"  Test:  {test_csv} ({len(test_lines)-1} images)")
    
    # Save breed list
    breed_list = sorted(all_breeds)
    breed_list_path = output_dir / 'breed_names_combined.txt'
    with open(breed_list_path, 'w') as f:
        for breed in breed_list:
            f.write(f"{breed}\n")
    
    print(f"  Breeds: {breed_list_path} ({len(breed_list)} breeds)")
    
    # Save breed mapping
    breed_mapping_path = output_dir / 'breed_mapping_combined.txt'
    with open(breed_mapping_path, 'w') as f:
        f.write("index,breed_name\n")
        for idx, breed in enumerate(breed_list):
            f.write(f"{idx},{breed}\n")
    
    print(f"  Mapping: {breed_mapping_path}")
    
    # Save statistics
    stats_path = output_dir / 'combined_dataset_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("Combined Dataset Statistics\n")
        f.write("="*60 + "\n\n")
        f.write("Data Sources:\n")
        f.write("  1. Kaggle Dog Breed Classification\n")
        f.write("  2. Stanford Dogs Dataset\n\n")
        f.write(f"Total Breeds: {len(breed_list)}\n")
        f.write(f"Overlapping Breeds: {len(overlapping)}\n")
        f.write(f"Total Images: {len(train_lines) + len(val_lines) + len(test_lines) - 3}\n")
        f.write(f"  Train: {len(train_lines)-1}\n")
        f.write(f"  Val:   {len(val_lines)-1}\n")
        f.write(f"  Test:  {len(test_lines)-1}\n\n")
        f.write("Overlapping Breeds (with Stanford augmentation):\n")
        for breed in sorted(overlapping):
            f.write(f"  - {breed}\n")
    
    print(f"  Stats: {stats_path}")
    
    return len(breed_list)


def update_config(config_path, num_breeds):
    """Update config file with combined dataset info."""
    
    config_path = Path(config_path)
    if not config_path.exists():
        print(f"\nWarning: Config file not found: {config_path}")
        return
    
    print("\n" + "="*60)
    print("Updating Configuration")
    print("="*60)
    
    with open(config_path, 'r') as f:
        lines = f.readlines()
    
    new_lines = []
    for line in lines:
        if 'num_classes:' in line and 'model:' not in line:
            new_lines.append(f"  num_classes: {num_breeds}  # Combined dataset\n")
        elif 'train_annotations:' in line:
            new_lines.append(f"  train_annotations: data/processed/train_combined.csv\n")
        elif 'val_annotations:' in line:
            new_lines.append(f"  val_annotations: data/processed/val_combined.csv\n")
        elif 'test_annotations:' in line:
            new_lines.append(f"  test_annotations: data/processed/test_combined.csv\n")
        else:
            new_lines.append(line)
    
    # Backup
    backup_path = config_path.with_suffix('.yaml.backup2')
    if not backup_path.exists():
        import shutil
        shutil.copy(config_path, backup_path)
        print(f"  Backup: {backup_path}")
    
    with open(config_path, 'w') as f:
        f.writelines(new_lines)
    
    print(f"  Updated: {config_path}")
    print(f"    - num_classes: {num_breeds}")


def main():
    kaggle_dir = Path('src/data/Kaggle_dataset')
    stanford_dir = Path('src/data/Stanford_dataset')
    output_dir = Path('data/processed')
    config_path = Path('configs/config.yaml')
    
    print("="*60)
    print("Combining Kaggle and Stanford Datasets")
    print("="*60)
    print(f"Kaggle:   {kaggle_dir}")
    print(f"Stanford: {stanford_dir}")
    print(f"Output:   {output_dir}")
    
    # Check if datasets exist
    if not kaggle_dir.exists():
        print(f"\nError: Kaggle dataset not found: {kaggle_dir}")
        return
    
    if not stanford_dir.exists():
        print(f"\nError: Stanford dataset not found: {stanford_dir}")
        return
    
    # Analyze datasets
    kaggle_breeds, stanford_breeds, overlapping = analyze_datasets(kaggle_dir, stanford_dir)
    
    # Create combined annotations
    num_breeds = create_combined_annotations(kaggle_breeds, stanford_breeds, overlapping, output_dir)
    
    # Update config
    update_config(config_path, num_breeds)
    
    print("\n" + "="*60)
    print("Dataset Combination Complete!")
    print("="*60)
    print("\nKey Achievement:")
    print("✓ AC-1.2: Training dataset combines 2 different data sources")
    print("  - Kaggle Dog Breed Classification")
    print("  - Stanford Dogs Dataset")
    print(f"  - {len(overlapping)} overlapping breeds augmented with Stanford data")
    print("\nNext steps:")
    print("1. Review: data/processed/combined_dataset_stats.txt")
    print("2. Update Dataset class to handle 'source' column")
    print("3. Start training with combined dataset")


if __name__ == '__main__':
    main()
