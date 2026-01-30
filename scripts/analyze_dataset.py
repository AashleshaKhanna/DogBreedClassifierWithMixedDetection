"""
Analyze the existing Dog Breed Classification dataset.
Dataset location: src/data/Dog Breed Classification/

This script:
1. Analyzes the existing train/val/test splits
2. Counts images per breed
3. Creates CSV annotation files for PyTorch DataLoader
4. Updates config with dataset information
"""

import os
from pathlib import Path
import pandas as pd
import yaml
from collections import Counter
import argparse


def count_images_in_dir(directory: Path):
    """Count image files in a directory."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f'*{ext}')))
    return count


def analyze_split(split_dir: Path, split_name: str):
    """Analyze a single split (train/val/test)."""
    print(f"\n{split_name.upper()} Split:")
    print("-" * 60)
    
    breed_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    breed_counts = {}
    
    for breed_dir in sorted(breed_dirs):
        breed_name = breed_dir.name
        num_images = count_images_in_dir(breed_dir)
        breed_counts[breed_name] = num_images
    
    total_images = sum(breed_counts.values())
    
    print(f"Number of breeds: {len(breed_counts)}")
    print(f"Total images: {total_images}")
    print(f"Images per breed - Min: {min(breed_counts.values())}, "
          f"Max: {max(breed_counts.values())}, "
          f"Mean: {sum(breed_counts.values()) / len(breed_counts):.1f}")
    
    return breed_counts, total_images


def create_annotation_csv(split_dir: Path, split_name: str, output_path: Path):
    """Create CSV annotation file for a split."""
    data = []
    
    breed_dirs = [d for d in split_dir.iterdir() if d.is_dir()]
    
    for breed_dir in sorted(breed_dirs):
        breed_name = breed_dir.name
        
        # Get all image files
        extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
        image_files = []
        for ext in extensions:
            image_files.extend(breed_dir.glob(f'*{ext}'))
        
        for img_path in image_files:
            # Store relative path from split_dir
            rel_path = img_path.relative_to(split_dir)
            data.append({
                'image_path': str(rel_path),
                'breed': breed_name
            })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    
    print(f"  Created: {output_path} ({len(df)} images)")
    
    return df


def main():
    parser = argparse.ArgumentParser(description='Analyze Dog Breed Classification dataset')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='src/data/Dog Breed Classification',
        help='Path to dataset directory'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/processed',
        help='Output directory for annotation files'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/config.yaml',
        help='Path to config file to update'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Dog Breed Classification Dataset Analysis")
    print("="*60)
    print(f"Dataset location: {data_dir}")
    
    # Check if dataset exists
    if not data_dir.exists():
        print(f"\nError: Dataset directory not found: {data_dir}")
        return
    
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    # Analyze each split
    train_counts, train_total = analyze_split(train_dir, 'train')
    val_counts, val_total = analyze_split(val_dir, 'val')
    test_counts, test_total = analyze_split(test_dir, 'test')
    
    # Overall statistics
    print("\n" + "="*60)
    print("OVERALL STATISTICS")
    print("="*60)
    print(f"Total breeds: {len(train_counts)}")
    print(f"Total images: {train_total + val_total + test_total}")
    print(f"  Train: {train_total} ({train_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"  Val:   {val_total} ({val_total/(train_total+val_total+test_total)*100:.1f}%)")
    print(f"  Test:  {test_total} ({test_total/(train_total+val_total+test_total)*100:.1f}%)")
    
    # Create annotation CSV files
    print("\n" + "="*60)
    print("Creating Annotation Files")
    print("="*60)
    
    train_csv = output_dir / 'train.csv'
    val_csv = output_dir / 'val.csv'
    test_csv = output_dir / 'test.csv'
    
    train_df = create_annotation_csv(train_dir, 'train', train_csv)
    val_df = create_annotation_csv(val_dir, 'val', val_csv)
    test_df = create_annotation_csv(test_dir, 'test', test_csv)
    
    # Save breed list
    breed_names = sorted(train_counts.keys())
    breed_list_path = output_dir / 'breed_names.txt'
    with open(breed_list_path, 'w') as f:
        for breed in breed_names:
            f.write(f"{breed}\n")
    print(f"  Created: {breed_list_path}")
    
    # Save breed to index mapping
    breed_to_idx = {breed: idx for idx, breed in enumerate(breed_names)}
    breed_mapping_path = output_dir / 'breed_mapping.txt'
    with open(breed_mapping_path, 'w') as f:
        f.write("index,breed_name\n")
        for breed, idx in sorted(breed_to_idx.items(), key=lambda x: x[1]):
            f.write(f"{idx},{breed}\n")
    print(f"  Created: {breed_mapping_path}")
    
    # Save detailed statistics
    stats_path = output_dir / 'dataset_stats.txt'
    with open(stats_path, 'w') as f:
        f.write("Dog Breed Classification Dataset Statistics\n")
        f.write("="*60 + "\n\n")
        f.write(f"Dataset Location: {data_dir}\n")
        f.write(f"Total Breeds: {len(breed_names)}\n")
        f.write(f"Total Images: {train_total + val_total + test_total}\n\n")
        
        f.write("Split Distribution:\n")
        f.write(f"  Train: {train_total} images\n")
        f.write(f"  Val:   {val_total} images\n")
        f.write(f"  Test:  {test_total} images\n\n")
        
        f.write("Images per Breed:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Breed':<40} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8}\n")
        f.write("-"*60 + "\n")
        
        for breed in breed_names:
            train_count = train_counts.get(breed, 0)
            val_count = val_counts.get(breed, 0)
            test_count = test_counts.get(breed, 0)
            total = train_count + val_count + test_count
            f.write(f"{breed:<40} {train_count:<8} {val_count:<8} {test_count:<8} {total:<8}\n")
    
    print(f"  Created: {stats_path}")
    
    # Update config file
    config_path = Path(args.config)
    if config_path.exists():
        print("\n" + "="*60)
        print("Updating Configuration")
        print("="*60)
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Update model config
        config['model']['num_classes'] = len(breed_names)
        
        # Update data paths
        config['data']['data_dir'] = str(train_dir.parent)
        config['data']['train_annotations'] = str(train_csv)
        config['data']['val_annotations'] = str(val_csv)
        config['data']['test_annotations'] = str(test_csv)
        
        # Backup original config
        backup_path = config_path.with_suffix('.yaml.backup')
        if not backup_path.exists():
            import shutil
            shutil.copy(config_path, backup_path)
            print(f"  Backup created: {backup_path}")
        
        # Save updated config
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        
        print(f"  Updated: {config_path}")
        print(f"    - num_classes: {len(breed_names)}")
        print(f"    - data_dir: {train_dir.parent}")
    
    print("\n" + "="*60)
    print("Dataset Analysis Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review dataset statistics:", stats_path)
    print("2. Check breed list:", breed_list_path)
    print("3. Verify config:", config_path)
    print("4. Start training with: python scripts/train_phase1.py")


if __name__ == '__main__':
    main()
