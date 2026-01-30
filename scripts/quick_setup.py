"""
Quick setup script to analyze dataset and create annotation files.
No external dependencies required (uses only Python standard library).
"""

import os
from pathlib import Path
import json


def count_images_in_dir(directory):
    """Count image files in a directory."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    count = 0
    for ext in extensions:
        count += len(list(directory.glob(f'*{ext}')))
    return count


def get_image_files(directory):
    """Get all image files in a directory."""
    extensions = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(directory.glob(f'*{ext}'))
    return image_files


def analyze_and_create_csv(data_dir, output_dir):
    """Analyze dataset and create CSV files."""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("Dog Breed Classification Dataset Setup")
    print("="*60)
    print(f"Dataset location: {data_dir}\n")
    
    # Check splits
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'
    
    if not train_dir.exists():
        print(f"Error: Train directory not found: {train_dir}")
        return
    
    # Process each split
    for split_name, split_dir in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        print(f"\nProcessing {split_name.upper()} split...")
        
        breed_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        
        # Create CSV content
        csv_lines = ['image_path,breed\n']
        total_images = 0
        
        for breed_dir in breed_dirs:
            breed_name = breed_dir.name
            image_files = get_image_files(breed_dir)
            
            for img_path in image_files:
                rel_path = img_path.relative_to(split_dir)
                # Use forward slashes for cross-platform compatibility
                rel_path_str = str(rel_path).replace('\\', '/')
                csv_lines.append(f"{rel_path_str},{breed_name}\n")
                total_images += 1
        
        # Write CSV
        csv_path = output_dir / f'{split_name}.csv'
        with open(csv_path, 'w') as f:
            f.writelines(csv_lines)
        
        print(f"  Created: {csv_path}")
        print(f"  Images: {total_images}")
        print(f"  Breeds: {len(breed_dirs)}")
    
    # Create breed list
    breed_dirs = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    breed_list_path = output_dir / 'breed_names.txt'
    with open(breed_list_path, 'w') as f:
        for breed in breed_dirs:
            f.write(f"{breed}\n")
    
    print(f"\nCreated breed list: {breed_list_path}")
    print(f"Total breeds: {len(breed_dirs)}")
    
    # Create breed mapping
    breed_mapping_path = output_dir / 'breed_mapping.txt'
    with open(breed_mapping_path, 'w') as f:
        f.write("index,breed_name\n")
        for idx, breed in enumerate(breed_dirs):
            f.write(f"{idx},{breed}\n")
    
    print(f"Created breed mapping: {breed_mapping_path}")
    
    # Update config
    config_path = Path('configs/config.yaml')
    if config_path.exists():
        print(f"\nUpdating config: {config_path}")
        
        # Read config
        with open(config_path, 'r') as f:
            lines = f.readlines()
        
        # Update num_classes
        new_lines = []
        for line in lines:
            if 'num_classes:' in line and 'model:' not in line:
                new_lines.append(f"  num_classes: {len(breed_dirs)}  # Updated by quick_setup.py\n")
            elif 'data_dir:' in line and 'data:' not in line:
                new_lines.append(f"  data_dir: {str(data_dir)}\n")
            elif 'train_annotations:' in line:
                new_lines.append(f"  train_annotations: {str(output_dir / 'train.csv')}\n")
            elif 'val_annotations:' in line:
                new_lines.append(f"  val_annotations: {str(output_dir / 'val.csv')}\n")
            elif 'test_annotations:' in line:
                new_lines.append(f"  test_annotations: {str(output_dir / 'test.csv')}\n")
            else:
                new_lines.append(line)
        
        # Write updated config
        with open(config_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"  Updated num_classes to: {len(breed_dirs)}")
    
    print("\n" + "="*60)
    print("Setup Complete!")
    print("="*60)
    print("\nFiles created:")
    print(f"  - {output_dir / 'train.csv'}")
    print(f"  - {output_dir / 'val.csv'}")
    print(f"  - {output_dir / 'test.csv'}")
    print(f"  - {output_dir / 'breed_names.txt'}")
    print(f"  - {output_dir / 'breed_mapping.txt'}")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Start training: python scripts/train_phase1.py")


if __name__ == '__main__':
    data_dir = 'src/data/Dog Breed Classification'
    output_dir = 'data/processed'
    
    analyze_and_create_csv(data_dir, output_dir)
