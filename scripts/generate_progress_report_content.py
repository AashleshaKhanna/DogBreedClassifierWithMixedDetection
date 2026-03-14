"""
Generate content for progress report:
- Data statistics and visualizations
- Sample images from dataset
- Model architecture diagrams
- Placeholder results (to be replaced with actual training results)
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from PIL import Image
import random


def generate_data_statistics():
    """Generate data statistics for progress report."""
    
    print("="*60)
    print("Generating Data Statistics for Progress Report")
    print("="*60)
    
    # Load annotations
    train_df = pd.read_csv('data/processed/train_combined.csv')
    val_df = pd.read_csv('data/processed/val_combined.csv')
    test_df = pd.read_csv('data/processed/test_combined.csv')
    
    # Overall statistics
    print("\n1. Dataset Overview:")
    print(f"   Total images: {len(train_df) + len(val_df) + len(test_df)}")
    print(f"   Training: {len(train_df)} ({len(train_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"   Validation: {len(val_df)} ({len(val_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"   Test: {len(test_df)} ({len(test_df)/(len(train_df)+len(val_df)+len(test_df))*100:.1f}%)")
    print(f"   Number of breeds: {train_df['breed'].nunique()}")
    
    # Source distribution
    print("\n2. Data Source Distribution:")
    source_counts = train_df['source'].value_counts()
    for source, count in source_counts.items():
        print(f"   {source.capitalize()}: {count} images ({count/len(train_df)*100:.1f}%)")
    
    # Class distribution
    print("\n3. Top 10 Breeds by Image Count:")
    breed_counts = train_df['breed'].value_counts().head(10)
    for breed, count in breed_counts.items():
        print(f"   {breed}: {count} images")
    
    # Create visualizations
    output_dir = Path('progress_report_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Figure 1: Dataset split distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Split sizes
    splits = ['Train', 'Val', 'Test']
    sizes = [len(train_df), len(val_df), len(test_df)]
    colors = ['#4CAF50', '#2196F3', '#FF9800']
    
    axes[0].bar(splits, sizes, color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Number of Images', fontsize=12, fontweight='bold')
    axes[0].set_title('Dataset Split Distribution', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    for i, (split, size) in enumerate(zip(splits, sizes)):
        axes[0].text(i, size + 500, f'{size:,}', ha='center', fontweight='bold')
    
    # Source distribution
    source_data = train_df['source'].value_counts()
    axes[1].pie(source_data.values, labels=[s.capitalize() for s in source_data.index], 
                autopct='%1.1f%%', colors=['#FF6B6B', '#4ECDC4'], startangle=90,
                textprops={'fontsize': 11, 'fontweight': 'bold'})
    axes[1].set_title('Training Data Sources', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'dataset_statistics.png'}")
    
    # Figure 2: Class distribution
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    
    breed_counts_all = train_df['breed'].value_counts()
    ax.bar(range(len(breed_counts_all)), breed_counts_all.values, 
           color='#3F51B5', edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Breed Index (sorted by frequency)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Training Images', fontsize=12, fontweight='bold')
    ax.set_title('Training Images per Breed (93 breeds)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add statistics
    ax.axhline(y=breed_counts_all.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {breed_counts_all.mean():.0f}')
    ax.axhline(y=breed_counts_all.median(), color='green', linestyle='--', linewidth=2, label=f'Median: {breed_counts_all.median():.0f}')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'class_distribution.png'}")
    
    plt.close('all')
    
    return output_dir


def generate_sample_images():
    """Generate sample images from dataset or create informative placeholders."""
    
    print("\n" + "="*60)
    print("Generating Sample Images")
    print("="*60)
    
    output_dir = Path('progress_report_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Load annotations
    train_df = pd.read_csv('data/processed/train_combined.csv')
    
    # Select diverse samples (mix of kaggle and stanford)
    breeds_to_show = train_df['breed'].value_counts().head(12).index.tolist()
    
    # Smaller figure with tighter spacing for larger text
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    images_found = 0
    breed_idx = 0
    
    # Try to find 6 images that actually exist
    while images_found < 6 and breed_idx < len(breeds_to_show):
        breed = breeds_to_show[breed_idx]
        breed_samples = train_df[train_df['breed'] == breed]
        
        # Try multiple samples from this breed
        for _ in range(min(3, len(breed_samples))):
            if images_found >= 6:
                break
                
            sample = breed_samples.sample(1).iloc[0]
            img_path = None
            
            # Construct image path - try multiple possible locations
            if sample['source'] == 'kaggle':
                parts = Path(sample['image_path']).parts
                breed_folder = parts[1]
                img_name = parts[2]
                
                # Try different possible locations
                possible_paths = [
                    Path(f'src/data/Kaggle_dataset/train/{breed_folder}/{img_name}'),
                    Path(f'src/data/Kaggle_dataset/val/{breed_folder}/{img_name}'),
                    Path(f'src/data/Kaggle_dataset/test/{breed_folder}/{img_name}'),
                    Path(f'src/data/Dog Breed Classification/train/{breed_folder}/{img_name}'),
                    Path(f'src/data/Dog Breed Classification/val/{breed_folder}/{img_name}'),
                    Path(f'src/data/Dog Breed Classification/test/{breed_folder}/{img_name}'),
                ]
            else:  # stanford
                parts = Path(sample['image_path']).parts
                breed_folder = parts[1]
                img_name = parts[2]
                
                possible_paths = [
                    Path(f'src/data/Stanford_dataset/Images/{breed_folder}/{img_name}'),
                    Path(f'src/data/Stanford_dataset/images/Images/{breed_folder}/{img_name}'),
                ]
            
            # Find first existing path
            for path in possible_paths:
                if path.exists():
                    img_path = path
                    break
            
            if img_path and img_path.exists():
                try:
                    img = Image.open(img_path)
                    axes[images_found].imshow(img)
                    axes[images_found].set_title(
                        f"{breed.replace('_', ' ').title()}\n({sample['source'].capitalize()} dataset)", 
                        fontsize=13, fontweight='bold', pad=8
                    )
                    axes[images_found].axis('off')
                    images_found += 1
                    print(f"  ✓ Loaded: {breed} from {sample['source']}")
                    break  # Move to next breed
                except Exception as e:
                    print(f"  ✗ Error loading {img_path}: {e}")
                    continue
        
        breed_idx += 1
    
    # If we couldn't find images, create informative placeholders
    if images_found == 0:
        print(f"\n  ⚠ Warning: Could not access image files")
        print("  Creating informative placeholders with dataset statistics...")
        
        # Create informative placeholders with breed info
        for idx in range(6):
            if idx < len(breeds_to_show):
                breed = breeds_to_show[idx]
                breed_data = train_df[train_df['breed'] == breed]
                count = len(breed_data)
                sources = breed_data['source'].value_counts()
                
                # Create a colored background
                colors = ['#FFE5E5', '#E5F5FF', '#E5FFE5', '#FFF5E5', '#F5E5FF', '#FFE5F5']
                axes[idx].set_facecolor(colors[idx])
                
                # Add breed information with larger font
                breed_title = breed.replace('_', ' ').title()
                info_text = f"{breed_title}\n\n"
                info_text += f"Training: {count}\n"
                for source, src_count in sources.items():
                    info_text += f"{source.capitalize()}: {src_count}\n"
                
                axes[idx].text(0.5, 0.5, info_text,
                             ha='center', va='center', fontsize=14, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=1.2', facecolor='white', 
                                     edgecolor='gray', linewidth=2.5, alpha=0.95))
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
                axes[idx].axis('off')
    elif images_found < 6:
        print(f"\n  ⚠ Warning: Only found {images_found}/6 images")
        print("  Creating placeholders for missing images...")
        for idx in range(images_found, 6):
            if idx < len(breeds_to_show):
                breed = breeds_to_show[idx]
                breed_data = train_df[train_df['breed'] == breed]
                count = len(breed_data)
                
                colors = ['#FFE5E5', '#E5F5FF', '#E5FFE5', '#FFF5E5', '#F5E5FF', '#FFE5F5']
                axes[idx].set_facecolor(colors[idx % 6])
                
                breed_title = breed.replace('_', ' ').title()
                info_text = f"{breed_title}\n\n{count} samples"
                axes[idx].text(0.5, 0.5, info_text,
                             ha='center', va='center', fontsize=14, fontweight='bold',
                             bbox=dict(boxstyle='round,pad=1.2', facecolor='white', 
                                     edgecolor='gray', linewidth=2.5, alpha=0.95))
                axes[idx].set_xlim(0, 1)
                axes[idx].set_ylim(0, 1)
                axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_images.png', dpi=300, bbox_inches='tight')
    print(f"\n✓ Saved: {output_dir / 'sample_images.png'}")
    if images_found > 0:
        print(f"  Successfully loaded {images_found}/6 sample images")
    else:
        print(f"  Created informative placeholders with breed statistics")
    
    plt.close('all')


def generate_model_comparison_table():
    """Generate model comparison table."""
    
    print("\n" + "="*60)
    print("Generating Model Comparison Table")
    print("="*60)
    
    comparison_data = {
        'Model': ['Baseline (ResNet-50)', 'Primary (EfficientNet-B3)'],
        'Parameters': ['25.6M', '12.2M'],
        'Architecture': ['Single-stage', 'Multi-stage pipeline'],
        'Pretrained': ['ImageNet', 'ImageNet'],
        'Training Strategy': ['2-phase', '3-phase progressive'],
        'Special Features': ['Standard transfer learning', 'Label smoothing + calibration']
    }
    
    df = pd.DataFrame(comparison_data)
    print("\nModel Comparison:")
    print(df.to_string(index=False))
    
    # Save as CSV
    output_dir = Path('progress_report_figures')
    df.to_csv(output_dir / 'model_comparison.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'model_comparison.csv'}")


def generate_placeholder_results():
    """Generate placeholder learning curves (to be replaced with actual results)."""
    
    print("\n" + "="*60)
    print("Generating Placeholder Learning Curves")
    print("="*60)
    print("NOTE: These are placeholder curves. Replace with actual training results!")
    
    output_dir = Path('progress_report_figures')
    output_dir.mkdir(exist_ok=True)
    
    # Simulated learning curves
    epochs = np.arange(1, 21)
    
    # Baseline model (simulated)
    baseline_train_acc = 20 + 45 * (1 - np.exp(-epochs/5)) + np.random.normal(0, 2, len(epochs))
    baseline_val_acc = 18 + 40 * (1 - np.exp(-epochs/5)) + np.random.normal(0, 2.5, len(epochs))
    
    # Primary model (simulated - better performance)
    primary_train_acc = 25 + 50 * (1 - np.exp(-epochs/6)) + np.random.normal(0, 1.5, len(epochs))
    primary_val_acc = 22 + 48 * (1 - np.exp(-epochs/6)) + np.random.normal(0, 2, len(epochs))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Baseline
    axes[0].plot(epochs, baseline_train_acc, 'b-', linewidth=2, label='Train Accuracy', marker='o')
    axes[0].plot(epochs, baseline_val_acc, 'r-', linewidth=2, label='Val Accuracy', marker='s')
    axes[0].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[0].set_title('Baseline ResNet-50 Learning Curves', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_ylim([0, 100])
    
    # Primary
    axes[1].plot(epochs, primary_train_acc, 'b-', linewidth=2, label='Train Accuracy', marker='o')
    axes[1].plot(epochs, primary_val_acc, 'r-', linewidth=2, label='Val Accuracy', marker='s')
    axes[1].set_xlabel('Epoch', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    axes[1].set_title('Primary EfficientNet-B3 Learning Curves', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / 'learning_curves_placeholder.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'learning_curves_placeholder.png'}")
    
    plt.close('all')


def main():
    """Generate all progress report content."""
    
    print("\n" + "="*60)
    print("PROGRESS REPORT CONTENT GENERATION")
    print("="*60)
    
    # Generate all content
    generate_data_statistics()
    generate_sample_images()
    generate_model_comparison_table()
    generate_placeholder_results()
    
    print("\n" + "="*60)
    print("CONTENT GENERATION COMPLETE!")
    print("="*60)
    print("\nGenerated files in 'progress_report_figures/':")
    print("  1. dataset_statistics.png - Dataset split and source distribution")
    print("  2. class_distribution.png - Images per breed distribution")
    print("  3. sample_images.png - Sample images from dataset")
    print("  4. model_comparison.csv - Model comparison table")
    print("  5. learning_curves_placeholder.png - Placeholder learning curves")
    print("\nNEXT STEPS:")
    print("  1. Train baseline model: python scripts/train_baseline.py")
    print("  2. Train primary model: python scripts/train_primary.py")
    print("  3. Replace placeholder learning curves with actual results")
    print("  4. Use these figures in your progress report LaTeX document")


if __name__ == '__main__':
    main()
