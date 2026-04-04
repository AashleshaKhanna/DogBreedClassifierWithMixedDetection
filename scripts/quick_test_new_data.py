"""
Quick Test Script for New Data Evaluation
==========================================

This script quickly tests your new data structure and provides a preview
of what the evaluation will look like (without needing trained models).

Usage:
    python scripts/quick_test_new_data.py
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
from PIL import Image

def check_new_data_structure():
    """Check if new data is properly organized"""
    print("="*80)
    print("NEW DATA STRUCTURE VERIFICATION")
    print("="*80)
    
    new_data_dir = Path('data/new_test')
    
    if not new_data_dir.exists():
        print(f"❌ ERROR: New data directory not found: {new_data_dir}")
        print("\nPlease create the directory structure:")
        print("  data/new_test/known_breeds/")
        print("  data/new_test/unknown_breeds/")
        print("  data/new_test/ood/")
        return False
    
    print(f"✓ New data directory exists: {new_data_dir}\n")
    
    # Check subdirectories
    categories = {
        'known_breeds': new_data_dir / 'known_breeds',
        'unknown_breeds': new_data_dir / 'unknown_breeds',
        'ood': new_data_dir / 'ood'
    }
    
    results = {}
    
    for category, path in categories.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        print("-" * 40)
        
        if not path.exists():
            print(f"  ❌ Directory not found: {path}")
            results[category] = {'exists': False, 'count': 0}
            continue
        
        # Count images
        if category == 'known_breeds':
            # Count by breed
            breed_counts = defaultdict(int)
            for breed_dir in path.iterdir():
                if breed_dir.is_dir():
                    images = list(breed_dir.glob('*.jpg')) + list(breed_dir.glob('*.png')) + list(breed_dir.glob('*.jpeg'))
                    breed_counts[breed_dir.name] = len(images)
            
            total = sum(breed_counts.values())
            print(f"  ✓ Found {len(breed_counts)} breeds with {total} total images")
            print(f"\n  Breed breakdown:")
            for breed, count in sorted(breed_counts.items()):
                status = "✓" if count >= 5 else "⚠️"
                print(f"    {status} {breed}: {count} images")
            
            results[category] = {
                'exists': True,
                'count': total,
                'breeds': len(breed_counts),
                'details': breed_counts
            }
        else:
            # Count images directly
            images = list(path.glob('*.jpg')) + list(path.glob('*.png')) + list(path.glob('*.jpeg')) + list(path.glob('*.avif'))
            count = len(images)
            status = "✓" if count >= 20 else "⚠️"
            print(f"  {status} Found {count} images")
            
            if count > 0:
                print(f"\n  Sample files:")
                for img in list(images)[:5]:
                    print(f"    - {img.name}")
            
            results[category] = {
                'exists': True,
                'count': count
            }
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    total_images = sum(r.get('count', 0) for r in results.values())
    print(f"\nTotal new test images: {total_images}")
    
    if 'known_breeds' in results and results['known_breeds']['exists']:
        print(f"  - Known breeds: {results['known_breeds']['count']} images from {results['known_breeds']['breeds']} breeds")
    
    if 'unknown_breeds' in results and results['unknown_breeds']['exists']:
        print(f"  - Unknown breeds: {results['unknown_breeds']['count']} images")
    
    if 'ood' in results and results['ood']['exists']:
        print(f"  - Out-of-distribution: {results['ood']['count']} images")
    
    # Recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    if total_images < 50:
        print("⚠️  You have fewer than 50 images. Aim for at least 50-100 for good evaluation.")
    else:
        print("✓ Good number of images for evaluation!")
    
    if 'known_breeds' in results and results['known_breeds'].get('breeds', 0) < 10:
        print("⚠️  Consider adding more breed diversity (aim for 10+ breeds)")
    
    if 'unknown_breeds' in results and results['unknown_breeds'].get('count', 0) < 20:
        print("⚠️  Add more unknown breed images (aim for 20+)")
    
    if 'ood' in results and results['ood'].get('count', 0) < 20:
        print("⚠️  Add more OOD images (aim for 20+)")
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("\n1. If structure looks good, train your models:")
    print("   python scripts/train_baseline.py")
    print("\n2. After training, evaluate on new data:")
    print("   python scripts/evaluate_new_data.py --model_type baseline --checkpoint checkpoints/baseline/best_model.pth")
    print("\n3. This evaluation is worth 10/60 points in your final report!")
    print("="*80)
    
    return True


def verify_image_quality():
    """Check if images can be loaded"""
    print("\n" + "="*80)
    print("IMAGE QUALITY CHECK")
    print("="*80)
    
    new_data_dir = Path('data/new_test')
    
    # Get all image files
    all_images = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.avif']:
        all_images.extend(new_data_dir.rglob(ext))
    
    if not all_images:
        print("No images found to check")
        return
    
    print(f"\nChecking {len(all_images)} images...")
    
    corrupted = []
    small_images = []
    
    for img_path in all_images:
        try:
            img = Image.open(img_path)
            width, height = img.size
            
            # Check if image is too small
            if width < 100 or height < 100:
                small_images.append((img_path, width, height))
        except Exception as e:
            corrupted.append((img_path, str(e)))
    
    if corrupted:
        print(f"\n❌ Found {len(corrupted)} corrupted images:")
        for path, error in corrupted[:5]:
            print(f"  - {path.name}: {error}")
    else:
        print("\n✓ All images can be loaded successfully!")
    
    if small_images:
        print(f"\n⚠️  Found {len(small_images)} very small images (< 100x100):")
        for path, w, h in small_images[:5]:
            print(f"  - {path.name}: {w}x{h}")
        print("  Consider replacing with higher resolution images")
    else:
        print("✓ All images have good resolution!")


def check_breed_names():
    """Check if known breed names match training breeds"""
    print("\n" + "="*80)
    print("BREED NAME VERIFICATION")
    print("="*80)
    
    # Load training breed names
    breed_file = Path('data/processed/breed_names_combined.txt')
    if not breed_file.exists():
        print(f"⚠️  Training breed names file not found: {breed_file}")
        print("Cannot verify breed name matching")
        return
    
    with open(breed_file, 'r') as f:
        training_breeds = set(line.strip() for line in f)
    
    print(f"\nTraining dataset has {len(training_breeds)} breeds")
    
    # Check known breeds directory
    known_breeds_dir = Path('data/new_test/known_breeds')
    if not known_breeds_dir.exists():
        print("Known breeds directory not found")
        return
    
    new_breeds = set()
    for breed_dir in known_breeds_dir.iterdir():
        if breed_dir.is_dir():
            new_breeds.add(breed_dir.name)
    
    print(f"New test data has {len(new_breeds)} breeds\n")
    
    # Check matches
    matching = new_breeds & training_breeds
    not_matching = new_breeds - training_breeds
    
    print(f"✓ {len(matching)} breeds match training data:")
    for breed in sorted(matching):
        print(f"  - {breed}")
    
    if not_matching:
        print(f"\n⚠️  {len(not_matching)} breeds DON'T match training data:")
        for breed in sorted(not_matching):
            print(f"  - {breed}")
        print("\nThese should be moved to 'unknown_breeds' directory!")
        print("\nTraining breeds include:")
        for breed in sorted(training_breeds)[:20]:
            print(f"  - {breed}")
        print(f"  ... and {len(training_breeds) - 20} more")


def main():
    print("\n" + "="*80)
    print("QUICK NEW DATA TEST")
    print("="*80)
    print("\nThis script verifies your new test data is ready for evaluation.")
    print("It checks:")
    print("  1. Directory structure")
    print("  2. Image counts")
    print("  3. Image quality")
    print("  4. Breed name matching")
    print("="*80)
    
    # Run checks
    check_new_data_structure()
    verify_image_quality()
    check_breed_names()
    
    print("\n" + "="*80)
    print("✓ VERIFICATION COMPLETE")
    print("="*80)
    print("\nIf everything looks good, you're ready to:")
    print("1. Train your models")
    print("2. Evaluate on new data")
    print("3. Get those 10 points in your final report!")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
