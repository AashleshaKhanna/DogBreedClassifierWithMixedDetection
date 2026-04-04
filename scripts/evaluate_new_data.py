"""
Evaluate models on new, never-before-seen data.
This is CRITICAL for the final report (10/60 points).

Expected directory structure:
data/new_test/
├── known_breeds/        # Dogs from the 93 trained breeds
│   ├── breed1/
│   ├── breed2/
│   └── ...
├── unknown_breeds/      # Dogs from breeds NOT in training
│   └── images/
└── ood/                # Out-of-distribution (cats, wolves, objects)
    └── images/
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import yaml
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse

from src.models.baseline import create_baseline_model
from src.models.classifier import create_model
from src.training.utils import get_device


class NewDataDataset(Dataset):
    """Dataset for new test images."""
    
    def __init__(self, root_dir, transform=None, breed_to_idx=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.breed_to_idx = breed_to_idx
        
        self.samples = []
        self.categories = []
        
        # Load known breeds
        known_dir = self.root_dir / 'known_breeds'
        if known_dir.exists():
            for breed_dir in known_dir.iterdir():
                if breed_dir.is_dir():
                    breed_name = breed_dir.name.lower().replace('_', ' ')
                    for img_path in breed_dir.glob('*'):
                        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            self.samples.append((img_path, breed_name, 'known'))
                            self.categories.append('known')
        
        # Load unknown breeds
        unknown_dir = self.root_dir / 'unknown_breeds'
        if unknown_dir.exists():
            for img_path in unknown_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, 'unknown', 'unknown'))
                    self.categories.append('unknown')
        
        # Load OOD samples
        ood_dir = self.root_dir / 'ood'
        if ood_dir.exists():
            for img_path in ood_dir.rglob('*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    self.samples.append((img_path, 'ood', 'ood'))
                    self.categories.append('ood')
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, breed, category = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            
            # Get label index if known breed
            if category == 'known' and self.breed_to_idx:
                label = self.breed_to_idx.get(breed, -1)
            else:
                label = -1  # Unknown/OOD
            
            return image, label, category, str(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a dummy tensor
            return torch.zeros(3, 224, 224), -1, category, str(img_path)


def load_model(model_type, checkpoint_path, num_classes, device):
    """Load trained model from checkpoint."""
    if model_type == 'baseline':
        model = create_baseline_model(num_classes=num_classes, pretrained=False, device=device)
    else:
        model = create_model(num_classes=num_classes, pretrained=False, device=device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def evaluate_new_data(model, dataloader, device, idx_to_breed):
    """Evaluate model on new data."""
    model.eval()
    
    results_by_category = {
        'known': {'correct': 0, 'total': 0, 'confidences': []},
        'unknown': {'total': 0, 'confidences': []},
        'ood': {'total': 0, 'confidences': []}
    }
    
    all_predictions = []
    
    with torch.no_grad():
        for images, labels, categories, paths in tqdm(dataloader, desc='Evaluating new data'):
            images = images.to(device)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            
            max_probs, predictions = probs.max(1)
            
            for i in range(len(images)):
                category = categories[i]
                confidence = max_probs[i].item()
                pred_idx = predictions[i].item()
                pred_breed = idx_to_breed[pred_idx]
                true_label = labels[i].item()
                
                results_by_category[category]['total'] += 1
                results_by_category[category]['confidences'].append(confidence)
                
                if category == 'known' and true_label >= 0:
                    if pred_idx == true_label:
                        results_by_category[category]['correct'] += 1
                
                all_predictions.append({
                    'path': paths[i],
                    'category': category,
                    'predicted_breed': pred_breed,
                    'confidence': confidence,
                    'correct': (pred_idx == true_label) if true_label >= 0 else None
                })
    
    return results_by_category, all_predictions


def plot_confidence_distribution(results_by_category, save_path):
    """Plot confidence distributions by category."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    categories = ['known', 'unknown', 'ood']
    titles = ['Known Breeds', 'Unknown Breeds', 'Out-of-Distribution']
    colors = ['green', 'orange', 'red']
    
    for ax, category, title, color in zip(axes, categories, titles, colors):
        confidences = results_by_category[category]['confidences']
        if confidences:
            ax.hist(confidences, bins=20, color=color, alpha=0.7, edgecolor='black')
            ax.axvline(np.mean(confidences), color='black', linestyle='--', 
                      linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
            ax.set_xlabel('Confidence', fontsize=12)
            ax.set_ylabel('Count', fontsize=12)
            ax.set_title(f'{title}\n(n={len(confidences)})', fontsize=13, fontweight='bold')
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=14)
            ax.set_title(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved confidence distribution to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate on new data')
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--new-data-dir', type=str, default='data/new_test')
    parser.add_argument('--baseline-checkpoint', type=str,
                       default='checkpoints/baseline/baseline_best.pth')
    parser.add_argument('--primary-checkpoint', type=str,
                       default='checkpoints/primary/primary_best.pth')
    parser.add_argument('--output-dir', type=str, default='results/new_data')
    args = parser.parse_args()
    
    # Check if new data directory exists
    new_data_dir = Path(args.new_data_dir)
    if not new_data_dir.exists():
        print(f"ERROR: New data directory not found: {new_data_dir}")
        print("\nPlease create the directory structure:")
        print("data/new_test/")
        print("├── known_breeds/")
        print("│   ├── breed1/")
        print("│   └── breed2/")
        print("├── unknown_breeds/")
        print("└── ood/")
        return
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = get_device()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("EVALUATING ON NEW DATA (NEVER-BEFORE-SEEN)")
    print("="*60)
    
    # Load breed mapping from training data
    from src.data.dataset import DogBreedDataset
    from src.data.augmentation import get_val_transforms
    
    train_dataset = DogBreedDataset(
        image_dir='src/data',
        annotations_file=config['data']['train_annotations'],
        transform=None,
        use_combined=True
    )
    breed_to_idx = train_dataset.breed_to_idx
    idx_to_breed = {v: k for k, v in breed_to_idx.items()}
    
    # Load new data
    print(f"\nLoading new data from: {new_data_dir}")
    transform = get_val_transforms(config['data']['image_size'])
    new_dataset = NewDataDataset(new_data_dir, transform=transform, breed_to_idx=breed_to_idx)
    
    if len(new_dataset) == 0:
        print("ERROR: No images found in new data directory!")
        return
    
    print(f"Total new samples: {len(new_dataset)}")
    print(f"  Known breeds: {new_dataset.categories.count('known')}")
    print(f"  Unknown breeds: {new_dataset.categories.count('unknown')}")
    print(f"  OOD samples: {new_dataset.categories.count('ood')}")
    
    new_loader = DataLoader(
        new_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0  # Use 0 for debugging
    )
    
    # Evaluate baseline
    if Path(args.baseline_checkpoint).exists():
        print("\n" + "="*60)
        print("BASELINE MODEL (ResNet-50)")
        print("="*60)
        
        baseline_model = load_model('baseline', args.baseline_checkpoint,
                                    len(breed_to_idx), device)
        baseline_results, baseline_preds = evaluate_new_data(
            baseline_model, new_loader, device, idx_to_breed
        )
        
        print("\nResults:")
        for category in ['known', 'unknown', 'ood']:
            total = baseline_results[category]['total']
            if total > 0:
                confidences = baseline_results[category]['confidences']
                avg_conf = np.mean(confidences)
                print(f"\n{category.upper()}:")
                print(f"  Samples: {total}")
                print(f"  Avg Confidence: {avg_conf:.4f}")
                
                if category == 'known':
                    correct = baseline_results[category]['correct']
                    accuracy = 100 * correct / total
                    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Save results
        with open(output_dir / 'baseline_new_data_results.json', 'w') as f:
            save_results = {
                'summary': {cat: {k: v for k, v in data.items() if k != 'confidences'}
                           for cat, data in baseline_results.items()},
                'predictions': baseline_preds[:100]  # Save first 100
            }
            json.dump(save_results, f, indent=2)
        
        plot_confidence_distribution(
            baseline_results,
            output_dir / 'baseline_confidence_distribution.png'
        )
    
    # Evaluate primary
    if Path(args.primary_checkpoint).exists():
        print("\n" + "="*60)
        print("PRIMARY MODEL (EfficientNet-B3)")
        print("="*60)
        
        primary_model = load_model('primary', args.primary_checkpoint,
                                   len(breed_to_idx), device)
        primary_results, primary_preds = evaluate_new_data(
            primary_model, new_loader, device, idx_to_breed
        )
        
        print("\nResults:")
        for category in ['known', 'unknown', 'ood']:
            total = primary_results[category]['total']
            if total > 0:
                confidences = primary_results[category]['confidences']
                avg_conf = np.mean(confidences)
                print(f"\n{category.upper()}:")
                print(f"  Samples: {total}")
                print(f"  Avg Confidence: {avg_conf:.4f}")
                
                if category == 'known':
                    correct = primary_results[category]['correct']
                    accuracy = 100 * correct / total
                    print(f"  Accuracy: {accuracy:.2f}% ({correct}/{total})")
        
        # Save results
        with open(output_dir / 'primary_new_data_results.json', 'w') as f:
            save_results = {
                'summary': {cat: {k: v for k, v in data.items() if k != 'confidences'}
                           for cat, data in primary_results.items()},
                'predictions': primary_preds[:100]
            }
            json.dump(save_results, f, indent=2)
        
        plot_confidence_distribution(
            primary_results,
            output_dir / 'primary_confidence_distribution.png'
        )
    
    print(f"\n{'='*60}")
    print(f"New data evaluation complete! Results saved to: {output_dir}")
    print(f"{'='*60}")
    print("\n⚠️  IMPORTANT FOR FINAL REPORT:")
    print("Document the following in your report:")
    print("1. Where you obtained the new data (sources)")
    print("2. How many images in each category")
    print("3. Performance comparison: new data vs original test set")
    print("4. Analysis of why performance differs")
    print("5. Confidence distributions for known/unknown/OOD")


if __name__ == '__main__':
    main()
