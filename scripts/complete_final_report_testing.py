"""
Complete Final Report Testing Script
=====================================

This script orchestrates all testing needed for the APS360 final report:
1. Trains baseline and primary models (if not already trained)
2. Evaluates on original test set
3. Evaluates on new unseen data (known breeds, unknown breeds, OOD)
4. Generates all quantitative and qualitative results
5. Creates all required figures and tables

Usage:
    python scripts/complete_final_report_testing.py --mode all
    python scripts/complete_final_report_testing.py --mode evaluate_only  # If models already trained
    python scripts/complete_final_report_testing.py --mode new_data_only  # Just new data evaluation
"""

import argparse
import os
import sys
import json
import time
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from tqdm import tqdm

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.classifier import BreedClassifier
from src.data.dataset import DogBreedDataset, get_transforms
from src.training.evaluate import evaluate_model, compute_per_class_accuracy, generate_confusion_matrix
from src.visualization.plots import plot_training_history, plot_confusion_matrix, plot_top_k_accuracy


class FinalReportTester:
    """Orchestrates all testing for final report"""
    
    def __init__(self, config_path='configs/config.yaml', output_dir='final_report_results'):
        self.config_path = config_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (self.output_dir / 'figures').mkdir(exist_ok=True)
        (self.output_dir / 'tables').mkdir(exist_ok=True)
        (self.output_dir / 'qualitative').mkdir(exist_ok=True)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load breed mappings
        self.load_breed_mappings()
        
    def load_breed_mappings(self):
        """Load breed name mappings"""
        breed_file = Path('data/processed/breed_names_combined.txt')
        if breed_file.exists():
            with open(breed_file, 'r') as f:
                self.breeds = [line.strip() for line in f]
            self.breed_to_idx = {breed: idx for idx, breed in enumerate(self.breeds)}
            self.idx_to_breed = {idx: breed for idx, breed in enumerate(self.breeds)}
            self.num_classes = len(self.breeds)
            print(f"Loaded {self.num_classes} breed classes")
        else:
            raise FileNotFoundError(f"Breed names file not found: {breed_file}")
    
    def check_models_exist(self):
        """Check if trained models exist"""
        baseline_path = Path('checkpoints/baseline/best_model.pth')
        primary_path = Path('checkpoints/primary/best_model.pth')
        
        return {
            'baseline': baseline_path.exists(),
            'primary': primary_path.exists(),
            'baseline_path': baseline_path if baseline_path.exists() else None,
            'primary_path': primary_path if primary_path.exists() else None
        }
    
    def train_models(self):
        """Train baseline and primary models"""
        print("\n" + "="*80)
        print("PHASE 1: MODEL TRAINING")
        print("="*80)
        
        models_status = self.check_models_exist()
        
        # Train baseline if needed
        if not models_status['baseline']:
            print("\n[1/2] Training Baseline Model (ResNet-50)...")
            print("This will take 2-4 hours with GPU")
            os.system('python scripts/train_baseline.py')
        else:
            print("\n[1/2] Baseline model already trained ✓")
        
        # Train primary if needed
        if not models_status['primary']:
            print("\n[2/2] Training Primary Model (EfficientNet-B3)...")
            print("This will take 3-5 hours with GPU")
            os.system('python scripts/train_primary.py')
        else:
            print("\n[2/2] Primary model already trained ✓")
    
    def evaluate_on_test_set(self, model_type='baseline'):
        """Evaluate model on original test set"""
        print(f"\nEvaluating {model_type} model on test set...")
        
        # Load model
        checkpoint_path = f'checkpoints/{model_type}/best_model.pth'
        model = self.load_model(model_type, checkpoint_path)
        
        # Load test data
        test_dataset = DogBreedDataset(
            annotations_file='data/processed/test_annotations.csv',
            img_dir='data/processed',
            transform=get_transforms(train=False)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
        
        # Evaluate
        results = evaluate_model(model, test_loader, self.device)
        
        # Save results
        results_file = self.output_dir / 'tables' / f'{model_type}_test_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Test Results for {model_type}:")
        print(f"  Top-1 Accuracy: {results['top1_accuracy']:.2%}")
        print(f"  Top-5 Accuracy: {results['top5_accuracy']:.2%}")
        print(f"  Average Loss: {results['avg_loss']:.4f}")
        
        return results
    
    def evaluate_on_new_data(self, model_type='baseline'):
        """Evaluate on new unseen data"""
        print(f"\nEvaluating {model_type} model on NEW DATA...")
        
        # This is worth 10/60 points in the final report!
        new_data_dir = Path('data/new_test')
        
        if not new_data_dir.exists():
            print(f"WARNING: New data directory not found: {new_data_dir}")
            print("Please collect new test data first!")
            return None
        
        # Run the dedicated new data evaluation script
        cmd = f'python scripts/evaluate_new_data.py --model_type {model_type} --checkpoint checkpoints/{model_type}/best_model.pth'
        os.system(cmd)
        
        print(f"✓ New data evaluation complete for {model_type}")
    
    def generate_quantitative_results(self):
        """Generate all quantitative results tables"""
        print("\n" + "="*80)
        print("PHASE 3: QUANTITATIVE RESULTS GENERATION")
        print("="*80)
        
        # Load results from both models
        baseline_results = self.load_results('baseline')
        primary_results = self.load_results('primary')
        
        # Create comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['Top-1 Accuracy', 'Top-5 Accuracy', 'Test Loss'],
            'Baseline (ResNet-50)': [
                f"{baseline_results['top1_accuracy']:.2%}",
                f"{baseline_results['top5_accuracy']:.2%}",
                f"{baseline_results['avg_loss']:.4f}"
            ],
            'Primary (EfficientNet-B3)': [
                f"{primary_results['top1_accuracy']:.2%}",
                f"{primary_results['top5_accuracy']:.2%}",
                f"{primary_results['avg_loss']:.4f}"
            ]
        })
        
        # Save table
        table_path = self.output_dir / 'tables' / 'model_comparison.csv'
        comparison_df.to_csv(table_path, index=False)
        print(f"✓ Saved comparison table: {table_path}")
        
        # Print for LaTeX
        print("\nLaTeX Table:")
        print(comparison_df.to_latex(index=False))
    
    def generate_qualitative_results(self, model_type='primary', num_samples=20):
        """Generate qualitative results (sample predictions)"""
        print("\n" + "="*80)
        print("PHASE 4: QUALITATIVE RESULTS GENERATION")
        print("="*80)
        
        print(f"Generating sample predictions for {model_type} model...")
        
        # Load model
        checkpoint_path = f'checkpoints/{model_type}/best_model.pth'
        model = self.load_model(model_type, checkpoint_path)
        model.eval()
        
        # Load test data
        test_dataset = DogBreedDataset(
            annotations_file='data/processed/test_annotations.csv',
            img_dir='data/processed',
            transform=get_transforms(train=False)
        )
        
        # Get random samples
        indices = np.random.choice(len(test_dataset), num_samples, replace=False)
        
        # Create figure for sample predictions
        fig, axes = plt.subplots(4, 5, figsize=(20, 16))
        axes = axes.flatten()
        
        correct_count = 0
        
        for idx, sample_idx in enumerate(indices):
            image, label = test_dataset[sample_idx]
            
            # Get prediction
            with torch.no_grad():
                output = model(image.unsqueeze(0).to(self.device))
                probs = torch.softmax(output, dim=1)
                top5_probs, top5_indices = torch.topk(probs, 5)
                
                pred_idx = top5_indices[0][0].item()
                pred_prob = top5_probs[0][0].item()
            
            # Display
            img_display = image.permute(1, 2, 0).numpy()
            img_display = (img_display * 0.229 + 0.485).clip(0, 1)  # Denormalize
            
            axes[idx].imshow(img_display)
            axes[idx].axis('off')
            
            true_breed = self.idx_to_breed[label]
            pred_breed = self.idx_to_breed[pred_idx]
            
            is_correct = (pred_idx == label)
            if is_correct:
                correct_count += 1
                color = 'green'
                symbol = '✓'
            else:
                color = 'red'
                symbol = '✗'
            
            title = f"{symbol} True: {true_breed}\nPred: {pred_breed} ({pred_prob:.1%})"
            axes[idx].set_title(title, fontsize=8, color=color)
        
        plt.tight_layout()
        save_path = self.output_dir / 'qualitative' / f'{model_type}_sample_predictions.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Sample predictions: {correct_count}/{num_samples} correct")
        print(f"✓ Saved figure: {save_path}")
    
    def generate_failure_analysis(self, model_type='primary', num_failures=10):
        """Analyze and visualize failure cases"""
        print(f"\nGenerating failure analysis for {model_type}...")
        
        # Load model
        checkpoint_path = f'checkpoints/{model_type}/best_model.pth'
        model = self.load_model(model_type, checkpoint_path)
        model.eval()
        
        # Load test data
        test_dataset = DogBreedDataset(
            annotations_file='data/processed/test_annotations.csv',
            img_dir='data/processed',
            transform=get_transforms(train=False)
        )
        
        # Find misclassified examples
        failures = []
        
        for idx in range(len(test_dataset)):
            image, label = test_dataset[idx]
            
            with torch.no_grad():
                output = model(image.unsqueeze(0).to(self.device))
                pred = output.argmax(dim=1).item()
            
            if pred != label:
                probs = torch.softmax(output, dim=1)
                failures.append({
                    'idx': idx,
                    'true_label': label,
                    'pred_label': pred,
                    'confidence': probs[0][pred].item(),
                    'image': image
                })
            
            if len(failures) >= num_failures:
                break
        
        # Visualize failures
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        for idx, failure in enumerate(failures[:10]):
            img_display = failure['image'].permute(1, 2, 0).numpy()
            img_display = (img_display * 0.229 + 0.485).clip(0, 1)
            
            axes[idx].imshow(img_display)
            axes[idx].axis('off')
            
            true_breed = self.idx_to_breed[failure['true_label']]
            pred_breed = self.idx_to_breed[failure['pred_label']]
            
            title = f"True: {true_breed}\nPred: {pred_breed}\n({failure['confidence']:.1%})"
            axes[idx].set_title(title, fontsize=8, color='red')
        
        plt.suptitle(f'Failure Cases - {model_type.upper()} Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        save_path = self.output_dir / 'qualitative' / f'{model_type}_failure_cases.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved failure analysis: {save_path}")
    
    def generate_all_figures(self):
        """Generate all required figures for final report"""
        print("\n" + "="*80)
        print("PHASE 5: FIGURE GENERATION")
        print("="*80)
        
        # 1. Learning curves
        print("\n[1/5] Generating learning curves...")
        os.system('python scripts/generate_learning_curves.py')
        
        # 2. Confusion matrices
        print("\n[2/5] Generating confusion matrices...")
        for model_type in ['baseline', 'primary']:
            self.generate_confusion_matrix(model_type)
        
        # 3. Sample predictions (already done in qualitative results)
        print("\n[3/5] Sample predictions already generated ✓")
        
        # 4. Failure analysis (already done)
        print("\n[4/5] Failure analysis already generated ✓")
        
        # 5. Performance comparison charts
        print("\n[5/5] Generating performance comparison charts...")
        self.generate_comparison_charts()
        
        print("\n✓ All figures generated!")
    
    def generate_confusion_matrix(self, model_type='baseline'):
        """Generate confusion matrix for a model"""
        print(f"Generating confusion matrix for {model_type}...")
        
        # Load model
        checkpoint_path = f'checkpoints/{model_type}/best_model.pth'
        model = self.load_model(model_type, checkpoint_path)
        
        # Load test data
        test_dataset = DogBreedDataset(
            annotations_file='data/processed/test_annotations.csv',
            img_dir='data/processed',
            transform=get_transforms(train=False)
        )
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Generate confusion matrix
        cm = generate_confusion_matrix(model, test_loader, self.device, self.num_classes)
        
        # Plot (simplified version for many classes)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, cmap='Blues', cbar=True, square=True)
        plt.title(f'Confusion Matrix - {model_type.upper()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        save_path = self.output_dir / 'figures' / f'{model_type}_confusion_matrix.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved confusion matrix: {save_path}")
    
    def generate_comparison_charts(self):
        """Generate comparison charts between baseline and primary"""
        baseline_results = self.load_results('baseline')
        primary_results = self.load_results('primary')
        
        # Bar chart comparison
        metrics = ['Top-1 Acc', 'Top-5 Acc']
        baseline_values = [baseline_results['top1_accuracy'], baseline_results['top5_accuracy']]
        primary_values = [primary_results['top1_accuracy'], primary_results['top5_accuracy']]
        
        x = np.arange(len(metrics))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(x - width/2, baseline_values, width, label='Baseline (ResNet-50)', color='skyblue')
        ax.bar(x + width/2, primary_values, width, label='Primary (EfficientNet-B3)', color='coral')
        
        ax.set_ylabel('Accuracy')
        ax.set_title('Model Performance Comparison')
        ax.set_xticks(x)
        ax.set_xticklabels(metrics)
        ax.legend()
        ax.set_ylim([0, 1])
        
        # Add value labels on bars
        for i, v in enumerate(baseline_values):
            ax.text(i - width/2, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        for i, v in enumerate(primary_values):
            ax.text(i + width/2, v + 0.02, f'{v:.1%}', ha='center', va='bottom')
        
        save_path = self.output_dir / 'figures' / 'model_comparison.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Saved comparison chart: {save_path}")
    
    def load_model(self, model_type, checkpoint_path):
        """Load a trained model"""
        model = BreedClassifier(
            num_classes=self.num_classes,
            model_type='resnet50' if model_type == 'baseline' else 'efficientnet_b3'
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model
    
    def load_results(self, model_type):
        """Load saved results"""
        results_file = self.output_dir / 'tables' / f'{model_type}_test_results.json'
        with open(results_file, 'r') as f:
            return json.load(f)
    
    def generate_final_summary(self):
        """Generate final summary report"""
        print("\n" + "="*80)
        print("FINAL SUMMARY")
        print("="*80)
        
        summary = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'models_trained': self.check_models_exist(),
            'output_directory': str(self.output_dir),
            'files_generated': {
                'figures': len(list((self.output_dir / 'figures').glob('*.png'))),
                'tables': len(list((self.output_dir / 'tables').glob('*.csv'))),
                'qualitative': len(list((self.output_dir / 'qualitative').glob('*.png')))
            }
        }
        
        # Save summary
        summary_file = self.output_dir / 'summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n✓ Testing complete!")
        print(f"✓ Results saved to: {self.output_dir}")
        print(f"\nGenerated files:")
        print(f"  - Figures: {summary['files_generated']['figures']}")
        print(f"  - Tables: {summary['files_generated']['tables']}")
        print(f"  - Qualitative: {summary['files_generated']['qualitative']}")
        
        print("\n" + "="*80)
        print("NEXT STEPS FOR FINAL REPORT")
        print("="*80)
        print("1. Review all generated figures in final_report_results/figures/")
        print("2. Review quantitative tables in final_report_results/tables/")
        print("3. Review qualitative results in final_report_results/qualitative/")
        print("4. Use these results to write your final report sections:")
        print("   - Quantitative Results (4 points)")
        print("   - Qualitative Results (4 points)")
        print("   - Evaluate on New Data (10 points)")
        print("   - Discussion (8 points)")
        print("="*80)
    
    def run_complete_workflow(self):
        """Run the complete testing workflow"""
        print("\n" + "="*80)
        print("COMPLETE FINAL REPORT TESTING WORKFLOW")
        print("="*80)
        print("This will:")
        print("1. Train models (if needed)")
        print("2. Evaluate on test set")
        print("3. Evaluate on new data (10 points!)")
        print("4. Generate quantitative results")
        print("5. Generate qualitative results")
        print("6. Generate all figures")
        print("="*80)
        
        # Phase 1: Training
        self.train_models()
        
        # Phase 2: Evaluation on test set
        print("\n" + "="*80)
        print("PHASE 2: TEST SET EVALUATION")
        print("="*80)
        for model_type in ['baseline', 'primary']:
            self.evaluate_on_test_set(model_type)
        
        # Phase 2.5: Evaluation on new data (CRITICAL - 10 points!)
        print("\n" + "="*80)
        print("PHASE 2.5: NEW DATA EVALUATION (10 POINTS!)")
        print("="*80)
        for model_type in ['baseline', 'primary']:
            self.evaluate_on_new_data(model_type)
        
        # Phase 3: Quantitative results
        self.generate_quantitative_results()
        
        # Phase 4: Qualitative results
        for model_type in ['baseline', 'primary']:
            self.generate_qualitative_results(model_type)
            self.generate_failure_analysis(model_type)
        
        # Phase 5: All figures
        self.generate_all_figures()
        
        # Final summary
        self.generate_final_summary()


def main():
    parser = argparse.ArgumentParser(description='Complete Final Report Testing')
    parser.add_argument('--mode', type=str, default='all',
                       choices=['all', 'evaluate_only', 'new_data_only', 'figures_only'],
                       help='Testing mode')
    parser.add_argument('--output_dir', type=str, default='final_report_results',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    tester = FinalReportTester(output_dir=args.output_dir)
    
    if args.mode == 'all':
        tester.run_complete_workflow()
    elif args.mode == 'evaluate_only':
        # Skip training, just evaluate
        for model_type in ['baseline', 'primary']:
            tester.evaluate_on_test_set(model_type)
            tester.evaluate_on_new_data(model_type)
        tester.generate_quantitative_results()
    elif args.mode == 'new_data_only':
        # Just evaluate on new data
        for model_type in ['baseline', 'primary']:
            tester.evaluate_on_new_data(model_type)
    elif args.mode == 'figures_only':
        # Just generate figures
        tester.generate_all_figures()


if __name__ == '__main__':
    main()
