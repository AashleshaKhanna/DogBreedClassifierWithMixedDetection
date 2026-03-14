"""
Generate high-quality figures for APS360 Progress Report.
Creates publication-ready visualizations optimized for academic reports.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import seaborn as sns
import pandas as pd
import numpy as np

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 14

# Color palette
COLORS = {
    'primary': '#2E86AB',
    'secondary': '#A23B72',
    'accent1': '#F18F01',
    'accent2': '#C73E1D',
    'success': '#06A77D',
    'neutral': '#6C757D',
    'light_blue': '#E3F2FD',
    'light_green': '#E8F5E9',
    'light_orange': '#FFF3E0',
}


def figure1_dataset_statistics():
    """
    Figure 1: Comprehensive Dataset Statistics
    Shows split distribution, source contribution, and key metrics.
    """
    print("\n" + "="*60)
    print("FIGURE 1: Dataset Statistics")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv('data/processed/train_combined.csv')
    val_df = pd.read_csv('data/processed/val_combined.csv')
    test_df = pd.read_csv('data/processed/test_combined.csv')
    
    # Create figure with 3 subplots
    fig = plt.figure(figsize=(14, 4.5))
    gs = fig.add_gridspec(1, 3, hspace=0.3, wspace=0.35)
    
    # Subplot 1: Split Distribution (Bar Chart)
    ax1 = fig.add_subplot(gs[0, 0])
    splits = ['Train', 'Val', 'Test']
    sizes = [len(train_df), len(val_df), len(test_df)]
    percentages = [s/sum(sizes)*100 for s in sizes]
    colors_split = [COLORS['primary'], COLORS['accent1'], COLORS['accent2']]
    
    bars = ax1.bar(splits, sizes, color=colors_split, edgecolor='black', linewidth=1.5, alpha=0.85)
    ax1.set_ylabel('Number of Images', fontweight='bold')
    ax1.set_title('(a) Dataset Split Distribution', fontweight='bold', pad=10)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, size, pct) in enumerate(zip(bars, sizes, percentages)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 500,
                f'{size:,}\n({pct:.1f}%)',
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Subplot 2: Source Distribution (Pie Chart)
    ax2 = fig.add_subplot(gs[0, 1])
    source_data = train_df['source'].value_counts()
    colors_pie = [COLORS['secondary'], COLORS['success']]
    
    wedges, texts, autotexts = ax2.pie(
        source_data.values, 
        labels=[s.capitalize() for s in source_data.index],
        autopct='%1.1f%%',
        colors=colors_pie,
        startangle=90,
        textprops={'fontsize': 11, 'fontweight': 'bold'},
        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5, 'alpha': 0.85}
    )
    
    # Add count labels
    for i, (label, count) in enumerate(zip(source_data.index, source_data.values)):
        texts[i].set_text(f'{label.capitalize()}\n({count:,} images)')
    
    ax2.set_title('(b) Training Data Sources', fontweight='bold', pad=10)
    
    # Subplot 3: Key Statistics (Text Box)
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.axis('off')
    
    # Calculate statistics
    total_images = len(train_df) + len(val_df) + len(test_df)
    num_breeds = train_df['breed'].nunique()
    avg_per_breed = len(train_df) / num_breeds
    min_samples = train_df['breed'].value_counts().min()
    max_samples = train_df['breed'].value_counts().max()
    
    # Create statistics text
    stats_text = f"""
    Dataset Overview
    {'─'*30}
    
    Total Images:        {total_images:,}
    Number of Breeds:    {num_breeds}
    
    Training Set:        {len(train_df):,}
    Validation Set:      {len(val_df):,}
    Test Set:            {len(test_df):,}
    
    Samples per Breed:
      • Average:         {avg_per_breed:.0f}
      • Minimum:         {min_samples}
      • Maximum:         {max_samples}
    
    Data Sources:        2
      • Kaggle Dataset
      • Stanford Dogs
    """
    
    # Add text box with statistics
    ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=1', facecolor=COLORS['light_blue'], 
                     edgecolor=COLORS['primary'], linewidth=2, alpha=0.9))
    
    ax3.set_title('(c) Dataset Summary', fontweight='bold', pad=10, x=0.5)
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('progress_report_figures')
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'figure1_dataset_statistics.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'figure1_dataset_statistics.png'}")
    
    plt.close()


def figure2_class_distribution():
    """
    Figure 2: Class Distribution Analysis
    Shows distribution of samples across breeds with statistical annotations.
    """
    print("\n" + "="*60)
    print("FIGURE 2: Class Distribution Analysis")
    print("="*60)
    
    # Load data
    train_df = pd.read_csv('data/processed/train_combined.csv')
    
    # Get breed counts
    breed_counts = train_df['breed'].value_counts().sort_values(ascending=False)
    
    # Create figure
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), 
                                    gridspec_kw={'height_ratios': [2, 1]})
    
    # Subplot 1: Full distribution
    x_pos = np.arange(len(breed_counts))
    bars = ax1.bar(x_pos, breed_counts.values, color=COLORS['primary'], 
                   edgecolor='black', linewidth=0.5, alpha=0.85)
    
    # Color top 10 and bottom 10 differently
    for i in range(10):
        bars[i].set_color(COLORS['success'])
    for i in range(len(bars)-10, len(bars)):
        bars[i].set_color(COLORS['accent2'])
    
    ax1.set_xlabel('Breed Index (sorted by frequency)', fontweight='bold')
    ax1.set_ylabel('Number of Training Images', fontweight='bold')
    ax1.set_title('(a) Training Samples per Breed (93 breeds)', fontweight='bold', pad=15)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_axisbelow(True)
    
    # Add statistical lines
    mean_val = breed_counts.mean()
    median_val = breed_counts.median()
    
    ax1.axhline(y=mean_val, color='red', linestyle='--', linewidth=2.5, 
                label=f'Mean: {mean_val:.0f}', alpha=0.8)
    ax1.axhline(y=median_val, color='green', linestyle='--', linewidth=2.5, 
                label=f'Median: {median_val:.0f}', alpha=0.8)
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color=COLORS['success'], label='Top 10 breeds', alpha=0.85),
        mpatches.Patch(color=COLORS['primary'], label='Middle breeds', alpha=0.85),
        mpatches.Patch(color=COLORS['accent2'], label='Bottom 10 breeds', alpha=0.85),
        mpatches.Patch(color='red', label=f'Mean: {mean_val:.0f}'),
        mpatches.Patch(color='green', label=f'Median: {median_val:.0f}'),
    ]
    ax1.legend(handles=legend_elements, loc='upper right', framealpha=0.95)
    
    # Subplot 2: Distribution histogram
    ax2.hist(breed_counts.values, bins=20, color=COLORS['secondary'], 
             edgecolor='black', linewidth=1.5, alpha=0.85)
    ax2.set_xlabel('Number of Training Images per Breed', fontweight='bold')
    ax2.set_ylabel('Frequency (Number of Breeds)', fontweight='bold')
    ax2.set_title('(b) Distribution of Sample Counts', fontweight='bold', pad=15)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_axisbelow(True)
    
    # Add statistics text
    stats_text = f'μ={mean_val:.0f}, σ={breed_counts.std():.0f}, min={breed_counts.min()}, max={breed_counts.max()}'
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
            fontsize=10, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white', 
                     edgecolor='black', linewidth=1.5, alpha=0.9))
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('progress_report_figures')
    plt.savefig(output_dir / 'figure2_class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'figure2_class_distribution.png'}")
    
    plt.close()


def figure3_pipeline_diagram():
    """
    Figure 3: Multi-Stage Pipeline Architecture
    Professional diagram showing the complete system architecture.
    """
    print("\n" + "="*60)
    print("FIGURE 3: Pipeline Architecture Diagram")
    print("="*60)
    
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Title
    ax.text(7, 6.5, 'Dog Breed Classifier with Mixed Detection', 
            ha='center', fontsize=18, fontweight='bold')
    ax.text(7, 6.1, 'Multi-Stage Pipeline Architecture', 
            ha='center', fontsize=13, style='italic', color=COLORS['neutral'])
    
    # Stage 1: Input
    input_box = FancyBboxPatch((0.5, 4), 1.8, 1.2, boxstyle="round,pad=0.1",
                               edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                               linewidth=3)
    ax.add_patch(input_box)
    ax.text(1.4, 4.8, 'Input\nImage', ha='center', va='center', 
            fontsize=11, fontweight='bold')
    ax.text(1.4, 4.3, '(Any size)', ha='center', va='center', 
            fontsize=9, style='italic')
    
    # Arrow 1
    arrow1 = FancyArrowPatch((2.3, 4.6), (3.2, 4.6),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['neutral'])
    ax.add_patch(arrow1)
    
    # Stage 2: Dog Detection (YOLOv5)
    detect_box = FancyBboxPatch((3.2, 3.5), 2.5, 2.2, boxstyle="round,pad=0.1",
                                edgecolor=COLORS['success'], facecolor=COLORS['light_green'],
                                linewidth=3)
    ax.add_patch(detect_box)
    ax.text(4.45, 5.3, 'Stage 1: Detection', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['success'])
    ax.text(4.45, 4.8, 'YOLOv5', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(4.45, 4.4, '• Detect dog regions\n• Crop & resize\n• Handle multiple dogs',
            ha='center', va='center', fontsize=9)
    
    # Decision diamond for detection
    ax.text(4.45, 3.7, 'Dog detected?', ha='center', va='center',
            fontsize=9, style='italic')
    
    # Arrow 2 (Yes path)
    arrow2 = FancyArrowPatch((5.7, 4.6), (6.6, 4.6),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['neutral'])
    ax.add_patch(arrow2)
    ax.text(6.15, 4.85, 'Yes', ha='center', fontsize=9, fontweight='bold',
            color=COLORS['success'])
    
    # Arrow 2b (No path - rejection)
    arrow2b = FancyArrowPatch((4.45, 3.5), (4.45, 2.5),
                             arrowstyle='->', mutation_scale=30, linewidth=2.5,
                             color=COLORS['accent2'], linestyle='--')
    ax.add_patch(arrow2b)
    ax.text(4.7, 3.0, 'No', ha='center', fontsize=9, fontweight='bold',
            color=COLORS['accent2'])
    
    # Rejection box
    reject_box = FancyBboxPatch((3.5, 1.8), 1.9, 0.7, boxstyle="round,pad=0.05",
                                edgecolor=COLORS['accent2'], facecolor='#FFE5E5',
                                linewidth=2, linestyle='--')
    ax.add_patch(reject_box)
    ax.text(4.45, 2.15, 'Reject: Not a dog', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['accent2'])
    
    # Stage 3: Classification (EfficientNet-B3)
    classify_box = FancyBboxPatch((6.6, 3.5), 2.8, 2.2, boxstyle="round,pad=0.1",
                                  edgecolor=COLORS['secondary'], facecolor='#F3E5F5',
                                  linewidth=3)
    ax.add_patch(classify_box)
    ax.text(8.0, 5.3, 'Stage 2: Classification', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['secondary'])
    ax.text(8.0, 4.8, 'EfficientNet-B3', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(8.0, 4.3, '• Pretrained on ImageNet\n• 12.2M parameters\n• Dropout (p=0.3)\n• 93 breed classes',
            ha='center', va='center', fontsize=9)
    
    # Arrow 3
    arrow3 = FancyArrowPatch((9.4, 4.6), (10.3, 4.6),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['neutral'])
    ax.add_patch(arrow3)
    
    # Stage 4: Calibration
    calib_box = FancyBboxPatch((10.3, 3.5), 2.5, 2.2, boxstyle="round,pad=0.1",
                               edgecolor=COLORS['accent1'], facecolor=COLORS['light_orange'],
                               linewidth=3)
    ax.add_patch(calib_box)
    ax.text(11.55, 5.3, 'Stage 3: Calibration', ha='center', va='center',
            fontsize=12, fontweight='bold', color=COLORS['accent1'])
    ax.text(11.55, 4.8, 'Temperature Scaling', ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(11.55, 4.3, '• Calibrate confidence\n• Reliable probabilities\n• OOD detection',
            ha='center', va='center', fontsize=9)
    
    # Decision for confidence threshold
    ax.text(11.55, 3.7, 'Confidence > θ?', ha='center', va='center',
            fontsize=9, style='italic')
    
    # Arrow 4 (High confidence)
    arrow4 = FancyArrowPatch((12.8, 4.6), (13.2, 4.6),
                            arrowstyle='->', mutation_scale=30, linewidth=2.5,
                            color=COLORS['neutral'])
    ax.add_patch(arrow4)
    ax.text(13.0, 4.85, 'Yes', ha='center', fontsize=9, fontweight='bold',
            color=COLORS['success'])
    
    # Arrow 4b (Low confidence - unknown)
    arrow4b = FancyArrowPatch((11.55, 3.5), (11.55, 2.5),
                             arrowstyle='->', mutation_scale=30, linewidth=2.5,
                             color=COLORS['accent2'], linestyle='--')
    ax.add_patch(arrow4b)
    ax.text(11.8, 3.0, 'No', ha='center', fontsize=9, fontweight='bold',
            color=COLORS['accent2'])
    
    # Unknown breed box
    unknown_box = FancyBboxPatch((10.5, 1.8), 2.1, 0.7, boxstyle="round,pad=0.05",
                                 edgecolor=COLORS['accent2'], facecolor='#FFE5E5',
                                 linewidth=2, linestyle='--')
    ax.add_patch(unknown_box)
    ax.text(11.55, 2.15, 'Unknown breed', ha='center', va='center',
            fontsize=10, fontweight='bold', color=COLORS['accent2'])
    
    # Output box
    output_box = FancyBboxPatch((13.2, 4), 0.7, 1.2, boxstyle="round,pad=0.05",
                                edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                                linewidth=3)
    ax.add_patch(output_box)
    ax.text(13.55, 4.6, 'Output', ha='center', va='center',
            fontsize=10, fontweight='bold')
    
    # Training info box at bottom
    training_box = FancyBboxPatch((0.5, 0.2), 13.4, 1.2, boxstyle="round,pad=0.1",
                                  edgecolor=COLORS['neutral'], facecolor='#F5F5F5',
                                  linewidth=2, linestyle='-', alpha=0.9)
    ax.add_patch(training_box)
    
    ax.text(7, 1.15, 'Training Strategy', ha='center', fontsize=12, 
            fontweight='bold', color=COLORS['neutral'])
    
    training_text = (
        'Phase 1 (10 epochs): Frozen backbone, train head only (LR=0.001)  |  '
        'Phase 2 (20 epochs): Fine-tune last 2 blocks (LR=0.0005)  |  '
        'Phase 3 (10 epochs): Full fine-tuning + label smoothing (LR=0.0001)'
    )
    ax.text(7, 0.65, training_text, ha='center', fontsize=9, style='italic')
    
    ax.text(7, 0.35, 'Dataset: 48,338 images | 93 breeds | Kaggle + Stanford sources',
            ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = Path('progress_report_figures')
    plt.savefig(output_dir / 'figure3_pipeline_diagram.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {output_dir / 'figure3_pipeline_diagram.png'}")
    
    plt.close()


def main():
    """Generate all three high-quality figures."""
    
    print("\n" + "="*70)
    print(" "*15 + "GENERATING HIGH-QUALITY REPORT FIGURES")
    print("="*70)
    
    # Generate all figures
    figure1_dataset_statistics()
    figure2_class_distribution()
    figure3_pipeline_diagram()
    
    print("\n" + "="*70)
    print("ALL FIGURES GENERATED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📊 FIGURE PLACEMENT GUIDE FOR PROGRESS REPORT:")
    print("-" * 70)
    
    print("\n1️⃣  FIGURE 1: Dataset Statistics")
    print("   File: figure1_dataset_statistics.png")
    print("   Place in: Data Processing section")
    print("   Caption: 'Comprehensive dataset statistics showing (a) split")
    print("            distribution across train/validation/test sets, (b)")
    print("            contribution from Kaggle and Stanford data sources, and")
    print("            (c) key dataset metrics including 48,338 total images")
    print("            across 93 dog breeds.'")
    
    print("\n2️⃣  FIGURE 2: Class Distribution Analysis")
    print("   File: figure2_class_distribution.png")
    print("   Place in: Data Processing section (after Figure 1)")
    print("   Caption: 'Class distribution analysis showing (a) training samples")
    print("            per breed with top 10 (green) and bottom 10 (red) breeds")
    print("            highlighted, and (b) histogram of sample counts")
    print("            demonstrating relatively balanced distribution with mean")
    print("            484 and median 485 samples per breed.'")
    
    print("\n3️⃣  FIGURE 3: Pipeline Architecture Diagram")
    print("   File: figure3_pipeline_diagram.png")
    print("   Place in: Architecture section OR Notable Contribution intro")
    print("   Caption: 'Multi-stage pipeline architecture for robust dog breed")
    print("            classification. Stage 1 uses YOLOv5 for dog detection and")
    print("            cropping, Stage 2 employs EfficientNet-B3 for breed")
    print("            classification, and Stage 3 applies temperature scaling")
    print("            for confidence calibration. The system rejects non-dog")
    print("            images and unknown breeds based on confidence thresholds.'")
    
    print("\n" + "="*70)
    print("💡 TIPS FOR MAXIMUM MARKS:")
    print("-" * 70)
    print("• Reference figures in text: 'As shown in Figure X...'")
    print("• Explain key insights from each figure in surrounding text")
    print("• Use \\ref{fig:label} in LaTeX for automatic numbering")
    print("• Ensure figures are readable when printed in grayscale")
    print("• All figures use professional color schemes and clear labels")
    print("="*70 + "\n")


if __name__ == '__main__':
    main()
