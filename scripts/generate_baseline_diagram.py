"""
Generate baseline model architecture diagram for progress report.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set style with larger fonts
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 12  # Increased from 10

# Colors
COLORS = {
    'primary': '#1976D2',
    'baseline': '#FF6B6B',
    'success': '#2E7D32',
    'light_blue': '#E3F2FD',
    'light_red': '#FFE5E5',
    'light_green': '#E8F5E9',
}

fig = plt.figure(figsize=(16, 7))  # Increased from (14, 6)
ax = plt.subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

# Title - larger font
ax.text(7, 5.5, 'Baseline Model: ResNet-50 Single-Stage Classifier', 
        ha='center', fontsize=18, fontweight='bold', color=COLORS['baseline'])

# Input - larger fonts
input_box = FancyBboxPatch((0.5, 2.5), 1.8, 1.5, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                           linewidth=3)
ax.add_patch(input_box)
ax.text(1.4, 3.6, 'Input Image', ha='center', fontsize=13, fontweight='bold')
ax.text(1.4, 3.2, '224×224×3', ha='center', fontsize=11, fontweight='bold')
ax.text(1.4, 2.9, 'RGB', ha='center', fontsize=10, style='italic')

# Arrow 1 - thicker
arrow1 = FancyArrowPatch((2.3, 3.25), (3.2, 3.25),
                        arrowstyle='->', mutation_scale=35, linewidth=3,
                        color=COLORS['primary'])
ax.add_patch(arrow1)

# ResNet-50 Backbone - larger fonts
resnet_box = FancyBboxPatch((3.2, 2.0), 3.5, 2.5, boxstyle="round,pad=0.1",
                            edgecolor=COLORS['baseline'], facecolor=COLORS['light_red'],
                            linewidth=3.5)
ax.add_patch(resnet_box)
ax.text(4.95, 4.2, 'ResNet-50 Backbone', ha='center', fontsize=14, 
        fontweight='bold', color=COLORS['baseline'])
ax.text(4.95, 3.8, '(Pretrained on ImageNet)', ha='center', fontsize=11, style='italic')

# ResNet layers - larger fonts
layers = [
    ('Conv1 + MaxPool', 3.5),
    ('Layer1 (3 blocks)', 3.2),
    ('Layer2 (4 blocks)', 2.9),
    ('Layer3 (6 blocks)', 2.6),
    ('Layer4 (3 blocks)', 2.3),
]

for layer_name, y_pos in layers:
    ax.text(4.95, y_pos, layer_name, ha='center', fontsize=10, fontweight='bold')

# Arrow 2
arrow2 = FancyArrowPatch((6.7, 3.25), (7.6, 3.25),
                        arrowstyle='->', mutation_scale=35, linewidth=3,
                        color=COLORS['primary'])
ax.add_patch(arrow2)

# Global Average Pooling - larger fonts
gap_box = FancyBboxPatch((7.6, 2.7), 1.8, 1.1, boxstyle="round,pad=0.08",
                         edgecolor=COLORS['baseline'], facecolor='white',
                         linewidth=2.5)
ax.add_patch(gap_box)
ax.text(8.5, 3.5, 'Global Avg Pool', ha='center', fontsize=12, fontweight='bold')
ax.text(8.5, 3.1, '2048 features', ha='center', fontsize=10, fontweight='bold')

# Arrow 3
arrow3 = FancyArrowPatch((9.4, 3.25), (10.2, 3.25),
                        arrowstyle='->', mutation_scale=35, linewidth=3,
                        color=COLORS['primary'])
ax.add_patch(arrow3)

# Fully Connected Layer - larger fonts
fc_box = FancyBboxPatch((10.2, 2.7), 1.8, 1.1, boxstyle="round,pad=0.08",
                        edgecolor=COLORS['baseline'], facecolor='white',
                        linewidth=2.5)
ax.add_patch(fc_box)
ax.text(11.1, 3.5, 'Fully Connected', ha='center', fontsize=12, fontweight='bold')
ax.text(11.1, 3.1, '2048 → 93', ha='center', fontsize=10, fontweight='bold')

# Arrow 4
arrow4 = FancyArrowPatch((12.0, 3.25), (12.7, 3.25),
                        arrowstyle='->', mutation_scale=35, linewidth=3,
                        color=COLORS['primary'])
ax.add_patch(arrow4)

# Output - larger fonts
output_box = FancyBboxPatch((12.7, 2.5), 1.2, 1.5, boxstyle="round,pad=0.08",
                            edgecolor=COLORS['success'], facecolor=COLORS['light_green'],
                            linewidth=3)
ax.add_patch(output_box)
ax.text(13.3, 3.6, 'Output', ha='center', fontsize=13, fontweight='bold')
ax.text(13.3, 3.2, '93 classes', ha='center', fontsize=11, fontweight='bold')
ax.text(13.3, 2.9, 'Softmax', ha='center', fontsize=10, style='italic')

# Training info box - larger fonts
training_box = FancyBboxPatch((0.5, 0.3), 13.4, 1.4, boxstyle="round,pad=0.1",
                              edgecolor='#555', facecolor='#F5F5F5',
                              linewidth=2.5, alpha=0.9)
ax.add_patch(training_box)

ax.text(7, 1.45, 'Training Configuration', ha='center', fontsize=13, 
        fontweight='bold', color='#333')

training_text = (
    'Phase 1 (10 epochs): Freeze backbone, train FC only (LR=0.001, SGD)  |  '
    'Phase 2 (20 epochs): Fine-tune entire network (LR=0.0001, SGD, momentum=0.9)'
)
ax.text(7, 1.05, training_text, ha='center', fontsize=10, fontweight='bold')

ax.text(7, 0.65, 'Parameters: 25.6M  |  Batch Size: 32  |  Loss: CrossEntropyLoss  |  '
        'Optimizer: SGD', ha='center', fontsize=10, style='italic', fontweight='bold')

plt.tight_layout()
plt.savefig('progress_report_figures/baseline_model_diagram.png', dpi=300, bbox_inches='tight')
print("✓ Generated: progress_report_figures/baseline_model_diagram.png")
plt.close()

print("\n" + "="*70)
print("BASELINE MODEL DIAGRAM GENERATED!")
print("="*70)
print("\n📍 This diagram shows:")
print("   • ResNet-50 architecture with layer details")
print("   • Input/output specifications")
print("   • Training configuration (2-phase approach)")
print("   • Model parameters and hyperparameters")
print("\n✨ Improvements:")
print("   • Larger fonts (12pt base, up to 18pt for title)")
print("   • Bolder text for better readability")
print("   • Thicker lines and arrows")
print("   • Increased figure size (16x7)")
print("="*70 + "\n")
