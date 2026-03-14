"""
Generate simple, intuitive project overview figure.
Shows dog images → model → breed predictions in a clear, visual way.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

# Colors
COLORS = {
    'primary': '#1976D2',
    'success': '#2E7D32',
    'warning': '#F57C00',
    'error': '#C62828',
    'light_blue': '#E3F2FD',
    'light_green': '#E8F5E9',
    'light_red': '#FFEBEE',
    'light_gray': '#F5F5F5',
}

fig = plt.figure(figsize=(14, 8))
ax = plt.subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 8)
ax.axis('off')

# Title
ax.text(7, 7.5, 'Dog Breed Classifier with Mixed Detection', 
        ha='center', fontsize=18, fontweight='bold', color=COLORS['primary'])
ax.text(7, 7.0, 'Automatically identify dog breeds from photos', 
        ha='center', fontsize=13, style='italic', color='#555')

# ============================================================================
# LEFT: INPUT EXAMPLES (3 dog scenarios)
# ============================================================================
ax.text(2.0, 6.3, 'INPUT: Any Photo', ha='center', fontsize=13, 
        fontweight='bold', color=COLORS['primary'])

# Input Example 1: Golden Retriever
input1_box = FancyBboxPatch((0.5, 4.8), 1.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                            linewidth=2.5)
ax.add_patch(input1_box)
ax.text(1.25, 5.7, '🐕', ha='center', fontsize=40)
ax.text(1.25, 5.0, 'Photo 1', ha='center', fontsize=9, style='italic')

# Input Example 2: Mixed breed
input2_box = FancyBboxPatch((2.3, 4.8), 1.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                            linewidth=2.5)
ax.add_patch(input2_box)
ax.text(3.05, 5.7, '🐶', ha='center', fontsize=40)
ax.text(3.05, 5.0, 'Photo 2', ha='center', fontsize=9, style='italic')

# Input Example 3: Not a dog
input3_box = FancyBboxPatch((0.5, 3.2), 1.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                            linewidth=2.5)
ax.add_patch(input3_box)
ax.text(1.25, 4.1, '🐱', ha='center', fontsize=40)
ax.text(1.25, 3.4, 'Photo 3', ha='center', fontsize=9, style='italic')

# Input Example 4: Multiple dogs
input4_box = FancyBboxPatch((2.3, 3.2), 1.5, 1.2, boxstyle="round,pad=0.05",
                            edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                            linewidth=2.5)
ax.add_patch(input4_box)
ax.text(3.05, 4.1, '🐕🐕', ha='center', fontsize=35)
ax.text(3.05, 3.4, 'Photo 4', ha='center', fontsize=9, style='italic')

# ============================================================================
# CENTER: THE MODEL
# ============================================================================
ax.text(7.0, 6.3, 'DEEP LEARNING MODEL', ha='center', fontsize=13, 
        fontweight='bold', color=COLORS['success'])

# Main model box
model_box = FancyBboxPatch((4.8, 2.8), 4.4, 3.8, boxstyle="round,pad=0.15",
                           edgecolor=COLORS['success'], facecolor=COLORS['light_green'],
                           linewidth=3)
ax.add_patch(model_box)

# Model components
ax.text(7.0, 6.0, '🧠 Neural Network', ha='center', fontsize=12, fontweight='bold')
ax.text(7.0, 5.6, 'EfficientNet-B3', ha='center', fontsize=11, style='italic')

# Three key capabilities
capabilities = [
    ('1️⃣ Detect Dogs', 5.1),
    ('2️⃣ Identify Breed', 4.5),
    ('3️⃣ Measure Confidence', 3.9),
]

for cap_text, y_pos in capabilities:
    ax.text(7.0, y_pos, cap_text, ha='center', fontsize=10, fontweight='bold')

# Training info
ax.text(7.0, 3.3, 'Trained on 48,338 images', ha='center', fontsize=9, style='italic')
ax.text(7.0, 3.05, '93 different dog breeds', ha='center', fontsize=9, style='italic')

# ============================================================================
# ARROWS FROM INPUT TO MODEL
# ============================================================================
# Arrow from inputs to model
arrow_in = FancyArrowPatch((3.8, 4.5), (4.8, 4.5),
                          arrowstyle='->', mutation_scale=35, linewidth=3,
                          color=COLORS['primary'])
ax.add_patch(arrow_in)

# ============================================================================
# RIGHT: OUTPUT EXAMPLES
# ============================================================================
ax.text(11.5, 6.3, 'OUTPUT: Prediction', ha='center', fontsize=13, 
        fontweight='bold', color=COLORS['success'])

# Output 1: Golden Retriever (correct breed)
output1_box = FancyBboxPatch((10.2, 4.8), 2.6, 1.2, boxstyle="round,pad=0.08",
                             edgecolor=COLORS['success'], facecolor=COLORS['light_green'],
                             linewidth=2.5)
ax.add_patch(output1_box)
ax.text(11.5, 5.7, '✓ Golden Retriever', ha='center', fontsize=11, fontweight='bold',
        color=COLORS['success'])
ax.text(11.5, 5.4, 'Confidence: 94%', ha='center', fontsize=9)
ax.text(11.5, 5.1, '(Pure breed)', ha='center', fontsize=8, style='italic')

# Output 2: Mixed breed detected
output2_box = FancyBboxPatch((10.2, 3.5), 2.6, 1.0, boxstyle="round,pad=0.08",
                             edgecolor=COLORS['warning'], facecolor=COLORS['light_gray'],
                             linewidth=2.5)
ax.add_patch(output2_box)
ax.text(11.5, 4.2, '⚠ Unknown Breed', ha='center', fontsize=11, fontweight='bold',
        color=COLORS['warning'])
ax.text(11.5, 3.85, 'Low confidence: 42%', ha='center', fontsize=9)
ax.text(11.5, 3.6, '(Likely mixed)', ha='center', fontsize=8, style='italic')

# Output 3: Not a dog
output3_box = FancyBboxPatch((10.2, 2.3), 2.6, 0.9, boxstyle="round,pad=0.08",
                             edgecolor=COLORS['error'], facecolor=COLORS['light_red'],
                             linewidth=2.5)
ax.add_patch(output3_box)
ax.text(11.5, 2.95, '✗ Not a Dog', ha='center', fontsize=11, fontweight='bold',
        color=COLORS['error'])
ax.text(11.5, 2.6, 'Rejected', ha='center', fontsize=9)
ax.text(11.5, 2.4, '(Cat detected)', ha='center', fontsize=8, style='italic')

# ============================================================================
# ARROWS FROM MODEL TO OUTPUT
# ============================================================================
# Arrow from model to outputs
arrow_out = FancyArrowPatch((9.2, 4.5), (10.2, 4.5),
                           arrowstyle='->', mutation_scale=35, linewidth=3,
                           color=COLORS['success'])
ax.add_patch(arrow_out)

# ============================================================================
# BOTTOM: WHY DEEP LEARNING?
# ============================================================================
why_box = FancyBboxPatch((0.5, 0.2), 13, 1.7, boxstyle="round,pad=0.12",
                         edgecolor='#1565C0', facecolor='#E3F2FD',
                         linewidth=2.5)
ax.add_patch(why_box)

ax.text(7.0, 1.7, '💡 Why Deep Learning?', ha='center', fontsize=13, 
        fontweight='bold', color='#1565C0')

reasons = [
    '✓ Learns visual features automatically (ears, coat, face shape)',
    '✓ Handles 93 breeds with subtle differences',
    '✓ Works with varied photos (lighting, angles, backgrounds)',
    '✓ Provides confidence scores to detect mixed breeds',
]

y_start = 1.35
for i, reason in enumerate(reasons):
    ax.text(7.0, y_start - i*0.25, reason, ha='center', fontsize=9.5)

plt.tight_layout()
plt.savefig('progress_report_figures/figure0_simple_overview.png', dpi=300, bbox_inches='tight')
print("✓ Generated: progress_report_figures/figure0_simple_overview.png")
plt.close()

print("\n" + "="*70)
print("SIMPLE PROJECT OVERVIEW FIGURE GENERATED!")
print("="*70)
print("\n📍 This figure is MUCH simpler and more intuitive:")
print("   • Shows actual dog photos as inputs (with emojis)")
print("   • Shows the model in the middle")
print("   • Shows 3 different outputs: breed identified, mixed breed, not a dog")
print("   • Bottom section explains why deep learning in simple terms")
print("\n📝 Anyone can understand this at a glance!")
print("="*70 + "\n")
