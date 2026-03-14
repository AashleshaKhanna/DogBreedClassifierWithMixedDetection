"""
Generate project overview figure for Brief Project Description.
Shows motivation, input/output, and why deep learning is appropriate.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
from matplotlib.patches import Polygon
import numpy as np

# Set style
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# Colors
COLORS = {
    'primary': '#1976D2',
    'success': '#388E3C',
    'warning': '#F57C00',
    'error': '#D32F2F',
    'info': '#0097A7',
    'light_blue': '#E3F2FD',
    'light_green': '#E8F5E9',
    'light_orange': '#FFF3E0',
    'light_red': '#FFEBEE',
}

fig = plt.figure(figsize=(14, 9))
ax = plt.subplot(111)
ax.set_xlim(0, 14)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(7, 8.5, 'Dog Breed Classifier with Mixed Detection', 
        ha='center', fontsize=16, fontweight='bold')
ax.text(7, 8.1, 'Project Overview: Motivation, Approach, and Impact', 
        ha='center', fontsize=12, style='italic', color='#555')

# ============================================================================
# SECTION 1: MOTIVATION & REAL-WORLD APPLICATIONS (Top)
# ============================================================================
ax.text(7, 7.5, '🎯 Motivation & Real-World Applications', 
        ha='center', fontsize=13, fontweight='bold', color=COLORS['primary'])

# Application boxes
apps = [
    ('Veterinary\nSystems', '🏥', 1.5, 6.2),
    ('Pet Adoption\nPlatforms', '🐕', 4.0, 6.2),
    ('Lost Pet\nRecovery', '🔍', 6.5, 6.2),
    ('Breed\nIdentification', '📱', 9.0, 6.2),
    ('Animal\nShelters', '🏠', 11.5, 6.2),
]

for app_name, emoji, x, y in apps:
    box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.9, boxstyle="round,pad=0.05",
                         edgecolor=COLORS['info'], facecolor=COLORS['light_blue'],
                         linewidth=2)
    ax.add_patch(box)
    ax.text(x, y+0.25, emoji, ha='center', fontsize=20)
    ax.text(x, y-0.15, app_name, ha='center', fontsize=8, fontweight='bold')

# Key challenge box
challenge_box = FancyBboxPatch((0.5, 5.0), 13, 0.8, boxstyle="round,pad=0.1",
                               edgecolor=COLORS['warning'], facecolor=COLORS['light_orange'],
                               linewidth=2.5, linestyle='--')
ax.add_patch(challenge_box)
ax.text(7, 5.5, '⚠️ Challenge: Must handle 93 breeds + reject unknown dogs/non-dogs with reliable confidence', 
        ha='center', fontsize=10, fontweight='bold', color=COLORS['warning'])
ax.text(7, 5.15, 'Traditional methods fail: Too many classes, high visual similarity, need confidence calibration', 
        ha='center', fontsize=9, style='italic')

# ============================================================================
# SECTION 2: INPUT → MODEL → OUTPUT (Middle)
# ============================================================================
ax.text(7, 4.4, '🔄 System Input/Output Flow', 
        ha='center', fontsize=13, fontweight='bold', color=COLORS['primary'])

# INPUT box
input_box = FancyBboxPatch((0.5, 2.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['primary'], facecolor=COLORS['light_blue'],
                           linewidth=3)
ax.add_patch(input_box)
ax.text(1.75, 3.7, 'INPUT', ha='center', fontsize=11, fontweight='bold', 
        color=COLORS['primary'])
ax.text(1.75, 3.35, '📷 Any Image', ha='center', fontsize=10, fontweight='bold')
ax.text(1.75, 3.0, '• Dog photos\n• Multiple dogs\n• Varied quality\n• Any background', 
        ha='center', fontsize=8)

# Arrow 1
arrow1 = FancyArrowPatch((3.0, 3.25), (4.2, 3.25),
                        arrowstyle='->', mutation_scale=35, linewidth=3,
                        color=COLORS['primary'])
ax.add_patch(arrow1)

# MODEL box (Deep Learning)
model_box = FancyBboxPatch((4.2, 2.0), 5.6, 2.5, boxstyle="round,pad=0.1",
                           edgecolor=COLORS['success'], facecolor=COLORS['light_green'],
                           linewidth=3)
ax.add_patch(model_box)
ax.text(7.0, 4.2, 'DEEP LEARNING MODEL', ha='center', fontsize=11, 
        fontweight='bold', color=COLORS['success'])

# Three stages inside model box
stage_y = 3.6
stages = [
    ('Stage 1: YOLOv5\nDog Detection', 5.0),
    ('Stage 2: EfficientNet-B3\nBreed Classification', 7.0),
    ('Stage 3: Temperature\nCalibration', 9.0),
]

for stage_text, stage_x in stages:
    stage_box = FancyBboxPatch((stage_x-0.7, stage_y-0.35), 1.4, 0.7, 
                               boxstyle="round,pad=0.05",
                               edgecolor='#2E7D32', facecolor='white',
                               linewidth=1.5)
    ax.add_patch(stage_box)
    ax.text(stage_x, stage_y, stage_text, ha='center', fontsize=7, 
            fontweight='bold')
    
    if stage_x < 9.0:
        arrow = FancyArrowPatch((stage_x+0.7, stage_y), (stage_x+1.3, stage_y),
                               arrowstyle='->', mutation_scale=15, linewidth=1.5,
                               color='#2E7D32')
        ax.add_patch(arrow)

# Why Deep Learning box
why_box = FancyBboxPatch((4.4, 2.2), 5.2, 1.0, boxstyle="round,pad=0.05",
                         edgecolor='#1B5E20', facecolor='#C8E6C9',
                         linewidth=1.5, linestyle='--')
ax.add_patch(why_box)
ax.text(7.0, 2.95, '✓ Why Deep Learning?', ha='center', fontsize=9, 
        fontweight='bold', color='#1B5E20')
ax.text(7.0, 2.65, '• Learns complex visual features automatically', ha='center', fontsize=7)
ax.text(7.0, 2.45, '• Handles 93 classes with high similarity', ha='center', fontsize=7)
ax.text(7.0, 2.25, '• Transfer learning from ImageNet (1000 classes)', ha='center', fontsize=7)

# Arrow 2
arrow2 = FancyArrowPatch((9.8, 3.25), (11.0, 3.25),
                        arrowstyle='->', mutation_scale=35, linewidth=3,
                        color=COLORS['primary'])
ax.add_patch(arrow2)

# OUTPUT box
output_box = FancyBboxPatch((11.0, 2.5), 2.5, 1.5, boxstyle="round,pad=0.1",
                            edgecolor=COLORS['success'], facecolor=COLORS['light_green'],
                            linewidth=3)
ax.add_patch(output_box)
ax.text(12.25, 3.7, 'OUTPUT', ha='center', fontsize=11, fontweight='bold',
        color=COLORS['success'])
ax.text(12.25, 3.35, '✅ Prediction', ha='center', fontsize=10, fontweight='bold')
ax.text(12.25, 3.0, '• Breed name\n• Confidence %\n• OR "Unknown"\n• OR "Not a dog"', 
        ha='center', fontsize=8)

# ============================================================================
# SECTION 3: KEY INNOVATIONS (Bottom)
# ============================================================================
ax.text(7, 1.6, '💡 Key Innovations & Impact', 
        ha='center', fontsize=13, fontweight='bold', color=COLORS['primary'])

innovations = [
    ('Multi-Source\nData', '48K images\n2 datasets', 1.8, 0.5),
    ('Multi-Stage\nPipeline', 'Detection +\nClassification', 4.5, 0.5),
    ('Confidence\nCalibration', 'Reliable\nprobabilities', 7.2, 0.5),
    ('Unknown\nDetection', 'Rejects OOD\nimages', 9.9, 0.5),
    ('93 Breeds', 'Comprehensive\ncoverage', 12.4, 0.5),
]

for title, desc, x, y in innovations:
    # Circle background
    circle = Circle((x, y+0.35), 0.5, facecolor=COLORS['light_blue'], 
                   edgecolor=COLORS['primary'], linewidth=2)
    ax.add_patch(circle)
    
    # Text
    ax.text(x, y+0.35, title, ha='center', va='center', fontsize=8, 
            fontweight='bold', color=COLORS['primary'])
    ax.text(x, y-0.15, desc, ha='center', fontsize=7, style='italic')

plt.tight_layout()
plt.savefig('progress_report_figures/figure0_project_overview.png', dpi=300, bbox_inches='tight')
print("✓ Generated: progress_report_figures/figure0_project_overview.png")
plt.close()

print("\n" + "="*70)
print("PROJECT OVERVIEW FIGURE GENERATED!")
print("="*70)
print("\n📍 PLACEMENT: At the very beginning of 'Brief Project Description'")
print("\n📝 CAPTION:")
print("   'Project overview showing real-world applications, system")
print("    input/output flow, and key innovations. The system takes any")
print("    image as input, processes it through a three-stage deep learning")
print("    pipeline, and outputs breed predictions with calibrated confidence")
print("    or rejects unknown/non-dog images.'")
print("="*70 + "\n")
