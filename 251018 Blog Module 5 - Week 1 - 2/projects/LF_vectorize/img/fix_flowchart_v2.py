import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set style with math font support
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['mathtext.fontset'] = 'dejavusans'
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create img directory if it doesn't exist
import os
os.makedirs('img', exist_ok=True)

# Figure 8: Flowchart quy trình vector hóa - FIXED VERSION with proper math symbols
def create_flowchart_fixed():
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Define positions with more space between boxes
    positions = [
        (6, 8, 'Compute Output', r'$\hat{y} = X \theta$'),
        (6, 6, 'Compute Loss', r'$L = MSE(\hat{y}, y)$'),
        (6, 4, 'Compute Gradient', r'$\nabla\theta = \frac{2}{N}X^T(\hat{y}-y)$'),
        (6, 2, 'Update Parameters', r'$\theta = \theta - \eta\nabla\theta$')
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    for i, (x, y, title, formula) in enumerate(positions):
        # Draw box with larger size
        ax.add_patch(FancyBboxPatch((x-2, y-0.5), 4, 1, boxstyle="round,pad=0.15", 
                                    facecolor=colors[i], edgecolor='black', linewidth=2))
        
        # Draw title and formula separately
        ax.text(x, y+0.15, title, fontsize=12, ha='center', va='center', fontweight='bold')
        ax.text(x, y-0.15, formula, fontsize=11, ha='center', va='center')
        
        # Draw arrow to next step - FIXED: arrows don't overlap boxes
        if i < len(positions) - 1:
            # Arrow starts below current box and ends above next box
            start_y = y - 0.5  # Bottom of current box
            end_y = positions[i+1][1] + 0.5  # Top of next box
            
            ax.annotate('', xy=(x, end_y), xytext=(x, start_y),
                        arrowprops=dict(arrowstyle='->', lw=3, color='red', alpha=0.8))
    
    ax.set_xlim(2, 10)
    ax.set_ylim(0.5, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Flowchart: Quy trình Vector hóa Gradient Descent', fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('img/figure8_flowchart_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create the fixed flowchart
if __name__ == "__main__":
    print("Creating fixed flowchart with proper math symbols...")
    create_flowchart_fixed()
    print("Fixed flowchart completed!")
