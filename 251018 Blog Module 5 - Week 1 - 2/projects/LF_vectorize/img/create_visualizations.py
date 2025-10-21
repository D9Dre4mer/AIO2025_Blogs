import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Create img directory if it doesn't exist
import os
os.makedirs('img', exist_ok=True)

# Figure 1: Scatter plot với regression line (area vs price)
def create_scatter_plot():
    plt.figure(figsize=(10, 6))
    
    # Generate sample data
    np.random.seed(42)
    area = np.random.uniform(30, 300, 50)
    noise = np.random.normal(0, 20, 50)
    price = 0.8 * area + 50 + noise
    
    # Plot scatter
    plt.scatter(area, price, alpha=0.6, color='skyblue', s=60, edgecolors='navy', linewidth=0.5)
    
    # Add regression line
    z = np.polyfit(area, price, 1)
    p = np.poly1d(z)
    plt.plot(area, p(area), "r--", alpha=0.8, linewidth=2, label=f'Regression Line: y = {z[0]:.2f}x + {z[1]:.1f}')
    
    plt.xlabel('Diện tích (m²)', fontsize=12)
    plt.ylabel('Giá nhà (triệu VND)', fontsize=12)
    plt.title('Linear Regression: Dự đoán giá nhà theo diện tích', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/figure1_scatter_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 2: MSE curve (parabol)
def create_mse_curve():
    plt.figure(figsize=(10, 6))
    
    error = np.linspace(-5, 5, 100)
    mse = error**2
    
    plt.plot(error, mse, 'b-', linewidth=3, label='MSE = (ŷ - y)²')
    
    # Highlight outlier effect
    plt.axvline(x=3, color='red', linestyle='--', alpha=0.7, label='Outlier tại x=3')
    plt.axvline(x=-3, color='red', linestyle='--', alpha=0.7)
    plt.annotate('Outlier làm tăng loss nhanh chóng', 
                xy=(3, 9), xytext=(4, 15),
                arrowprops=dict(arrowstyle='->', color='red', alpha=0.7),
                fontsize=10, color='red')
    
    plt.xlabel('Sai số (ŷ - y)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Mean Squared Error (MSE) Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/figure2_mse_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 3: MAE curve (V-shape)
def create_mae_curve():
    plt.figure(figsize=(10, 6))
    
    error = np.linspace(-5, 5, 100)
    mae = np.abs(error)
    
    plt.plot(error, mae, 'g-', linewidth=3, label='MAE = |ŷ - y|')
    
    plt.xlabel('Sai số (ŷ - y)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Mean Absolute Error (MAE) Curve', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/figure3_mae_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 4: Huber Loss curve
def create_huber_curve():
    plt.figure(figsize=(10, 6))
    
    error = np.linspace(-5, 5, 100)
    delta = 1.0
    
    def huber_loss(err, delta):
        return np.where(np.abs(err) <= delta, 
                       0.5 * err**2, 
                       delta * np.abs(err) - 0.5 * delta**2)
    
    huber = huber_loss(error, delta)
    
    plt.plot(error, huber, 'purple', linewidth=3, label=f'Huber Loss (δ={delta})')
    
    # Mark delta points
    plt.axvline(x=delta, color='orange', linestyle='--', alpha=0.7, label=f'δ = {delta}')
    plt.axvline(x=-delta, color='orange', linestyle='--', alpha=0.7)
    
    plt.xlabel('Sai số (ŷ - y)', fontsize=12)
    plt.ylabel('Loss Value', fontsize=12)
    plt.title('Huber Loss: Kết hợp MSE và MAE', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/figure4_huber_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 5: Histogram trước và sau normalization
def create_normalization():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Original data
    np.random.seed(42)
    original_data = np.random.uniform(30, 300, 1000)
    
    ax1.hist(original_data, bins=30, alpha=0.7, color='skyblue', edgecolor='navy')
    ax1.set_xlabel('Diện tích (m²)', fontsize=12)
    ax1.set_ylabel('Tần số', fontsize=12)
    ax1.set_title('Trước Normalization\n(30-300 m²)', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Normalized data
    normalized_data = (original_data - original_data.min()) / (original_data.max() - original_data.min())
    
    ax2.hist(normalized_data, bins=30, alpha=0.7, color='lightcoral', edgecolor='darkred')
    ax2.set_xlabel('Giá trị chuẩn hóa', fontsize=12)
    ax2.set_ylabel('Tần số', fontsize=12)
    ax2.set_title('Sau Normalization\n(0-1)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('img/figure5_normalization.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 6: Overfit vs Regularized regression
def create_overfitting():
    plt.figure(figsize=(12, 6))
    
    # Generate data
    np.random.seed(42)
    x = np.linspace(0, 10, 20)
    y_true = 2 * x + 1
    noise = np.random.normal(0, 2, 20)
    y_noisy = y_true + noise
    
    # Overfit line (high degree polynomial)
    poly_overfit = np.polyfit(x, y_noisy, 15)
    p_overfit = np.poly1d(poly_overfit)
    x_smooth = np.linspace(0, 10, 100)
    
    # Regularized line (simple linear)
    poly_reg = np.polyfit(x, y_noisy, 1)
    p_reg = np.poly1d(poly_reg)
    
    plt.scatter(x, y_noisy, color='blue', alpha=0.6, s=60, label='Dữ liệu huấn luyện')
    plt.plot(x_smooth, p_overfit(x_smooth), 'r-', linewidth=2, alpha=0.8, label='Overfit (quá sát dữ liệu)')
    plt.plot(x_smooth, p_reg(x_smooth), 'g-', linewidth=2, alpha=0.8, label='Regularized (mượt hơn)')
    
    plt.xlabel('X', fontsize=12)
    plt.ylabel('Y', fontsize=12)
    plt.title('So sánh Overfitting vs Regularization', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/figure6_overfitting.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 7: Vectorization matrix multiplication diagram
def create_vectorization():
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Matrix X
    ax.add_patch(FancyBboxPatch((1, 5), 2, 3, boxstyle="round,pad=0.1", 
                                facecolor='lightblue', edgecolor='navy', linewidth=2))
    ax.text(2, 6.5, 'X', fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(2, 6, '[x₁ 1]', fontsize=10, ha='center', va='center')
    ax.text(2, 5.5, '[x₂ 1]', fontsize=10, ha='center', va='center')
    ax.text(2, 5, '[⋮ ⋮]', fontsize=10, ha='center', va='center')
    
    # Matrix theta
    ax.add_patch(FancyBboxPatch((5, 6), 1.5, 1, boxstyle="round,pad=0.1", 
                                facecolor='lightgreen', edgecolor='darkgreen', linewidth=2))
    ax.text(5.75, 6.5, 'θ', fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(5.75, 6, '[w]', fontsize=10, ha='center', va='center')
    ax.text(5.75, 5.5, '[b]', fontsize=10, ha='center', va='center')
    
    # Arrow
    ax.annotate('', xy=(5, 6.5), xytext=(3, 6.5),
                arrowprops=dict(arrowstyle='->', lw=3, color='red'))
    
    # Result matrix
    ax.add_patch(FancyBboxPatch((8, 5), 2, 3, boxstyle="round,pad=0.1", 
                                facecolor='lightcoral', edgecolor='darkred', linewidth=2))
    ax.text(9, 6.5, 'ŷ', fontsize=16, fontweight='bold', ha='center', va='center')
    ax.text(9, 6, '[ŷ₁]', fontsize=10, ha='center', va='center')
    ax.text(9, 5.5, '[ŷ₂]', fontsize=10, ha='center', va='center')
    ax.text(9, 5, '[⋮]', fontsize=10, ha='center', va='center')
    
    # Labels
    ax.text(2, 4, 'Ma trận dữ liệu', fontsize=12, ha='center', va='center')
    ax.text(5.75, 4.5, 'Tham số', fontsize=12, ha='center', va='center')
    ax.text(9, 4, 'Dự đoán', fontsize=12, ha='center', va='center')
    
    ax.set_xlim(0, 12)
    ax.set_ylim(3, 9)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Vectorization: Phép nhân ma trận X @ θ → ŷ', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('img/figure7_vectorization.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 8: Flowchart quy trình vector hóa
def create_flowchart():
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Define positions
    positions = [
        (5, 7, 'Compute Output\nŷ = X @ θ'),
        (5, 5.5, 'Compute Loss\nL = MSE(ŷ, y)'),
        (5, 4, 'Compute Gradient\n∇θ = (2/N)X^T(ŷ-y)'),
        (5, 2.5, 'Update Parameters\nθ = θ - η∇θ')
    ]
    
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    
    for i, (x, y, text) in enumerate(positions):
        # Draw box
        ax.add_patch(FancyBboxPatch((x-1.5, y-0.4), 3, 0.8, boxstyle="round,pad=0.1", 
                                    facecolor=colors[i], edgecolor='black', linewidth=2))
        ax.text(x, y, text, fontsize=10, ha='center', va='center', fontweight='bold')
        
        # Draw arrow to next step
        if i < len(positions) - 1:
            ax.annotate('', xy=(x, y-0.4), xytext=(x, y+0.4),
                        arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    
    ax.set_xlim(2, 8)
    ax.set_ylim(1, 8)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Flowchart: Quy trình Vector hóa Gradient Descent', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('img/figure8_flowchart.png', dpi=300, bbox_inches='tight')
    plt.close()

# Figure 9: Loss curve giảm dần theo epoch
def create_loss_curve():
    plt.figure(figsize=(10, 6))
    
    # Simulate loss curve
    epochs = np.arange(0, 1000, 10)
    loss = 100 * np.exp(-epochs/200) + 5 + np.random.normal(0, 0.5, len(epochs))
    
    plt.plot(epochs, loss, 'b-', linewidth=2, alpha=0.8)
    plt.fill_between(epochs, loss, alpha=0.3, color='lightblue')
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Loss Curve: Giảm dần theo số epoch', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('img/figure9_loss_curve.png', dpi=300, bbox_inches='tight')
    plt.close()

# Create all figures
if __name__ == "__main__":
    print("Creating visualizations...")
    
    create_scatter_plot()
    print("Figure 1: Scatter plot completed")
    
    create_mse_curve()
    print("Figure 2: MSE curve completed")
    
    create_mae_curve()
    print("Figure 3: MAE curve completed")
    
    create_huber_curve()
    print("Figure 4: Huber Loss curve completed")
    
    create_normalization()
    print("Figure 5: Normalization histogram completed")
    
    create_overfitting()
    print("Figure 6: Overfitting comparison completed")
    
    create_vectorization()
    print("Figure 7: Vectorization diagram completed")
    
    create_flowchart()
    print("Figure 8: Flowchart completed")
    
    create_loss_curve()
    print("Figure 9: Loss curve completed")
    
    print("\nAll visualizations created and saved to img/ folder")
