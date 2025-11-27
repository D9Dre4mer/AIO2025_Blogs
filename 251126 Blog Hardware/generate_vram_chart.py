"""
Script tạo biểu đồ VRAM Requirements cho các mô hình AI
Dựa trên dữ liệu thực tế từ các mô hình LLM phổ biến
"""

import matplotlib.pyplot as plt
import numpy as np

# Dữ liệu thực tế dựa trên công thức tính VRAM
# Training FP32: Model (4B) + Gradients (4B) + Optimizer (4B) = 12 bytes/parameter
# Training FP16: Model (2B) + Gradients (2B) + Optimizer (2B) = 6 bytes/parameter
# Inference FP16: Chỉ model weights = 2 bytes/parameter
# Inference INT8: Chỉ model weights = 1 byte/parameter

model_sizes = ['1B', '7B', '13B', '70B']
model_sizes_numeric = [1, 7, 13, 70]  # Để tính toán

# Tính VRAM requirements (theo GB)
training_fp32 = [size * 12 / 1024**3 * 1e9 for size in model_sizes_numeric]  # 12 bytes per parameter
training_fp16 = [size * 6 / 1024**3 * 1e9 for size in model_sizes_numeric]  # 6 bytes per parameter
inference_fp16 = [size * 2 / 1024**3 * 1e9 for size in model_sizes_numeric]  # 2 bytes per parameter
inference_int8 = [size * 1 / 1024**3 * 1e9 for size in model_sizes_numeric]  # 1 byte per parameter

# Làm tròn đến số nguyên
training_fp32 = [round(x) for x in training_fp32]
training_fp16 = [round(x) for x in training_fp16]
inference_fp16 = [round(x) for x in inference_fp16]
inference_int8 = [round(x) for x in inference_int8]

# Giới hạn VRAM của các GPU
gpu_limits = {
    'RTX 5090': 32,
    'A100 40GB': 40,
    'A100/H100 80GB': 80
}

# Tạo figure với kích thước lớn
fig, ax = plt.subplots(figsize=(14, 8))

# Vị trí của các nhóm
x = np.arange(len(model_sizes))
width = 0.2  # Độ rộng của mỗi thanh

# Vẽ các thanh
bars1 = ax.bar(x - 1.5*width, training_fp32, width, label='Training FP32', color='#DC3545', alpha=0.9)
bars2 = ax.bar(x - 0.5*width, training_fp16, width, label='Training FP16', color='#FF9800', alpha=0.9)
bars3 = ax.bar(x + 0.5*width, inference_fp16, width, label='Inference FP16', color='#FFC107', alpha=0.9)
bars4 = ax.bar(x + 1.5*width, inference_int8, width, label='Inference INT8', color='#28A745', alpha=0.9)

# Vẽ các đường giới hạn VRAM
ax.axhline(y=gpu_limits['RTX 5090'], color='#1E88E5', linestyle='--', linewidth=2, 
           label=f"RTX 5090 ({gpu_limits['RTX 5090']}GB)", alpha=0.7)
ax.axhline(y=gpu_limits['A100 40GB'], color='#FFC107', linestyle='--', linewidth=2, 
           label=f"A100 40GB ({gpu_limits['A100 40GB']}GB)", alpha=0.7)
ax.axhline(y=gpu_limits['A100/H100 80GB'], color='#DC3545', linestyle='--', linewidth=2, 
           label=f"A100/H100 80GB ({gpu_limits['A100/H100 80GB']}GB)", alpha=0.7)

# Thêm nhãn số liệu trên mỗi thanh
def add_value_labels(bars):
    for bar in bars:
        height = bar.get_height()
        # Chỉ hiển thị nhãn nếu thanh không quá cao (tránh cluttered)
        if height < 200:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}GB',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')
        else:
            # Với thanh quá cao, hiển thị ở trên cùng
            ax.text(bar.get_x() + bar.get_width()/2., 200,
                   f'{int(height)}GB',
                   ha='center', va='bottom', fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))

add_value_labels(bars1)
add_value_labels(bars2)
add_value_labels(bars3)
add_value_labels(bars4)

# Cấu hình trục
ax.set_xlabel('Kích thước mô hình (Parameters)', fontsize=12, fontweight='bold')
ax.set_ylabel('VRAM cần thiết (GB)', fontsize=12, fontweight='bold')
ax.set_title('VRAM Requirements theo Kích thước Mô hình và Độ chính xác', 
             fontsize=14, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(model_sizes)
ax.set_ylim(0, 220)  # Giới hạn trục Y đến 220GB để hiển thị break mark cho 70B

# Thêm grid
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_axisbelow(True)

# Thêm legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)

# Thêm break mark cho mô hình 70B (vì vượt quá 200GB)
# Vẽ một đường zigzag để chỉ break
break_y = 200
break_x = x[-1]  # Vị trí của 70B
ax.plot([break_x - 0.3, break_x - 0.1, break_x + 0.1, break_x + 0.3], 
        [break_y, break_y + 5, break_y, break_y + 5],
        color='black', linewidth=2)

# Thêm text giải thích
ax.text(0.02, 0.98, 
        'Lưu ý: Training cần thêm VRAM cho gradients và optimizer states.\n'
        'Inference chỉ cần model weights.',
        transform=ax.transAxes, fontsize=9,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Điều chỉnh layout
plt.tight_layout()

# Lưu hình ảnh
plt.savefig('vram_vs_model_size.png', dpi=300, bbox_inches='tight')
plt.savefig('vram_vs_model_size.pdf', bbox_inches='tight')

import sys
import io

# Set UTF-8 encoding for output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

print("Chart saved successfully!")
print(f"- vram_vs_model_size.png (300 DPI)")
print(f"- vram_vs_model_size.pdf")
print("\nData:")
print(f"Training FP32: {training_fp32} GB")
print(f"Training FP16: {training_fp16} GB")
print(f"Inference FP16: {inference_fp16} GB")
print(f"Inference INT8: {inference_int8} GB")

# Hiển thị biểu đồ
plt.show()

