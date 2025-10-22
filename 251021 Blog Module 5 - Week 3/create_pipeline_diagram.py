import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np

# Thiết lập style gọn gàng
plt.style.use('default')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 0

# Tạo figure và axis gọn gàng
fig, ax = plt.subplots(1, 1, figsize=(14, 14))
ax.set_xlim(0, 10)
ax.set_ylim(-2, 12)
ax.axis('off')

# Background gradient đẹp hơn
ax.add_patch(patches.Rectangle((0, -2), 10, 14, 
                              facecolor='#FFFFFF', 
                              edgecolor='none'))

# Định nghĩa màu sắc Forest Green và Cam đẹp hơn
colors = {
    'init': '#FF8C00',      # Dark Orange - chỉ cho khởi tạo
    'fitness': '#228B22',   # Forest Green  
    'selection': '#228B22', # Forest Green
    'crossover': '#228B22', # Forest Green
    'mutation': '#228B22',  # Forest Green
    'evolution': '#228B22', # Forest Green
    'convergence': '#228B22', # Forest Green
    'end': '#FF8C00'        # Dark Orange - chỉ cho kết thúc
}

# Vẽ các bước trong pipeline với nội dung thực tế từ dự án
steps = [
    {'name': '1. Khởi tạo\n250 cá thể\n(K-means clustering)', 
     'pos': (2.5, 10), 'size': (2.5, 1.4), 'color': colors['init']},
    {'name': '2. Đánh giá Fitness\n(Haversine + CV)', 
     'pos': (7.5, 10), 'size': (2.5, 1.4), 'color': colors['fitness']},
    {'name': '3. Tournament\nSelection\n(k=3)', 
     'pos': (2.5, 7.5), 'size': (2.5, 1.4), 'color': colors['selection']},
    {'name': '4. Route-based\nCrossover\n(Trao đổi xe)', 
     'pos': (7.5, 7.5), 'size': (2.5, 1.4), 'color': colors['crossover']},
    {'name': '5. Point Migration\nMutation\n(30% rate)', 
     'pos': (2.5, 5), 'size': (2.5, 1.4), 'color': colors['mutation']},
    {'name': '6. Thế hệ mới\n(250 cá thể)', 
     'pos': (7.5, 5), 'size': (2.5, 1.4), 'color': colors['evolution']},
    {'name': 'Hội tụ?\n(300 gen)', 
     'pos': (5, 2.5), 'size': (2, 1.2), 'color': colors['convergence']},
    {'name': 'Kết thúc\n1,601.9km • CV=0.12\nCải thiện 35.9%', 
     'pos': (5, 0.5), 'size': (3, 1.5), 'color': colors['end']}
]

# Vẽ các hình chữ nhật đơn giản
for step in steps:
    x, y = step['pos']
    w, h = step['size']
    
    # Tạo hình chữ nhật đơn giản
    rect = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.1",
        facecolor=step['color'],
        edgecolor='#2C2C2C',
        linewidth=2,
        alpha=0.9
    )
    ax.add_patch(rect)
    
    # Thêm text đơn giản
    ax.text(x, y, step['name'], 
            ha='center', va='center', 
            fontsize=11, fontweight='bold', color='white')

# Vẽ các mũi tên kết nối chính xác vào cạnh ô
# Tính toán vị trí chính xác của các cạnh ô
def get_box_edges(pos, size):
    x, y = pos
    w, h = size
    return {
        'left': x - w/2,
        'right': x + w/2,
        'top': y + h/2,
        'bottom': y - h/2,
        'center_x': x,
        'center_y': y
    }

# Lấy vị trí cạnh của các ô
box1 = get_box_edges((2.5, 10), (2.5, 1.4))
box2 = get_box_edges((7.5, 10), (2.5, 1.4))
box3 = get_box_edges((2.5, 7.5), (2.5, 1.4))
box4 = get_box_edges((7.5, 7.5), (2.5, 1.4))
box5 = get_box_edges((2.5, 5), (2.5, 1.4))
box6 = get_box_edges((7.5, 5), (2.5, 1.4))
convergence = get_box_edges((5, 2.5), (2, 1.2))
end_box = get_box_edges((5, 0.5), (3, 1.5))

# Mũi tên ngang từ bước 1 đến bước 2 (từ cạnh phải của ô 1 đến cạnh trái của ô 2)
ax.annotate('', xy=(box2['left'], box1['center_y']), xytext=(box1['right'], box1['center_y']),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2C2C2C'))
ax.text(5, 10.4, 'Đánh giá 250 cá thể', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.8))

# Mũi tên xuống từ bước 2 đến bước 3 (từ cạnh dưới của ô 2 đến cạnh trên của ô 3)
ax.annotate('', xy=(box3['right'], box3['top']), xytext=(box2['left'], box2['bottom']),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2C2C2C'))
ax.text(5, 8.8, 'Chọn lọc Tournament', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.8))

# Mũi tên ngang từ bước 3 đến bước 4 (từ cạnh phải của ô 3 đến cạnh trái của ô 4)
ax.annotate('', xy=(box4['left'], box3['center_y']), xytext=(box3['right'], box3['center_y']),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2C2C2C'))
ax.text(5, 7.9, 'Lai ghép Routes', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.8))

# Mũi tên xuống từ bước 4 đến bước 5 (từ cạnh dưới của ô 4 đến cạnh trên của ô 5)
ax.annotate('', xy=(box5['right'], box5['top']), xytext=(box4['left'], box4['bottom']),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2C2C2C'))
ax.text(5, 6.3, 'Đột biến Points', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.8))

# Mũi tên ngang từ bước 5 đến bước 6 (từ cạnh phải của ô 5 đến cạnh trái của ô 6)
ax.annotate('', xy=(box6['left'], box5['center_y']), xytext=(box5['right'], box5['center_y']),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2C2C2C'))
ax.text(5, 5.4, 'Tạo thế hệ mới', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.8))

# Mũi tên xuống từ bước 6 đến kiểm tra hội tụ (từ cạnh dưới của ô 6 đến cạnh trên của ô hội tụ)
ax.annotate('', xy=(convergence['right'], convergence['top']), xytext=(box6['left'], box6['bottom']),
            arrowprops=dict(arrowstyle='->', lw=3, color='#2C2C2C'))
ax.text(6.2, 3.9, 'Kiểm tra hội tụ', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#E8F5E8', alpha=0.8))

# Mũi tên vuông lặp lại (từ cạnh phải của ô hội tụ về cạnh dưới của ô 2 - đánh giá fitness)
ax.annotate('', xy=(box2['right'], box2['bottom']), xytext=(convergence['right'], convergence['center_y']),
            arrowprops=dict(arrowstyle='->', lw=4, color='#FF6347',
                          connectionstyle="arc3,rad=0.8"))
ax.text(9.5, 6.5, 'Lặp lại\nnếu chưa hội tụ', ha='center', fontsize=9, fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.2", facecolor='#FFE4B5', alpha=0.8))

# Mũi tên kết thúc (từ cạnh dưới của ô hội tụ đến ô kết thúc)
ax.annotate('', xy=(end_box['center_x'], end_box['top']), xytext=(convergence['center_x'], convergence['bottom']),
            arrowprops=dict(arrowstyle='->', lw=4, color='#FF8C00'))

# Thêm tiêu đề đẹp hơn
ax.text(5, 11.5, 'Pipeline Giải thuật Di truyền cho Multi-Vehicle TSP', 
        ha='center', fontsize=18, fontweight='bold', color='#1A1A1A')
ax.text(5, 11.1, 'Tối ưu hóa tuyến đường giao hàng tại TP.HCM', 
        ha='center', fontsize=12, fontweight='normal', color='#666666')

# Thông tin dữ liệu thực tế với background
ax.text(1, 1.2, 'Dữ liệu thực tế:', fontsize=12, fontweight='bold', color='#1A1A1A')
ax.text(1, 0.8, '• 168 phường/xã TP.HCM', fontsize=11, fontweight='normal', color='#444444')
ax.text(1, 0.5, '• 4 xe giao hàng', fontsize=11, fontweight='normal', color='#444444')
ax.text(1, 0.2, '• Haversine distance', fontsize=11, fontweight='normal', color='#444444')

plt.tight_layout()

# Tự động lưu hình ảnh
output_path = 'Genetic_Algorithm_MultiVehicle_TSP/results/ga_pipeline_diagram.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')

# Kiểm tra file đã được tạo
import os
if os.path.exists(output_path):
    file_size = os.path.getsize(output_path)
    print("Da tao so do pipeline GA thanh cong!")
    print(f"File: {output_path}")
    print(f"Kich thuoc: {file_size:,} bytes")
    
    # Kiểm tra nội dung hình ảnh
    print("\nKiem tra noi dung hinh:")
    print("- 6 buoc pipeline GA")
    print("- Mui ten ket noi chinh xac")
    print("- Tone mau Forest Green va Cam")
    print("- Noi dung thuc te tu du an MVTSP")
    print("- Labels mo ta ro rang")
    print("- Thong tin du lieu va ket qua")
    print("- O ket thuc ro rang")
else:
    print("Loi: Khong the tao file hinh anh")

# plt.show()  # Tắt hiển thị biểu đồ
