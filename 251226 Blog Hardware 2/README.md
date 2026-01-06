# Notebooks Chứng minh Cross-GPU Reproducibility Issues

Bộ notebook này được thiết kế để chứng minh và test các vấn đề về reproducibility khi train mô hình deep learning trên các GPU khác nhau (T4, L4, A100, RTX 3060).

## 📁 Cấu trúc Notebooks

### 1. `01_demonstrate_gpu_differences.ipynb`
**Mục tiêu**: Chứng minh rằng cùng mô hình, cùng seed nhưng kết quả khác nhau trên các GPU khác nhau.

**Nội dung**:
- Train mô hình CNN trên CIFAR-10 với cùng seed
- Test với các cấu hình khác nhau (TF32 on/off, num_workers)
- So sánh và visualize kết quả
- Lưu kết quả vào file JSON trong thư mục `results/`

**Cách sử dụng**:
- Chạy notebook này trên các GPU khác nhau (T4, L4, A100, RTX 3060)
- Mỗi lần chạy sẽ tạo file JSON với thông tin GPU và cấu hình

### 2. `02_test_deterministic_configs.ipynb`
**Mục tiêu**: Test các phương pháp khắc phục vấn đề reproducibility.

**Nội dung**:
- Test với TF32 tắt/bật
- Test với cuDNN benchmark tắt/bật
- Test với num_workers = 0
- Test với deterministic algorithms
- Test với AMP (Automatic Mixed Precision)
- So sánh tất cả các cấu hình

**Kết quả**: Xác định cấu hình nào giúp reproducibility tốt nhất

### 3. `03_test_optimizers_sensitivity.ipynb`
**Mục tiêu**: Chứng minh độ nhạy của optimizer với reproducibility.

**Nội dung**:
- Test với các optimizer khác nhau (Adam, RMSProp, SGD)
- Chứng minh adaptive optimizers khuếch đại sai số nhỏ
- So sánh kết quả giữa các optimizer
- Test độ nhạy bằng cách chạy nhiều lần với cùng seed

### 4. `04_analyze_cross_gpu_results.ipynb`
**Mục tiêu**: Phân tích và so sánh kết quả từ nhiều GPU/TPU.

**Nội dung**:
- Load tất cả file JSON từ thư mục `results/`
- Phân loại kết quả theo GPU/TPU và cấu hình
- So sánh kết quả giữa các GPU/TPU (cùng cấu hình)
- Phân tích Statistical vs Bit-exact Reproducibility
- Tạo bảng tổng hợp và visualization

**Cách sử dụng**:
1. Chạy notebook 01, 02 và 03 trên các GPU/TPU khác nhau
2. Copy tất cả file JSON vào thư mục `results/`
3. Chạy notebook này để phân tích

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu trúc thư mục

Sau khi chạy, cấu trúc thư mục sẽ như sau:

```
.
├── 01_demonstrate_gpu_differences.ipynb
├── 02_test_deterministic_configs.ipynb
├── 03_test_optimizers_sensitivity.ipynb
├── 04_analyze_cross_gpu_results.ipynb
├── requirements.txt
├── README.md
├── data/                    # CIFAR-10 dataset (tự động download)
└── results/                 # Kết quả training (JSON files và plots)
    ├── train_*.json
    ├── deterministic_*.json
    ├── optimizer_*.json
    ├── comparison.png
    ├── summary_all_results.csv
    └── ...
```

## 📊 Quy trình Test

### Bước 1: Chạy trên GPU đầu tiên
1. Mở `01_demonstrate_gpu_differences.ipynb`
2. Chạy tất cả cells
3. Kết quả sẽ được lưu vào `results/train_<GPU_NAME>_*.json`

### Bước 2: Chạy trên GPU thứ hai (và các GPU khác)
1. Copy notebook `01_demonstrate_gpu_differences.ipynb` sang hệ thống khác
2. Chạy lại notebook (hoặc chỉ cần chạy phần training)
3. Copy file JSON kết quả về thư mục `results/` chung

### Bước 3: Phân tích kết quả
1. Mở `04_analyze_cross_gpu_results.ipynb`
2. Chạy tất cả cells để:
   - Load tất cả kết quả
   - So sánh giữa các GPU
   - Phân tích reproducibility
   - Tạo visualization và bảng tổng hợp

### Bước 4: Test các cấu hình deterministic
1. Mở `02_test_deterministic_configs.ipynb`
2. Chạy tất cả cells để test các cấu hình khác nhau
3. Xem kết quả so sánh để xác định cấu hình tốt nhất

### Bước 5: Test độ nhạy optimizer
1. Mở `03_test_optimizers_sensitivity.ipynb`
2. Chạy để thấy cách optimizer ảnh hưởng đến reproducibility

## 📈 Kết quả mong đợi

### Statistical Reproducibility (Chấp nhận được)
- Test accuracy lệch < 0.5%: ✅ TỐT
- Test accuracy lệch 0.5-2%: ⚠️ CHẤP NHẬN ĐƯỢC (nếu xu hướng giống nhau)
- Correlation giữa các trajectory > 0.95

### Vấn đề nghiêm trọng
- Test accuracy lệch > 2%: ❌ CẦN KIỂM TRA
- Quỹ đạo học khác hẳn (một cái overfit, một cái underfit)
- Correlation < 0.95

## 🔧 Cấu hình Deterministic (Khuyến nghị)

Để reproducibility tốt nhất:
```python
torch.backends.cudnn.allow_tf32 = False
torch.backends.cublas.allow_tf32 = False
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# num_workers = 0 trong DataLoader
```

**Lưu ý**: Cấu hình này có thể làm chậm training nhưng giúp reproducibility tốt hơn.

## 📝 Ghi chú

- Tất cả notebook sử dụng dataset CIFAR-10 (tự động download)
- Kết quả được lưu dưới dạng JSON để dễ phân tích sau
- Visualization được lưu vào thư mục `results/`
- Mỗi notebook có thể chạy độc lập hoặc theo thứ tự

## 🎯 Mục tiêu của bộ notebook

1. **Chứng minh vấn đề**: Cùng mô hình, cùng seed → kết quả khác nhau trên GPU khác nhau
2. **Phân tích nguyên nhân**: TF32, cuDNN, floating-point precision, optimizer sensitivity
3. **Đề xuất giải pháp**: Các cấu hình deterministic và best practices
4. **Phân biệt**: Statistical reproducibility vs Bit-exact reproducibility

## 📚 Tài liệu tham khảo

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- TF32 Precision: https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/
- Non-Determinism in Deep Learning: https://arxiv.org/abs/2001.11396

