# Cross-GPU Reproducibility Research

Bộ notebook này được thiết kế để nghiên cứu và chứng minh các vấn đề về **cross-GPU reproducibility** - tính tái lập kết quả khi train mô hình deep learning trên các GPU/TPU khác nhau. Dựa trên **65 thực nghiệm** chi tiết trên 6 loại hardware khác nhau, project này phân tích nguyên nhân gốc rễ và đưa ra các giải pháp thực tế.

## 🎯 Mục tiêu

1. **Chứng minh vấn đề**: Cùng mô hình, cùng seed → kết quả khác nhau trên GPU/TPU khác nhau
2. **Phân tích nguyên nhân**: TF32, cuDNN, floating-point precision, optimizer sensitivity
3. **Đề xuất giải pháp**: Các cấu hình deterministic và best practices
4. **Phân biệt**: Statistical reproducibility vs Bit-exact reproducibility

## 🖥️ Hardware được test

- **Tesla T4** (Turing architecture)
- **NVIDIA L4** (Ada Lovelace architecture)
- **NVIDIA A100-SXM4-40GB** (Ampere architecture)
- **NVIDIA GeForce RTX 3060** (Ampere architecture)
- **NVIDIA GeForce RTX 5090** (Blackwell architecture)
- **TPU v6e-1** (Google Cloud TPU)

## 📁 Cấu trúc Notebooks

### Phase 1: Baseline Training (`01*_train_*.ipynb`)

Các notebook này chạy training baseline trên từng GPU/TPU để chứng minh sự khác biệt ban đầu.

- `01a_train_T4.ipynb` - Training trên Tesla T4
- `01b_train_L4.ipynb` - Training trên NVIDIA L4
- `01c_train_A100.ipynb` - Training trên NVIDIA A100
- `01d_train_RTX3060.ipynb` - Training trên RTX 3060
- `01e_train_TPU_v6e1.ipynb` - Training trên TPU v6e-1
- `01f_train_RTX5090.ipynb` - Training trên RTX 5090

**Mục tiêu**: 
- Train mô hình CNN trên CIFAR-10 với cùng seed
- Test với các cấu hình khác nhau (TF32 on/off, num_workers)
- Lưu kết quả vào file JSON trong thư mục `results/`

### Phase 2: Deterministic Configs Testing (`02*_test_*.ipynb`)

Các notebook này test các phương pháp khắc phục vấn đề reproducibility.

- `02a_test_T4.ipynb` - Test deterministic configs trên T4
- `02b_test_L4.ipynb` - Test deterministic configs trên L4
- `02c_test_A100.ipynb` - Test deterministic configs trên A100
- `02d_test_RTX3060.ipynb` - Test deterministic configs trên RTX 3060
- `02e_test_TPU_v6e1.ipynb` - Test deterministic configs trên TPU
- `02f_test_RTX5090.ipynb` - Test deterministic configs trên RTX 5090

**Nội dung**:
- Test với TF32 tắt/bật
- Test với cuDNN benchmark tắt/bật
- Test với num_workers = 0
- Test với deterministic algorithms
- Test với AMP (Automatic Mixed Precision)
- So sánh tất cả các cấu hình

**Kết quả**: Xác định cấu hình nào giúp reproducibility tốt nhất

### Phase 3: Optimizer Sensitivity Testing (`03*_test_optimizers_*.ipynb`)

Các notebook này chứng minh độ nhạy của optimizer với reproducibility.

- `03a_test_optimizers_T4.ipynb` - Test optimizers trên T4
- `03b_test_optimizers_L4.ipynb` - Test optimizers trên L4
- `03c_test_optimizers_A100.ipynb` - Test optimizers trên A100
- `03d_test_optimizers_RTX3060.ipynb` - Test optimizers trên RTX 3060
- `03e_test_optimizers_TPU_v6e1.ipynb` - Test optimizers trên TPU
- `03f_test_optimizers_RTX5090.ipynb` - Test optimizers trên RTX 5090

**Nội dung**:
- Test với các optimizer khác nhau (Adam, RMSProp, SGD)
- Chứng minh adaptive optimizers khuếch đại sai số nhỏ
- So sánh kết quả giữa các optimizer
- Test độ nhạy bằng cách chạy nhiều lần với cùng seed

### Phase 4: Cross-GPU Analysis (`04_analyze_cross_gpu_results.ipynb`)

**Mục tiêu**: Phân tích và so sánh kết quả từ tất cả GPU/TPU.

**Nội dung**:
- Load tất cả file JSON từ thư mục `results/`
- Phân loại kết quả theo GPU/TPU và cấu hình
- So sánh kết quả giữa các GPU/TPU (cùng cấu hình)
- Phân tích Statistical vs Bit-exact Reproducibility
- Tạo bảng tổng hợp và visualization

**Cách sử dụng**:
1. Chạy các notebook 01, 02 và 03 trên các GPU/TPU khác nhau
2. Copy tất cả file JSON vào thư mục `results/`
3. Chạy notebook này để phân tích

### Phase 5: Blog Analysis (`05_blog_analysis.ipynb`)

**Mục tiêu**: Phân tích chi tiết cho bài blog, tạo các visualization và số liệu thống kê.

**Nội dung**:
- So sánh accuracy giữa các GPU/TPU
- Phân tích tác động của cấu hình (TF32, AMP, cuDNN benchmark)
- Phân tích trajectory (loss và accuracy theo epoch)
- Phân tích độ nhạy của optimizer
- Tính toán các metrics reproducibility (correlation, max difference)
- Tạo các hình ảnh cho blog post

**Output**: 
- `results/blog_figures/*.png` - Các hình ảnh visualization
- `results/blog_analysis_summary.md` - Tóm tắt kết quả phân tích
- `results/summary_all_results.csv` - Bảng tổng hợp tất cả kết quả

## 🚀 Cài đặt

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu trúc thư mục

Sau khi chạy, cấu trúc thư mục sẽ như sau:

```
.
├── 01a_train_T4.ipynb
├── 01b_train_L4.ipynb
├── 01c_train_A100.ipynb
├── 01d_train_RTX3060.ipynb
├── 01e_train_TPU_v6e1.ipynb
├── 01f_train_RTX5090.ipynb
├── 02a_test_T4.ipynb
├── 02b_test_L4.ipynb
├── 02c_test_A100.ipynb
├── 02d_test_RTX3060.ipynb
├── 02e_test_TPU_v6e1.ipynb
├── 02f_test_RTX5090.ipynb
├── 03a_test_optimizers_T4.ipynb
├── 03b_test_optimizers_L4.ipynb
├── 03c_test_optimizers_A100.ipynb
├── 03d_test_optimizers_RTX3060.ipynb
├── 03e_test_optimizers_TPU_v6e1.ipynb
├── 03f_test_optimizers_RTX5090.ipynb
├── 04_analyze_cross_gpu_results.ipynb
├── 05_blog_analysis.ipynb
├── Blog.md                          # Bài blog chính
├── requirements.txt
├── README.md
├── data/                            # CIFAR-10 dataset (tự động download)
│   └── cifar-10-batches-py/
├── img/                             # Hình ảnh cho blog
│   └── reproducibility.png
└── results/                         # Kết quả training và phân tích
    ├── train_*.json                 # Kết quả baseline training
    ├── deterministic_*.json         # Kết quả test deterministic configs
    ├── optimizer_*.json             # Kết quả test optimizers
    ├── summary_all_results.csv      # Bảng tổng hợp
    ├── blog_figures/                # Hình ảnh cho blog
    │   ├── gpu_accuracy_comparison.png
    │   ├── config_impact_comparison.png
    │   ├── trajectory_comparison.png
    │   ├── optimizer_sensitivity.png
    │   └── ...
    └── blog_analysis_summary.md     # Tóm tắt phân tích
```

## 📊 Quy trình Test

### Bước 1: Chạy Baseline Training trên từng GPU/TPU

1. Mở notebook tương ứng với GPU/TPU bạn có (ví dụ: `01a_train_T4.ipynb`)
2. Chạy tất cả cells
3. Kết quả sẽ được lưu vào `results/train_<GPU_NAME>_*.json`

**Lưu ý**: Mỗi GPU/TPU cần chạy notebook riêng của nó.

### Bước 2: Test Deterministic Configs

1. Mở notebook tương ứng (ví dụ: `02a_test_T4.ipynb`)
2. Chạy tất cả cells để test các cấu hình khác nhau
3. Kết quả sẽ được lưu vào `results/deterministic_<GPU_NAME>_*.json`

### Bước 3: Test Optimizer Sensitivity

1. Mở notebook tương ứng (ví dụ: `03a_test_optimizers_T4.ipynb`)
2. Chạy để thấy cách optimizer ảnh hưởng đến reproducibility
3. Kết quả sẽ được lưu vào `results/optimizer_<GPU_NAME>_*.json`

### Bước 4: Phân tích Cross-GPU

1. **Quan trọng**: Đảm bảo tất cả file JSON từ các GPU/TPU đã được copy vào thư mục `results/`
2. Mở `04_analyze_cross_gpu_results.ipynb`
3. Chạy tất cả cells để:
   - Load tất cả kết quả
   - So sánh giữa các GPU/TPU
   - Phân tích reproducibility
   - Tạo visualization và bảng tổng hợp

### Bước 5: Phân tích cho Blog

1. Mở `05_blog_analysis.ipynb`
2. Chạy tất cả cells để:
   - Tạo các visualization chi tiết cho blog
   - Tính toán các metrics reproducibility
   - Tạo bảng tổng hợp và summary
3. Kết quả sẽ được lưu vào `results/blog_figures/` và `results/blog_analysis_summary.md`

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
torch.use_deterministic_algorithms(True)
# num_workers = 0 trong DataLoader
```

**Lưu ý**: Cấu hình này có thể làm chậm training đáng kể (thường 5-20% tùy workload và GPU) nhưng giúp reproducibility tốt hơn.

## 📝 Ghi chú

- Tất cả notebook sử dụng dataset CIFAR-10 (tự động download)
- Kết quả được lưu dưới dạng JSON để dễ phân tích sau
- Visualization được lưu vào thư mục `results/` và `results/blog_figures/`
- Mỗi notebook có thể chạy độc lập hoặc theo thứ tự
- Tổng cộng **65 thực nghiệm** đã được thực hiện trên 6 loại hardware

## 📚 Tài liệu tham khảo

- PyTorch Reproducibility: https://pytorch.org/docs/stable/notes/randomness.html
- TF32 Precision: https://blogs.nvidia.com/blog/2020/05/14/tensorfloat-32-precision-format/
- Non-Determinism in Deep Learning: https://arxiv.org/abs/2001.11396
- PyTorch TF32 on Ampere: https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere

## 🔗 Liên kết

- **Blog Post**: Xem file `Blog.md` để đọc bài viết chi tiết về nghiên cứu này
- **GitHub Repository**: [AIO2025_Blogs/251226 Blog Hardware 2](https://github.com/D9Dre4mer/AIO2025_Blogs/tree/main/251226%20Blog%20Hardware%202)
