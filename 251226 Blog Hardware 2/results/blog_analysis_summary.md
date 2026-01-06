# Phân tích Ảnh hưởng của Khác biệt GPU đến Tính chính xác Mô hình

## Key Findings

### 1. So sánh Accuracy giữa các GPU

Với cùng config (TF32=False, AMP=False, num_workers=0):

- **TPU**: 78.45% (std: 0.00%)
- **NVIDIA GeForce RTX 3060**: 77.92% (std: 0.24%)
- **NVIDIA L4**: 77.85% (std: 0.10%)
- **NVIDIA GeForce RTX 5090**: 77.81% (std: 0.07%)
- **NVIDIA A100-SXM4-40GB**: 77.71% (std: 0.10%)
- **Tesla T4**: 77.64% (std: 0.05%)

### 2. Ảnh hưởng của Cấu hình

Các yếu tố ảnh hưởng đến accuracy:

- **TF32**: Ảnh hưởng khác nhau tùy GPU (chỉ GPU Ampere+ hỗ trợ)
- **AMP**: Có thể ảnh hưởng đến accuracy và reproducibility
- **num_workers**: Ảnh hưởng nhỏ đến accuracy nhưng có thể ảnh hưởng reproducibility
- **Deterministic algorithms**: Giúp cải thiện reproducibility

### 3. Optimizer Sensitivity

Các optimizer khác nhau có độ nhạy cảm khác nhau với floating-point differences:

- **Adam**: Mean accuracy = 76.60%
- **RMSprop**: Mean accuracy = 72.68%
- **SGD**: Mean accuracy = 69.09%

### 4. Statistical vs Bit-exact Reproducibility

Phân loại kết quả:

- **Statistical (Chấp nhận)**: 1 config(s)

**Kết luận**:
- Bit-exact reproducibility rất khó đạt được trên GPU khác nhau
- Statistical reproducibility là mục tiêu thực tế
- Cần so sánh xu hướng (trend) thay vì giá trị tuyệt đối

### 5. Khuyến nghị

1. **So sánh xu hướng**: Quan trọng hơn so sánh giá trị tuyệt đối
2. **Ghi rõ GPU + precision**: Trong báo cáo thực nghiệm
3. **Không debug chỉ dựa trên accuracy**: Cần xem training trajectory
4. **Sử dụng deterministic config**: Khi cần reproducibility cao
