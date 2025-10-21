# Hành Trình Toàn Diện Trong Thế Giới AI: Từ Tối Ưu Hóa Đến Vận Hành Xuất Sắc

*Khám phá ba trụ cột cốt lõi: Gradient Descent, Explainable AI và MLOps*

## 📋 Tóm Tắt

**📚 Module 5 - Tuần 01 & 02**  

Chia sẻ ba trụ cột cốt lõi trong AI:

🔹 **Gradient Descent**: Các hàm mất mát, chuẩn hóa dữ liệu, vectorization.

🔹 **Explainable AI**: Phương pháp LIME và Anchor để làm AI có thể hiểu được.

🔹 **MLOps**: Thu hẹp khoảng cách nghiên cứu-production, tự động hóa quy trình.

**Đối tượng:** Kỹ sư AI, Data Scientists. **Thời gian:** 15-20 phút.

---

## 1. Lời Mở Đầu: Hành Trình Từ Lý Thuyết Đến Thực Tiễn

Trong thế giới AI hiện đại, việc xây dựng một mô hình Machine Learning thành công không chỉ dừng lại ở việc đạt được độ chính xác cao trên tập dữ liệu huấn luyện. Đó là một hành trình phức tạp bao gồm ba trụ cột cốt lõi:

1. **Tối ưu hóa nền tảng** - Hiểu sâu về Gradient Descent và các hàm mất mát
2. **Giải thích và minh bạch** - Làm cho AI có thể hiểu được thông qua XAI
3. **Vận hành và quản lý** - Đưa AI từ phòng thí nghiệm ra thế giới thực qua MLOps

Bài viết này sẽ dẫn dắt bạn qua một hành trình toàn diện, từ những nguyên lý toán học cơ bản đến việc vận hành các hệ thống AI phức tạp trong môi trường production.

## 2. Phần I: Gradient Descent - Nền Tảng Toán Học Của Học Máy

### 2.1 Linear Regression Là Gì?

Linear Regression (Hồi quy tuyến tính) là mô hình cơ bản trong học máy, tìm **đường thẳng tốt nhất** mô tả mối quan hệ giữa **biến đầu vào (x)** và **biến đầu ra (y)**:

$$\hat{y} = w x + b$$

**Chú thích chi tiết:**
- **ŷ (y-hat)**: giá trị dự đoán của mô hình (predicted value)
- **w**: hệ số góc/độ dốc của đường thẳng (slope/weight) - cho biết mức độ thay đổi của y khi x tăng 1 đơn vị
- **x**: giá trị đầu vào (input feature) - biến độc lập
- **b**: hệ số chặn/giao điểm trục tung (intercept/bias) - giá trị của y khi x = 0  

Ví dụ:  
> Dự đoán giá nhà (`price`) theo diện tích (`area`):  
> `price = w * area + b`

<div align="center">

![Linear Regression Scatter Plot](https://i.ibb.co/qLpttjFt/figure1-scatter-plot.png)

*Hình 1: Minh họa Linear Regression với scatter plot của diện tích và giá nhà*

</div>

### 2.2 Loss Function – Cách Máy Tính "Đo" Sai Số

Để biết mô hình dự đoán tốt hay không, ta cần **hàm mất mát (Loss Function)** – đo độ lệch giữa dự đoán và thực tế.

#### 2.2.1 Mean Squared Error (MSE)

$$L = \frac{1}{N} \sum_{i=1}^{N} (\hat{y_i} - y_i)^2$$

**Chú thích chi tiết:**
- **L**: giá trị hàm mất mát (loss value) - càng nhỏ càng tốt
- **N**: tổng số mẫu dữ liệu trong tập huấn luyện
- **ŷᵢ**: giá trị dự đoán của mô hình cho mẫu thứ i
- **yᵢ**: giá trị thực tế (ground truth) của mẫu thứ i
- **(ŷᵢ - yᵢ)²**: bình phương sai số cho mẫu thứ i - phép bình phương làm tăng trọng số của các lỗi lớn

**Gradient của MSE:**
$$\frac{\partial L}{\partial w} = 2x(\hat{y} - y), \quad 
\frac{\partial L}{\partial b} = 2(\hat{y} - y)$$

**Chú thích chi tiết:**
- **∂L/∂w**: đạo hàm riêng của hàm mất mát theo trọng số w - cho biết hướng và mức độ cần điều chỉnh w
- **∂L/∂b**: đạo hàm riêng của hàm mất mát theo bias b - cho biết hướng và mức độ cần điều chỉnh b
- **2x(ŷ - y)**: gradient cho w - tỷ lệ thuận với giá trị đầu vào x và sai số (ŷ - y)
- **2(ŷ - y)**: gradient cho b - chỉ phụ thuộc vào sai số, không phụ thuộc vào giá trị đầu vào

<div align="center">

![MSE Curve](https://i.ibb.co/RT86sny6/figure2-mse-curve.png)

*Hình 2: Biểu đồ đường cong MSE - outlier làm tăng loss nhanh chóng*

</div>

#### 2.2.2 Mean Absolute Error (MAE)

$$L = \frac{1}{N} \sum_{i=1}^{N} |\hat{y_i} - y_i|$$

**Chú thích chi tiết:**
- **L**: giá trị hàm mất mát MAE (Mean Absolute Error)
- **N**: tổng số mẫu dữ liệu trong tập huấn luyện
- **ŷᵢ**: giá trị dự đoán của mô hình cho mẫu thứ i
- **yᵢ**: giá trị thực tế của mẫu thứ i
- **|ŷᵢ - yᵢ|**: giá trị tuyệt đối của sai số - không có phép bình phương nên ít nhạy với outlier hơn MSE

**Gradient của MAE:**
$$\frac{\partial L}{\partial w} = x \cdot \text{sign}(\hat{y} - y)$$

**Chú thích chi tiết:**
- **∂L/∂w**: đạo hàm riêng của hàm mất mát MAE theo trọng số w
- **sign(ŷ - y)**: hàm signum - trả về +1 nếu ŷ > y, -1 nếu ŷ < y, 0 nếu ŷ = y
- **x · sign(ŷ - y)**: gradient cho w - tỷ lệ thuận với x và chỉ phụ thuộc vào dấu của sai số
- **Lưu ý**: gradient không xác định tại điểm ŷ = y (sai số = 0), có thể gây khó khăn trong tối ưu hóa

<div align="center">

![MAE Curve](https://i.ibb.co/C546R7hR/figure3-mae-curve.png)

*Hình 3: Biểu đồ đường cong MAE (hình chữ V) để so sánh với MSE*

</div>

#### 2.2.3 Huber Loss – "Lấy Cái Hay Của Cả Hai"

Huber Loss kết hợp ưu điểm của MSE và MAE:

$$L_\delta = 
\begin{cases}
\frac{1}{2}(\hat{y} - y)^2, & \text{nếu } |\hat{y} - y| \leq \delta \\
\delta \cdot |\hat{y} - y| - \frac{1}{2}\delta^2, & \text{nếu } |\hat{y} - y| > \delta
\end{cases}$$

**Chú thích chi tiết:**
- **L_δ**: giá trị hàm mất mát Huber với ngưỡng δ (delta)
- **δ**: tham số ngưỡng - quyết định khi nào chuyển từ MSE sang MAE
- **Trường hợp 1**: khi |ŷ - y| ≤ δ (sai số nhỏ) → sử dụng MSE: ½(ŷ - y)²
- **Trường hợp 2**: khi |ŷ - y| > δ (sai số lớn) → sử dụng MAE: δ|ŷ - y| - ½δ²
- **½δ²**: hằng số điều chỉnh để đảm bảo tính liên tục của hàm tại điểm chuyển đổi

🧠 **Hiểu nôm na:**  
Huber Loss giống như ta "mềm mại" với lỗi nhỏ, nhưng "khoan dung" với outlier.

<div align="center">

![Huber Loss Curve](https://i.ibb.co/rRXx3cZd/figure4-huber-curve.png)

*Hình 4: Biểu đồ Huber Loss - đoạn giữa là parabol (MSE), hai bên là tuyến tính (MAE)*

</div>

### 2.3 Chuẩn Hóa Dữ Liệu (Normalization)

Khi các đặc trưng (features) có thang giá trị khác nhau, việc học sẽ chậm hoặc không hội tụ.  
Giải pháp: **chuẩn hóa dữ liệu** về cùng phạm vi.

$$x' = \frac{x - x_{min}}{x_{max} - x_{min}}$$

**Chú thích chi tiết:**
- **x'**: giá trị sau khi chuẩn hóa (normalized value) - nằm trong khoảng [0, 1]
- **x**: giá trị gốc trước khi chuẩn hóa (original value)
- **x_min**: giá trị nhỏ nhất trong tập dữ liệu
- **x_max**: giá trị lớn nhất trong tập dữ liệu
- **x_max - x_min**: khoảng giá trị (range) của dữ liệu gốc
- **Mục đích**: đưa tất cả các đặc trưng về cùng một thang đo để gradient descent hội tụ nhanh hơn

<div align="center">

![Normalization Histogram](https://i.ibb.co/7NbXzZwK/figure5-normalization.png)

*Hình 5: Histogram trước và sau normalization - thang giá trị thay đổi từ 30-300 sang 0-1*

</div>

### 2.4 Regularization – Chống Overfitting

Khi mô hình học "quá kỹ" dữ liệu huấn luyện, nó dễ **overfit**, tức là học cả nhiễu.

Giải pháp là thêm điều khoản **phạt trọng số lớn** vào hàm loss:

$$L_{reg} = (\hat{y} - y)^2 + \lambda(w_1^2 + w_2^2 + \dots)$$

**Chú thích chi tiết:**
- **L_reg**: hàm mất mát có regularization (regularized loss function)
- **(ŷ - y)²**: phần mất mát gốc (original loss) - MSE trong ví dụ này
- **λ (lambda)**: hệ số regularization - điều chỉnh mức độ phạt trọng số lớn
- **w₁² + w₂² + ...**: tổng bình phương các trọng số (L2 regularization)
- **λ(w₁² + w₂² + ...)**: phần phạt (penalty term) - ngăn trọng số trở nên quá lớn
- **Mục đích**: cân bằng giữa việc fit dữ liệu và giữ mô hình đơn giản để tránh overfitting

<div align="center">

![Overfitting Comparison](https://i.ibb.co/YFpFGb6h/figure6-overfitting.png)

*Hình 6: So sánh hai đường hồi quy - một đường fit "quá sát" dữ liệu (overfit) và một đường mượt hơn (regularized)*

</div>

### 2.5 Vectorization – Khi Toán Học Giúp Code Chạy Nhanh Hơn ⚡

Thay vì tính từng mẫu riêng lẻ, ta có thể **gom tất cả dữ liệu thành ma trận** và dùng phép nhân vector để cập nhật tham số cùng lúc.

#### 2.5.1 Biểu diễn ma trận

$$\mathbf{X} =
\begin{bmatrix}
x_1 & 1 \\
x_2 & 1 \\
\vdots & \vdots \\
x_N & 1
\end{bmatrix},
\quad
\boldsymbol{\theta} =
\begin{bmatrix}
w \\ b
\end{bmatrix},
\quad
\hat{\mathbf{y}} = \mathbf{X}\boldsymbol{\theta}$$

**Chú thích chi tiết:**
- **X**: ma trận thiết kế (design matrix) - chứa tất cả dữ liệu đầu vào
- **x₁, x₂, ..., x_N**: các giá trị đầu vào của N mẫu dữ liệu
- **Cột thứ 2 toàn số 1**: để nhân với bias b trong phép nhân ma trận
- **θ**: vector tham số chứa trọng số w và bias b
- **ŷ**: vector dự đoán - kết quả của phép nhân ma trận Xθ
- **Lợi ích**: tính toán song song cho tất cả mẫu cùng lúc thay vì từng mẫu một

#### 2.5.2 Gradient Descent Dạng Vector

$$\boldsymbol{\theta} = \boldsymbol{\theta} - \eta \cdot \frac{1}{N}\mathbf{X}^T ( \hat{\mathbf{y}} - \mathbf{y} )$$

**Chú thích chi tiết:**
- **θ**: vector tham số cần cập nhật [w, b]
- **η (eta)**: learning rate - tốc độ học, điều chỉnh mức độ thay đổi trong mỗi bước
- **1/N**: chuẩn hóa theo số mẫu để gradient không phụ thuộc vào kích thước tập dữ liệu
- **X^T**: ma trận chuyển vị của X - để tính gradient cho tất cả tham số cùng lúc
- **(ŷ - y)**: vector sai số giữa dự đoán và giá trị thực
- **X^T(ŷ - y)**: gradient vector cho tất cả tham số
- **Mục đích**: cập nhật tất cả tham số cùng lúc một cách hiệu quả


### 2.6 Tổng Kết Các Hàm Mất Mát

| Hàm mất mát | Ưu điểm | Nhược điểm | Phù hợp khi... |
|--------------|-----------|--------------|----------------|
| **MSE** | Dễ đạo hàm, ổn định | Nhạy với outlier | Dữ liệu sạch |
| **MAE** | Bền với outlier | Đạo hàm không mượt | Dữ liệu nhiễu |
| **Huber** | Kết hợp cả hai | Cần chọn δ | Có outlier nhẹ |

**Các kỹ thuật hỗ trợ:**  
- **Normalization:** giúp hội tụ nhanh hơn  
- **Regularization:** giảm overfitting  
- **Vectorization:** tối ưu tốc độ tính toán

### 2.7 Ví Dụ Python Thực Tế

```python
import numpy as np

# Data
X = np.array([[6.7, 1], [4.6, 1], [3.5, 1], [5.5, 1]])
y = np.array([[9.1], [5.9], [4.6], [6.7]])

# Initialize
theta = np.array([[0.049], [-0.34]])
eta = 0.01
N = len(y)

# Gradient Descent
for epoch in range(1000):
    y_pred = X @ theta
    grad = (2/N) * X.T @ (y_pred - y)
    theta -= eta * grad

print(theta)
```

<div align="center">

![Loss Curve](https://i.ibb.co/CKbMsdCC/figure9-loss-curve.png)

*Hình 7: Loss curve giảm dần theo số epoch*

</div>

## 3. Phần II: Explainable AI - Làm Cho AI Có Thể Hiểu Được

### 3.1 Lời Mở: Tại Sao Cần Giải Thích Mô Hình?

Bạn có một mô hình cho điểm tín dụng chạy rất tốt trên thước đo AUC và F1. Một ngày, mô hình từ chối một hồ sơ khách hàng mà nhân viên thẩm định tin rằng đáng được duyệt. Ban lãnh đạo hỏi: Vì sao hệ thống lại ra quyết định như thế. Bạn mở dashboard, thấy vài cột đặc trưng được gán tầm quan trọng cao. Câu trả lời đó chưa đủ. Nhân viên muốn biết **ngay trong trường hợp này** mô hình đã nhìn vào điều gì. Manager muốn biết quyết định **ổn định** ra sao nếu dữ liệu đầu vào biến động nhỏ. Kỹ sư muốn biết cách **kiểm chứng** lời giải thích.

**Giải thích không phải để trang trí báo cáo. Mục tiêu là hỗ trợ ra quyết định có trách nhiệm, kiểm thử và cải thiện mô hình trong bối cảnh cụ thể.**

### 3.2 Interpretability Và Explainability: Hai Khái Niệm Cần Phân Biệt

**Diễn giải được (interpretability):** Mức độ mà con người có thể hiểu trực tiếp cách mô hình ánh xạ đầu vào sang đầu ra. Ví dụ hồi quy tuyến tính với vài đặc trưng đã chuẩn hoá có thể được xem là diễn giải được.

**Giải thích được (explainability):** Khả năng đưa ra lời giải thích về hành vi của mô hình, có thể bằng phương pháp hậu kiểm và mô hình thay thế. Ví dụ LIME và Anchor giải thích **cục bộ** dự đoán của một mô hình bất kỳ.

Trong bài viết này, chúng ta chú trọng cách giải thích hậu kiểm cho mô hình hộp đen, ưu tiên phương pháp bất phụ thuộc mô hình và áp dụng được trong thực tế.

### 3.3 Bản Đồ XAI Trong 10 Phút

#### 3.3.1 Phân Loại Nhanh Các Hướng Tiếp Cận

Có ba trục phân loại hữu ích để định vị một kỹ thuật XAI:

1. **Thời điểm can thiệp:** Ante hoc so với Post hoc
   - **Ante hoc:** thiết kế mô hình vốn đã dễ diễn giải, ví dụ hồi quy tuyến tính thưa hoặc cây quyết định nông
   - **Post hoc:** giải thích mô hình có sẵn, thường là hộp đen. LIME và Anchor thuộc nhóm này

2. **Mức phụ thuộc mô hình:** Phụ thuộc mô hình so với Bất phụ thuộc mô hình
   - **Phụ thuộc mô hình** dựa vào cấu trúc và gradient nội tại, ví dụ Integrated Gradients
   - **Bất phụ thuộc mô hình** chỉ cần truy cập hàm dự đoán, ví dụ LIME, Anchor

3. **Phạm vi hiệu lực:** Cục bộ so với Toàn cục
   - **Cục bộ** giải thích một dự đoán cụ thể hoặc một vùng lân cận quanh một điểm dữ liệu
   - **Toàn cục** mô tả xu hướng và cấu trúc chung của mô hình trên toàn tập dữ liệu

#### 3.3.2 Ba Họ Phương Pháp Phổ Biến

- **Gán đóng góp theo đặc trưng:** ví dụ LIME, SHAP, saliency dựa nhiễu. Phù hợp khi ta cần biết yếu tố nào kéo dự đoán lên hoặc xuống cho một điểm cụ thể
- **Dựa trên ví dụ:** ví dụ prototype, criticism, case based reasoning. Hữu ích khi người dùng tin tưởng bằng so sánh gần nhất
- **Dựa trên quy tắc:** ví dụ Anchor hoặc rule list. Phù hợp khi người dùng ưa các mệnh đề điều kiện dễ kiểm chứng

| **Kỹ thuật** | **Thời điểm** | **Phụ thuộc mô hình** | **Phạm vi** |
|---|---|---|---|
| LIME | Post hoc | Bất phụ thuộc | Cục bộ |
| Anchor | Post hoc | Bất phụ thuộc | Cục bộ |
| SHAP mẫu hoá | Post hoc | Bất phụ thuộc | Cục bộ đến bán toàn cục |
| Cây quyết định nông | Ante hoc | Nền tảng mô hình | Toàn cục |
| Integrated Gradients | Post hoc | Phụ thuộc mô hình | Cục bộ |

### 3.4 Trực Giác LIME

**LIME (Local Interpretable Model-agnostic Explanations)** cung cấp một **mô hình thay thế cục bộ** $g$ để mô tả hành vi của mô hình gốc $f$ quanh một điểm quan tâm $x$. Thay vì cố hiểu toàn bộ $f$, ta "phóng to" vào vùng lân cận của $x$ bằng một thước đo gần--xa, sinh các điểm nhiễu có trọng số theo độ gần, rồi khớp một mô hình đơn giản (thường là tuyến tính thưa) để suy luận đóng góp của đặc trưng.

#### 3.4.1 Định Nghĩa Và Hàm Mục Tiêu

Gọi $\pi_x(z)$ là trọng số lân cận (kernel) đo mức "gần" giữa $z$ và $x$, $L$ là mất mát đo chênh lệch dự đoán giữa $f$ và mô hình thay thế $g$, và $\Omega(g)$ là phạt độ phức tạp. LIME tối ưu:
$$g^\star = \arg\min_{g \in \mathcal{G}} L(f,g,\pi_x) + \Omega(g)$$

**Chú thích chi tiết:**
- **g***: mô hình thay thế tối ưu (optimal surrogate model)
- **arg min**: tìm đối số (argument) làm cho hàm đạt giá trị nhỏ nhất
- **g ∈ G**: mô hình thay thế g thuộc tập các mô hình có thể G (thường là mô hình tuyến tính đơn giản)
- **L(f,g,π_x)**: hàm mất mát đo độ khác biệt giữa mô hình gốc f và mô hình thay thế g
- **π_x**: phân phối trọng số lân cận quanh điểm x cần giải thích
- **Ω(g)**: hàm phạt độ phức tạp của mô hình g (regularization term)
- **Mục đích**: tìm mô hình đơn giản nhất có thể mô tả hành vi của mô hình phức tạp quanh điểm x

Với phân loại nhị phân, $L$ thường là mất mát logistic hoặc bình phương có trọng số; với hồi quy, thường là MSE có trọng số:
$$L(f,g,\pi_x) = \mathbb{E}_{z \sim \pi_x}\!\Big[w(z)\,\ell\!\big(f(z),g(z)\big)\Big], \quad w(z)=\pi_x(z)$$

**Chú thích chi tiết:**
- **L(f,g,π_x)**: hàm mất mát có trọng số của LIME
- **E_{z~π_x}**: kỳ vọng (expectation) theo phân phối π_x quanh điểm x
- **w(z)**: trọng số của điểm z, bằng π_x(z)
- **ℓ(f(z),g(z))**: hàm mất mát cơ bản giữa dự đoán của mô hình gốc f(z) và mô hình thay thế g(z)
- **f(z)**: dự đoán của mô hình gốc tại điểm z
- **g(z)**: dự đoán của mô hình thay thế tại điểm z
- **Mục đích**: đo độ khác biệt giữa hai mô hình, có trọng số cao cho các điểm gần x

#### 3.4.2 Quy Trình LIME Theo Từng Bước

1. **Chọn điểm cần giải thích** $x$ và lớp đích (nếu phân loại đa lớp)
2. **Sinh mẫu lân cận:** tạo $N$ biến thể $z_i' \sim q(\cdot\,|\,x')$ bằng cách bật/tắt các thành phần của $x'$ rồi ánh xạ $z_i=\phi(z_i')$
3. **Gán trọng số lân cận:** $w_i=\pi_x(z_i)$ với khoảng cách $D$ phù hợp và kernel width $\sigma$
4. **Gọi mô hình gốc:** lấy $y_i=f(z_i)$
5. **Khớp mô hình thay thế** $g$: thường dùng hồi quy tuyến tính thưa để tối thiểu $L+\Omega$
6. **Trình bày lời giải thích:** hệ số $\beta_j$ của $g$ cho biết mức đóng góp cục bộ của đặc trưng $j$ vào dự đoán tại $x$

#### 3.4.3 Các Tiêu Chí Đánh Giá Lời Giải Thích

**Fidelity cục bộ:** Mức độ mô hình thay thế hoặc quy tắc tái hiện hành vi của mô hình gốc quanh điểm đang xét:
$$\mathrm{Fid}(x) = \mathbb{E}_{z \sim \pi_x}\big[\ell\big(f(z),g(z)\big)\big]$$

**Chú thích chi tiết:**
- **Fid(x)**: độ trung thực cục bộ (local fidelity) tại điểm x
- **E_{z~π_x}**: kỳ vọng theo phân phối lân cận π_x quanh điểm x
- **ℓ(f(z),g(z))**: hàm mất mát giữa dự đoán của mô hình gốc f(z) và mô hình thay thế g(z)
- **f(z)**: dự đoán của mô hình gốc tại điểm z
- **g(z)**: dự đoán của mô hình thay thế tại điểm z
- **Mục đích**: đo mức độ mô hình thay thế g có thể tái hiện hành vi của mô hình gốc f quanh điểm x
- **Giá trị**: càng nhỏ càng tốt - cho thấy mô hình thay thế càng giống mô hình gốc

**Stability:** Mức độ lời giải thích ít thay đổi khi lặp lại với hạt giống ngẫu nhiên khác, với nhiễu nhỏ ở đầu vào, hoặc với cấu hình lân cận.

**Sparsity:** Mức độ gọn của lời giải thích, ví dụ số đặc trưng được chọn hoặc độ dài quy tắc.

**Coverage:** Với quy tắc, phần trăm các điểm trên phân phối dữ liệu mà quy tắc có thể áp dụng.

### 3.5 Anchor: Quy Tắc Có Độ Tin Cậy Cao

**Anchor** tìm các quy tắc dạng if-then có độ chính xác cao trong một miền áp dụng nhất định, kèm độ bao phủ để nói rõ phạm vi mà quy tắc có hiệu lực.

#### 3.5.1 Ý Tưởng Cốt Lõi

Anchor tiếp cận theo dạng quy tắc thoả ngưỡng độ chính xác, đồng thời tìm độ bao phủ lớn nhất trong phạm vi còn giữ được độ tin cậy. Ví dụ: "Nếu tuổi > 30 VÀ thu nhập > 50,000,000 VND thì mô hình sẽ từ chối hồ sơ với độ chính xác 95% và độ bao phủ 15%".

#### 3.5.2 Quy Trình Anchor

1. **Khởi tạo:** Bắt đầu với một quy tắc rỗng
2. **Mở rộng:** Thêm các điều kiện để tăng độ chính xác
3. **Kiểm tra:** Đánh giá độ chính xác và độ bao phủ
4. **Tối ưu:** Cân bằng giữa độ chính xác và độ bao phủ

### 3.6 Khi Nào Dùng Phương Pháp Cục Bộ

Phương pháp cục bộ hữu ích khi câu hỏi nghiệp vụ mang tính từng ca cụ thể như từ chối một hồ sơ tín dụng, gợi ý một đơn thuốc, hay duyệt một giao dịch. Khi đó ta quan tâm vùng lân cận của điểm $x$, đặc trưng bởi một **phân phối lân cận** $\pi_x$.

**Sơ đồ quyết định nhanh để chọn công cụ:**

1. **Bạn cần lời giải thích cho một ca cụ thể hay cho bức tranh chung**
   - Trường hợp cụ thể: ưu tiên phương pháp cục bộ như LIME hoặc Anchor
   - Bức tranh chung: cân nhắc mô hình diễn giải đơn giản, phân tích toàn cục

2. **Người dùng mục tiêu muốn đọc gì**
   - Điểm cộng trừ theo đặc trưng: LIME
   - Mệnh đề điều kiện dễ kiểm tra: Anchor

3. **Ràng buộc tính toán**
   - Chi phí lấy mẫu hạn chế: giới hạn số mẫu và kích thước lân cận
   - Dữ liệu hình ảnh: chú ý phân đoạn superpixel và tác động đến ổn định lời giải thích

## 4. Phần III: MLOps - Từ Hỗn Loạn Thử Nghiệm Đến Vận Hành Xuất Sắc

### 4.1 Lời Mở Đầu: Cơn Ác Mộng Lúc 3 Giờ Sáng

Hãy tưởng tượng bạn là một kỹ sư trong một công ty thương mại điện tử hàng đầu tại Việt Nam. Đội ngũ Khoa học Dữ liệu (Data Science) vừa cho ra mắt một hệ thống gợi ý sản phẩm (recommendation engine) vô cùng thông minh. Mọi chỉ số trong môi trường thử nghiệm đều hoàn hảo. Ban lãnh đạo kỳ vọng doanh thu sẽ tăng vọt.

Thế rồi, vào một đêm thứ Bảy, lúc 3 giờ sáng, điện thoại của bạn reo lên liên hồi. Hệ thống cảnh báo khẩn cấp: Doanh số tại Sài Gòn sụt giảm thảm hại. Khi kiểm tra, bạn bàng hoàng phát hiện ra hệ thống đang gợi ý... **áo phao và áo giữ nhiệt cho người dùng ở Sài Gòn, giữa lúc thành phố đang trải qua đợt nắng nóng đỉnh điểm.**

<div align="center">

![Sự cố vận hành có thể xảy ra bất cứ lúc nào](https://i.ibb.co/bMMJXscX/3am-crisis.png)

*Hình 8: Sự cố vận hành có thể xảy ra bất cứ lúc nào trong hệ thống ML*

</div>

Một loạt câu hỏi hiện ra trong đầu bạn:
- Phiên bản mô hình nào đang chạy trên production?
- Nó được huấn luyện trên bộ dữ liệu nào?
- Làm thế nào mà nó có thể vượt qua được các khâu kiểm thử?
- Ai đã triển khai nó và vào lúc nào?

Nếu bạn không thể trả lời những câu hỏi này một cách nhanh chóng, bạn không đơn độc. Đây chính là "cơn ác mộng" mà rất nhiều tổ chức đã và đang phải đối mặt. Nó phơi bày một sự thật trần trụi: việc xây dựng một mô hình Machine Learning (ML) hoạt động tốt trên Jupyter Notebook chỉ là **10% của tảng băng chìm**. 90% còn lại, phần phức tạp và quyết định sự thành bại của một dự án AI, nằm ở việc vận hành, duy trì và quản lý nó một cách bền vững trong môi trường thực tế.

<div align="center">

![Phần chìm của tảng băng trong các hệ thống ML](https://i.ibb.co/GvxDV7t8/hidden-complexity.png)

*Hình 9: Phần chìm của tảng băng trong các hệ thống ML*

</div>

Và đó chính là lý do **MLOps** ra đời. Nó không phải là một công cụ, cũng không phải là một công nghệ đơn lẻ. MLOps là một triết lý, một văn hóa, một tập hợp các phương pháp thực hành tốt nhất nhằm thu hẹp khoảng cách giữa thế giới thử nghiệm của các nhà khoa học dữ liệu và thế giới vận hành của các kỹ sư.

### 4.2 Hành Trình Lịch Sử: Tại Sao MLOps Là Một Điều Tất Yếu?

#### 4.2.1 Những Năm Nền Tảng (1960s - 1990s): Giấc Mơ Ban Đầu

Giai đoạn này chứng kiến sự ra đời của các khái niệm sơ khai như mạng nơ-ron Perceptron. AI lúc này chủ yếu nằm trong các phòng thí nghiệm, với nhiều kỳ vọng nhưng cũng nhanh chóng rơi vào "mùa đông AI" do những hạn chế về năng lực tính toán và dữ liệu.

#### 4.2.2 Thời Kỳ Phục Hưng (2000s - 2010): Sự Trỗi Dậy Của Deep Learning

Mọi thứ thay đổi vào những năm 2000 và đặc biệt là sau 2010. Ba yếu tố cùng hội tụ:

1. **Đột phá về thuật toán:** Nghiên cứu của Geoffrey Hinton đã khơi lại cuộc cách mạng về Deep Learning
2. **Sức mạnh tính toán:** Sự chuyển dịch từ CPU sang GPU đã cho phép huấn luyện các mô hình phức tạp hơn rất nhiều
3. **Dữ liệu lớn (Big Data):** Internet bùng nổ, tạo ra nguồn "nhiên liệu" khổng lồ cho các mô hình ML

#### 4.2.3 Kỷ Nguyên Công Nghiệp Hóa (2010 - 2015): "Vấn Đề Chiếc Laptop"

Các công ty bắt đầu ồ ạt triển khai ML. Tuy nhiên, họ nhanh chóng đối mặt với một thực tế phũ phàng: một mô hình hoạt động hoàn hảo trên laptop của nhà khoa học dữ liệu lại thất bại thảm hại khi đưa lên môi trường production.

- **Môi trường không nhất quán:** Thư viện, phiên bản Python, và cấu hình trên máy cá nhân khác xa so với máy chủ
- **Dữ liệu động:** Dữ liệu thực tế luôn thay đổi, không "sạch" và tĩnh như dữ liệu huấn luyện
- **Yêu cầu về quy mô và độ tin cậy:** Production đòi hỏi khả năng phục vụ hàng triệu người dùng và phải hoạt động 24/7

Khoảng cách giữa nghiên cứu và sản xuất ngày càng lớn, tạo ra một "nút thắt cổ chai" khổng lồ. 87% các dự án ML không bao giờ đến được tay người dùng cuối.

<div align="center">

![Sự khác biệt giữa môi trường Nghiên cứu và Production](https://i.ibb.co/1JfCmLK1/research-vs-production.png)

*Hình 10: Sự khác biệt giữa môi trường Nghiên cứu và Production*

</div>

#### 4.2.4 Sự Ra Đời Của MLOps (2015 - 2018): Lời Giải Cho Bài Toán Vận Hành

Cộng đồng nhận ra rằng để "công nghiệp hóa" AI thành công, chúng ta cần một phương pháp luận mới. Thuật ngữ "MLOps" ra đời, là sự kết hợp các nguyên tắc của **DevOps** với các quy trình đặc thù của **Machine Learning** và **Data Engineering**. MLOps tập trung giải quyết các thách thức cốt lõi:

- **Khả năng tái lập (Reproducibility):** Đảm bảo có thể tạo lại mô hình một cách nhất quán
- **Quản lý phiên bản (Versioning):** Theo dõi sự thay đổi của cả code, dữ liệu và mô hình
- **Triển khai (Deployment):** Đưa mô hình lên production một cách đáng tin cậy
- **Giám sát (Monitoring):** Theo dõi hiệu suất và phát hiện các vấn đề như "data drift"
- **Quản trị (Governance):** Đảm bảo tuân thủ và sử dụng AI một cách có trách nhiệm

### 4.3 MLOps vs. DevOps: Người Thừa Kế Hay Một Thực Thể Hoàn Toàn Mới?

Một câu hỏi tôi thường gặp là: "MLOps có phải chỉ là DevOps dành cho Machine Learning không?" Câu trả lời là **vừa đúng, vừa không**. MLOps thừa hưởng triết lý cốt lõi của DevOps, tuy nhiên, hệ thống ML có những đặc thù rất riêng.

#### 4.3.1 Analogy: Nhà Hàng

Hãy tưởng tượng DevOps giống như việc vận hành một chuỗi nhà hàng thức ăn nhanh. Mọi thứ đều có công thức chuẩn, quy trình lắp ráp (build) và phục vụ (deploy) được tự động hóa tối đa.

MLOps thì giống như việc vận hành một nhà hàng sao Michelin:

- **Nguyên liệu (Dữ liệu) là Vua:** Chất lượng món ăn phụ thuộc tuyệt đối vào độ tươi ngon của nguyên liệu và có thể thay đổi theo mùa (data drift)
- **Công thức (Mô hình) mang tính thử nghiệm:** Bếp trưởng (nhà khoa học dữ liệu) liên tục thử nghiệm các công thức mới. Cần phải có một hệ thống để ghi lại tất cả các thử nghiệm này
- **Chất lượng món ăn (Hiệu suất mô hình) có thể suy giảm:** Một món ăn được yêu thích hôm nay có thể trở nên nhàm chán vào ngày mai (concept drift). Cần liên tục theo dõi phản hồi để điều chỉnh

<div align="center">

![Vận hành một hệ thống ML giống như điều hành một nhà bếp chuyên nghiệp](https://i.ibb.co/QjcwJhR0/restaurant-analogy.png)

*Hình 11: Vận hành một hệ thống ML giống như điều hành một nhà bếp chuyên nghiệp*

</div>

#### 4.3.2 Những Điểm Khác Biệt Cốt Lõi

| **Khía cạnh** | **DevOps** | **MLOps** |
|---|---|---|
| **Thành phần chính** | Mã nguồn (Code), Hạ tầng (Infrastructure) | Code, **Dữ liệu (Data)**, **Mô hình (Models)** |
| **Tập trung kiểm thử** | Chức năng, Tích hợp, Hiệu năng hệ thống | Chất lượng dữ liệu, Hiệu suất mô hình, **Sự suy giảm hiệu suất (Drift)** |
| **Quản lý phiên bản** | Code, Cấu hình | Code, Cấu hình, **Dữ liệu**, **Mô hình**, **Các thử nghiệm** |
| **Giám sát** | Sức khỏe hệ thống (CPU, RAM), Logs | Sức khỏe hệ thống + **Độ trôi dữ liệu (Data Drift)**, **Độ trôi khái niệm (Concept Drift)**, **Chất lượng dự đoán** |
| **Vòng đời phát triển** | Tuyến tính hơn (Plan -> Code -> Build -> Test -> Deploy) | Mang tính thử nghiệm và lặp lại cao (**Data -> Model -> Deploy -> Monitor -> Retrain**) |

Sự xuất hiện của **Dữ liệu** và **Mô hình** như những "công dân hạng nhất" (first-class citizens) đã làm thay đổi hoàn toàn cuộc chơi.

<div align="center">

![Vòng đời MLOps mở rộng với các giai đoạn đặc thù](https://i.ibb.co/Q7WRpV39/mlops-lifecycle.png)

*Hình 12: Vòng đời MLOps mở rộng với các giai đoạn đặc thù*

</div>

### 4.4 Các Trụ Cột Cốt Lõi Của MLOps

Một hệ thống MLOps trưởng thành được xây dựng trên nhiều trụ cột. Dưới đây là những trụ cột quan trọng nhất mà bất kỳ kỹ sư nào cũng cần nắm vững.

#### 4.4.1 Quản Lý Phiên Bản Toàn Diện (Version Everything)

Đây là nền tảng của mọi thứ. Trong MLOps, chúng ta không chỉ `git commit` mã nguồn.

- **Version Code:** Sử dụng Git như thông thường để quản lý code tiền xử lý, huấn luyện, và triển khai
- **Version Data:** Dữ liệu là "mã nguồn" của mô hình. Các công cụ như **DVC (Data Version Control)** hay Pachyderm cho phép chúng ta "version" dữ liệu
- **Version Model:** Mỗi mô hình được huấn luyện là một "artifact" cần được lưu trữ và quản lý phiên bản thông qua các Model Registry (như trong MLflow, SageMaker)

#### 4.4.2 Tự Động Hóa Quy Trình (Automated Pipelines - CI/CD for ML)

Tự động hóa là trái tim của MLOps. Một quy trình ML (ML Pipeline) tự động hóa tất cả các bước từ dữ liệu thô đến mô hình trên production.

- **Continuous Integration (CI):** Bao gồm kiểm thử và xác thực Code, Dữ liệu, và Mô hình
- **Continuous Deployment (CD):** Bao gồm đóng gói và triển khai Mô hình một cách tự động, thường sử dụng các chiến lược như Canary Release hoặc A/B Testing

#### 4.4.3 Giám Sát Liên Tục (Continuous Monitoring)

Công việc của một kỹ sư MLOps không kết thúc khi mô hình được triển khai.

- **Giám sát Hệ thống:** Theo dõi các chỉ số vận hành như độ trễ (latency), lưu lượng (traffic), tỷ lệ lỗi (error rate)
- **Giám sát Hiệu suất Mô hình:** Theo dõi các chỉ số nghiệp vụ (business metrics) như tỷ lệ click, tỷ lệ chuyển đổi
- **Giám sát Độ trôi (Drift Detection):** Phát hiện **Data Drift** và **Concept Drift** để kích hoạt cảnh báo hoặc quy trình huấn luyện lại (retraining)

#### 4.4.4 Quản Trị Và Khả Năng Giải Thích (Governance & Explainability)

- **Model Lineage:** Khả năng truy vết nguồn gốc của một mô hình: nó được huấn luyện từ code nào, dữ liệu nào, bởi ai, và khi nào
- **Explainability:** Sử dụng các kỹ thuật như SHAP hoặc LIME để giải thích các dự đoán của mô hình, giúp xây dựng lòng tin và tuân thủ các quy định

### 4.5 MLOps Thực Chiến: Học Hỏi Từ Những Người Khổng Lồ

#### 4.5.1 Netflix: Metaflow - Đặt Con Người Vào Trung Tâm

Họ xây dựng **Metaflow**, một framework cho phép các nhà khoa học dữ liệu dễ dàng mở rộng quy mô từ local lên cloud mà không cần thay đổi code. Triết lý của họ là: **Hãy để công cụ thích ứng với con người, chứ không phải bắt con người chạy theo công cụ.**

<div align="center">

![Kiến trúc đơn giản nhưng mạnh mẽ của Metaflow](https://i.ibb.co/PGYb5c0n/metaflow-solution.png)

*Hình 13: Kiến trúc đơn giản nhưng mạnh mẽ của Metaflow*

</div>

#### 4.5.2 Uber: Michelangelo & Feature Store - Nền Tảng Cho Quy Mô Lớn

Họ xây dựng nền tảng **Michelangelo**, với "trái tim" là **Feature Store** - một kho lưu trữ tập trung các đặc trưng có thể tái sử dụng, giúp loại bỏ sự trùng lặp và tăng tốc độ phát triển.

<div align="center">

![Kết quả ấn tượng của Uber sau khi áp dụng MLOps](https://i.ibb.co/pjHFP83Z/uber-results.png)

*Hình 14: Kết quả ấn tượng của Uber sau khi áp dụng MLOps*

</div>

#### 4.5.3 OpenAI: RLHF - Khi Phản Hồi Của Con Người Là Một Phần Của "Ops"

OpenAI tiên phong trong việc sử dụng **Reinforcement Learning from Human Feedback (RLHF)**, tích hợp sự đánh giá tinh vi của con người vào vòng lặp cải tiến mô hình.

<div align="center">

![Quy trình RLHF tích hợp phản hồi con người vào vòng lặp vận hành](https://i.ibb.co/s9FPDPqT/rlhf-process.png)

*Hình 15: Quy trình RLHF tích hợp phản hồi con người vào vòng lặp vận hành*

</div>

### 4.6 Con Người Vận Hành Hệ Thống: Sự Trỗi Dậy Của Kỹ Sư MLOps

#### 4.6.1 Các Mô Hình Tổ Chức

Có hai mô hình phổ biến:

1. **Nhà Khoa Học Dữ Liệu Toàn Năng (End-to-End Data Scientist):** Một người đảm nhận toàn bộ vòng đời. Mô hình này linh hoạt, phù hợp với các startup hoặc dự án nhỏ
2. **Đội Ngũ Đa Chức Năng (Cross-Functional Team):** Một đội ngũ bao gồm các chuyên gia với vai trò rõ ràng. Đây là mô hình phổ biến và có khả năng mở rộng tốt hơn

<div align="center">

![Sự cộng hưởng của các chuyên gia trong đội ngũ MLOps](https://i.ibb.co/rKyY7Fbx/cross-functional-team.png)

*Hình 16: Sự cộng hưởng của các chuyên gia trong đội ngũ MLOps*

</div>

#### 4.6.2 Các Vai Trò Chính Trong Đội Ngũ MLOps

- **Data Scientist:** Tập trung vào việc phân tích dữ liệu, thử nghiệm và xây dựng mô hình để giải quyết bài toán kinh doanh
- **Data Engineer:** Xây dựng và duy trì các đường ống dữ liệu (data pipelines) vững chắc, đảm bảo dữ liệu chất lượng cao luôn sẵn sàng
- **ML Engineer:** Là cầu nối giữa Data Scientist và MLOps Engineer. Họ tối ưu hóa mô hình, xây dựng các pipeline huấn luyện và tích hợp mô hình vào các ứng dụng
- **MLOps Engineer / AI Platform Engineer:** Chuyên gia về hạ tầng và tự động hóa. Họ là những người đảm bảo toàn bộ cỗ máy AI vận hành trơn tru, đáng tin cậy và có khả năng mở rộng

#### 4.6.3 Lộ Trình Sự Nghiệp

MLOps là một miền đất hứa cho các kỹ sư. Lộ trình phát triển thường đi theo hai hướng chính:

1. **Từ DevOps -> MLOps Engineer:** Nếu bạn đã có nền tảng vững chắc về DevOps, Kubernetes, CI/CD, bạn có thể học thêm kiến thức về ML
2. **Từ Data Scientist/Software Engineer -> ML Engineer:** Nếu bạn mạnh về xây dựng mô hình hoặc phát triển phần mềm, bạn có thể trau dồi thêm kỹ năng về vận hành

<div align="center">

![Sự tiến hóa của cấu trúc đội ngũ MLOps theo quy mô tổ chức](https://i.ibb.co/kVyWDJQ1/team-evolution.png)

*Hình 17: Sự tiến hóa của cấu trúc đội ngũ MLOps theo quy mô tổ chức*

</div>

### 4.7 Con Đường Phía Trước: Từ MLOps Đến LLMOps Và AgenticAI Ops

Thế giới AI không ngừng vận động. MLOps là nền tảng, nhưng trên nền tảng đó, những phương pháp vận hành mới đang hình thành để đáp ứng sự phức tạp ngày càng tăng của các hệ thống AI.

<div align="center">

![Sự tiến hóa của các framework vận hành AI](https://i.ibb.co/60g5CB4B/ops-spectrum.png)

*Hình 18: Sự tiến hóa của các framework vận hành AI*

</div>

- **LLMOps:** Khi các mô hình ngôn ngữ lớn (LLMs) trở nên phổ biến, các thách thức vận hành mới cũng xuất hiện, tập trung vào Quản lý Prompt, Cơ sở dữ liệu Vector và giám sát các vấn đề đặc thù của LLM
- **AgenticAI Ops:** Đây là tương lai xa hơn, khi các hệ thống AI (agents) có khả năng tự chủ lập kế hoạch, sử dụng các công cụ và thực thi các tác vụ phức tạp, đòi hỏi việc vận hành tập trung vào Điều phối công cụ, Quản lý bộ nhớ và các lan can an toàn

<div align="center">

![Các tầng vận hành AI, mỗi tầng mới xây dựng dựa trên nền tảng của tầng trước đó](https://i.ibb.co/Qvyy6bHN/ops-layers.png)

*Hình 19: Các tầng vận hành AI, mỗi tầng mới xây dựng dựa trên nền tảng của tầng trước đó*

</div>

Việc nắm vững MLOps hôm nay chính là bạn đang xây dựng nền móng vững chắc để sẵn sàng chinh phục những đỉnh cao mới của LLMOps và AgenticAI Ops trong tương lai.

## 5. Kết Luận: Hành Trình Toàn Diện Trong Thế Giới AI

Chúng ta đã đi qua một chặng đường dài, từ những nguyên lý toán học cơ bản của Gradient Descent, đến việc làm cho AI có thể hiểu được thông qua Explainable AI, và cuối cùng là vận hành các hệ thống AI phức tạp trong môi trường production thông qua MLOps.

### 5.1 Ba Trụ Cột Cốt Lõi

**1. Gradient Descent - Nền Tảng Toán Học:**
- Hiểu sâu về đạo hàm, gradient và các hàm mất mát
- Nắm vững vai trò của chuẩn hóa dữ liệu trong tối ưu hóa
- Lựa chọn hàm mất mát phù hợp với đặc tính dữ liệu và mục tiêu bài toán

**2. Explainable AI - Minh Bạch Và Tin Cậy:**
- Phân biệt giữa interpretability và explainability
- Áp dụng LIME và Anchor để giải thích mô hình hộp đen
- Đánh giá chất lượng lời giải thích thông qua fidelity, stability, sparsity và coverage

**3. MLOps - Vận Hành Xuất Sắc:**
- Thu hẹp khoảng cách giữa nghiên cứu và production
- Xây dựng hệ thống quản lý phiên bản toàn diện
- Tự động hóa quy trình và giám sát liên tục

### 5.2 Những Gì Cần Ghi Nhớ

1. **AI thành công là sự kết hợp của ba yếu tố:** Lý thuyết vững chắc, khả năng giải thích và vận hành hiệu quả
2. **Mỗi trụ cột đều quan trọng:** Không thể bỏ qua bất kỳ yếu tố nào trong hành trình xây dựng hệ thống AI
3. **Hành trình là một cuộc marathon:** Hãy bắt đầu từ những bước nhỏ nhất và kiên trì học hỏi
4. **Tương lai thuộc về những người hiểu cả ba:** Kỹ sư AI trong tương lai cần nắm vững cả lý thuyết, khả năng giải thích và vận hành

### 5.3 Những Bước Đầu Tiên Bạn Có Thể Làm Ngay Hôm Nay

**Về Gradient Descent:**
- Thực hành tính gradient bằng tay cho các hàm đơn giản
- So sánh hiệu quả của MSE và MAE trên các bộ dữ liệu khác nhau
- Thực hiện chuẩn hóa dữ liệu đúng cách để tránh data leakage

**Về Explainable AI:**
- Sử dụng LIME để giải thích một mô hình phân loại đơn giản
- Thử nghiệm với Anchor để tìm quy tắc có độ tin cậy cao
- Đo lường fidelity và stability của các lời giải thích

**Về MLOps:**
- Bắt đầu đưa code ML vào Git và sử dụng DVC để quản lý dữ liệu
- Sử dụng MLflow để theo dõi các thử nghiệm
- Xây dựng một pipeline đơn giản để tự động hóa quy trình huấn luyện

Thế giới AI đang phát triển với tốc độ vũ bão, và vai trò của những kỹ sư có khả năng "thuần hóa" sự phức tạp của nó sẽ ngày càng trở nên quan trọng. Như Peter Drucker đã nói: **"Cách tốt nhất để dự đoán tương lai là tạo ra nó."** Chúc bạn thành công trên hành trình tạo ra tương lai của AI.

## 6. Tài Liệu Tham Khảo

> **Chú thích:** Một số hình minh họa trong bài viết được lấy từ các nguồn: AIO, và được tạo tự động bằng AI.

### Gradient Descent & Optimization
- Ruder, S. (2016). An overview of gradient descent optimization algorithms. *arXiv preprint arXiv:1609.04747*
- Goodfellow, I., Bengio, Y., & Courville, A. (2016). Deep learning. MIT press
- Bottou, L., Curtis, F. R., & Nocedal, J. (2018). Optimization methods for large-scale machine learning. *SIAM Review*, 60(2), 223-311

### Explainable AI
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2016). "Why should I trust you?" Explaining the predictions of any classifier. *Proceedings of the 22nd ACM SIGKDD international conference on knowledge discovery and data mining*
- Ribeiro, M. T., Singh, S., & Guestrin, C. (2018). Anchors: High-precision model-agnostic explanations. *Proceedings of the AAAI conference on artificial intelligence*
- Doshi-Velez, F., & Kim, B. (2017). Towards a rigorous science of interpretable machine learning. *arXiv preprint arXiv:1702.08608*

### MLOps & Production ML
- Sculley, D., et al. (2015). Hidden Technical Debt in Machine Learning Systems. *NIPS 2015*
- Netflix Metaflow Documentation (2020). *Netflix Technology Blog*
- Uber Michelangelo Platform (2017). *Uber Engineering Blog*
- Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback. *arXiv preprint arXiv:2203.02155*

### Tools & Frameworks
- MLflow Documentation (2023). *Apache Software Foundation*
- DVC Documentation (2023). *Iterative.ai*
- LIME Documentation (2023). *GitHub*