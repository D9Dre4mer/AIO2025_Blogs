1. Giới thiệu

LightGBM (Light Gradient Boosting Machine) là một thư viện mạnh mẽ được phát triển bởi Microsoft, giúp huấn luyện mô hình Gradient Boosting nhanh hơn và tiết kiệm bộ nhớ hơn.
Trong blog này, chúng ta sẽ:

Ôn lại các mô hình nền tảng (Decision Tree, Random Forest, AdaBoost, Gradient Boosting, XGBoost).

Hiểu rõ cải tiến cốt lõi của LightGBM: Leaf-wise Growth, Histogram-based Split, GOSS, EFB.

Trình bày qua các ví dụ chi tiết (iris dataset, taxi trip dataset).

Minh họa kết quả bằng TikZ và gợi ý thêm hình ảnh thực tế.

2. Ôn tập nhanh: Các mô hình trước LightGBM
Mô hình	Hàm loss Regression	Hàm loss Classification	Cách phát triển cây	Điểm mới chính
Decision Tree	MSE	Gini/Entropy	Level-wise (top-down)	Chia theo split tốt nhất
Random Forest	MSE	Gini/Entropy	Level-wise + Bagging	Giảm variance qua sampling
AdaBoost	MSE	Exponential Loss	Sequential boosting	Tăng trọng số điểm sai
Gradient Boosting	Any diff. loss	Log-loss	Sequential boosting	Boosting theo gradient
XGBoost	Any diff. loss + Reg	Log-loss + Reg	Level-wise + Reg.	Taylor bậc 2, regularization
LightGBM	Any diff. loss + Reg	Log-loss + Reg	Leaf-wise + Histogram	Tăng tốc, giảm bộ nhớ, GOSS, EFB

👉 Ý tưởng: LightGBM giữ nền tảng boosting nhưng tối ưu tốc độ và hiệu quả bộ nhớ.

3. Cách LightGBM xây dựng cây
3.1 Level-wise vs. Leaf-wise

Thông thường các mô hình (XGBoost, Gradient Boosting) mở rộng cây theo level-wise (từng tầng).
LightGBM chọn chiến lược leaf-wise: luôn mở rộng lá có tiềm năng giảm loss nhiều nhất.

Minh họa TikZ:

\begin{tikzpicture}[level distance=1.5cm, every node/.style={circle,draw,minimum size=7mm}]
\node {Root}
  child {node {A} 
    child {node {A1}}
    child {node {A2}}
  }
  child {node {B}
    child {node {B1}
      child {node {B11}} % mở rộng lá có tiềm năng nhất
    }
    child {node {B2}}
  };
\end{tikzpicture}


👉 Trong ví dụ trên, thay vì mở rộng toàn bộ tầng như Level-wise, LightGBM chỉ mở rộng B1 vì nó hứa hẹn giảm loss lớn nhất.

3.2 Histogram-based Split

Thay vì thử tất cả các ngưỡng chia (threshold), LightGBM gom dữ liệu thành bins.
Ví dụ với Petal_Width có 6 giá trị, thay vì xét 5 điểm chia, ta chỉ xét các bin:

Bin	Giá trị	Nhóm dữ liệu
1	[0.2–0.567]	{0.2, 0.5}
2	(0.567–0.933]	{0.6, 0.7, 0.9}
3	(0.933–1.3]	{1.3}

👉 Giúp giảm số lượng phép thử, tăng tốc độ huấn luyện.

3.3 Exclusive Feature Bundling (EFB)

Với dữ liệu categorical (ví dụ màu sắc: Red, Blue, Green), one-hot encoding tạo 3 cột riêng biệt.

LightGBM gom chúng thành một cột duy nhất (bundle), vì các feature này không bao giờ đồng thời xuất hiện giá trị ≠ 0.

Ví dụ:

ID	Red	Blue	Green	→	Color_bundle
S1	1	0	0	→	0
S2	0	1	0	→	1
S3	0	0	1	→	2

👉 Tiết kiệm bộ nhớ và tăng tốc tính toán.

3.4 Gradient-based One-Side Sampling (GOSS)

Với dataset rất lớn, không cần dùng toàn bộ mẫu.

LightGBM chọn:

Tất cả mẫu có gradient lớn (khó học, quan trọng).

Một phần ngẫu nhiên các mẫu dễ (gradient nhỏ).

👉 Vẫn giữ được thông tin, nhưng giảm số lượng mẫu.

4. Case Study
4.1 Taxi Trip Duration (NYC)

Dataset: 1.4 triệu chuyến taxi.

Nhiệm vụ: Dự đoán thời gian chuyến đi.

Kết quả:

Model	RMSE ↓	Training time ↓
GradientBoosting	0.3466	452s
XGBoost	0.3497	3.39s
LightGBM	0.3482	2.65s

👉 LightGBM đạt tốc độ nhanh nhất, RMSE gần tốt nhất.

4.2 HEPMASS (High Energy Physics)

Dataset: 3.5 triệu sự kiện va chạm hạt.

Nhiệm vụ: Phân loại tín hiệu (signal) vs. nhiễu (background).

Kết quả:

Model	Accuracy ↑	AUC-ROC ↑	Training time ↓
GradientBoosting	0.9181	0.9711	4100s
XGBoost	0.9178	0.9712	20s
LightGBM	0.9179	0.9711	21s

👉 Với dữ liệu số liệu dày đặc (dense numeric), XGBoost nhỉnh hơn một chút. Nhưng LightGBM vẫn rất cạnh tranh.

5. Giải thích mô hình bằng SHAP

LightGBM hỗ trợ tốt SHAP (SHapley Additive exPlanations), giúp:

Hiểu rõ feature nào quan trọng nhất.

So sánh LightGBM và XGBoost ở cùng thang đo.

Ví dụ: Với dataset Taxi Trip, cả hai mô hình đều chỉ ra Top-3 feature giống nhau, do đó kết quả đáng tin cậy.

6. Kết luận

LightGBM vượt trội trong các dataset lớn, nhiều feature categorical.

Tốc độ huấn luyện nhanh, bộ nhớ hiệu quả, độ chính xác cao.

Tuy nhiên, với dataset nhỏ hoặc chỉ gồm numeric dense, XGBoost đôi khi nhỉnh hơn.

Với nhu cầu giải thích mô hình, SHAP kết hợp cùng LightGBM là lựa chọn mạnh mẽ.

7. Gợi ý thêm hình ảnh minh họa ngoài TikZ

Infographic: so sánh tốc độ train (cột GradientBoosting vs. XGBoost vs. LightGBM).

Diagram: minh họa quy trình GOSS (mẫu lớn gradient được giữ lại, mẫu nhỏ gradient được lấy ngẫu nhiên).

SHAP bar chart: biểu diễn mức đóng góp của feature.