📌 Nội dung chi tiết cần thêm
1. Giới thiệu / Mở đầu

Bối cảnh: Giới thiệu PG&E (Pacific Gas and Electric) – một trong những công ty điện và khí đốt lớn nhất Hoa Kỳ, phục vụ hàng triệu khách hàng.

Mục tiêu cuộc thi IISE PG&E Challenge 2025: Dự báo tải điện theo giờ cho cả một năm, trong bối cảnh khu vực bị ảnh hưởng mạnh bởi năng lượng mặt trời.

Ràng buộc:

Chỉ dùng dữ liệu cung cấp (Load, Temperature, GHI).

Không được dùng dữ liệu ngoài.

Dự báo theo “day-ahead”.

2. Data Overview

Dữ liệu cung cấp:

2 năm dữ liệu lịch sử.

Endogenous: Load (mục tiêu cần dự báo).

Exogenous: Nhiệt độ (Temperature) và Bức xạ toàn phần GHI (Global Horizontal Irradiance) từ 5 trạm.

Giải thích khái niệm:

Endogenous = phụ thuộc, được quyết định trong mô hình (Load).

Exogenous = độc lập, yếu tố bên ngoài tác động (Temperature, GHI).

Mẫu dữ liệu (Year, Month, Day, Hour, Load, Temp site 1-5, GHI site 1-5).

Thống kê mô tả: Skewness (γ1), Kurtosis (γ2).

Phân tích biến thiên:

Hourly patterns: chu kỳ 24h.

Weekly: ngày thường vs cuối tuần.

Seasonal: xu hướng khác biệt theo mùa.

Phân tích tương quan:

Load ↔ GHI: quan hệ nghịch (nhiều nắng → ít tải).

Load ↔ Temperature: quan hệ thuận (nóng/lạnh cực đoan → tăng tải).

GHI ↔ Temperature: collinearity cao giữa các site.

Kết luận EDA:

Tải điện có tính phi tuyến.

Có hiệu ứng chu kỳ ngày, tuần, mùa.

Có hiện tượng multicollinearity, cần xử lý bằng PCA, PLS, hoặc các mô hình cây như RF/XGBoost.

3. Modeling Strategy

Dimension Reduction:

PCA: trích xuất thành phần chính, không xét Y (unsupervised).

PLS: trích xuất latent variables, tối đa covariance(X,Y), supervised → phù hợp hơn.

Feature Engineering:

Time features: sinusoidal encoding (chu kỳ ngày/tuần/mùa), day-of-week, holiday flags.

Weather features: lag1, lag24, delta1, delta24.

Behavioral features: Heating Degree Hours (HDH), Cooling Degree Hours (CDH).

Model Selection (so sánh):

Linear models: yếu → loại.

Random Forest: khá tốt nhưng kém interpretability.

XGBoost: chọn làm mô hình chính.

LSTM: có thể dùng cho chuỗi dài.

Transformers: overkill với dataset nhỏ.

Hyperparameter Tuning:

Grid Search (Greedy): dùng Master–Slave (MPI) phân chia job.

Bayesian Optimization: surrogate model + acquisition function → cân bằng exploration vs exploitation.

4. Evaluation

Thí nghiệm: so sánh dần theo feature set:

Raw Data → +DimRed → +Time → +Weather → +Behavioral.

Kết quả (slide 47): Bảng thể hiện MSE, RMSE, MAE, MAPE.

Final model: XGBoost với hyperparams:

Estimators = 200

Max depth = 7

Learning rate = 0.01 – 0.1

Subsample = 0.8 – 1.0

5. Future Works

Probabilistic Forecasting:

Lý do: cần dải dự báo để quản lý lưới, thị trường năng lượng, rủi ro.

Kỹ thuật: Quantile Regression (P10/P50/P90), Conformal prediction, Bootstrap, Bayesian/Deep models.