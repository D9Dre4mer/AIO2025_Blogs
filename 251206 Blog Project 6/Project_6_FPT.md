# Dự Báo Giá Cổ Phiếu FPT: Tầm Nhìn Tổng Quan Vượt Qua Khung Cửa Hẹp

*Tóm tắt: Bài viết này trình bày quá trình phát triển một hệ thống dự đoán giá cổ phiếu FPT từ các mô hình Linear cơ bản đến mô hình PatchTST kết hợp với các kỹ thuật điều chỉnh bias. Nghiên cứu được thực hiện trong khuôn khổ cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge, với mục tiêu dự đoán giá đóng cửa cho 100 ngày tiếp theo. Bài viết cũng trình bày các thách thức trong việc xử lý dữ liệu time series trong bối cảnh thị trường chứng khoán có nhiều biến động.*

---

## Giới thiệu

Dự đoán giá cổ phiếu là một trong những bài toán time series forecasting phức tạp nhất. Thị trường chứng khoán chịu ảnh hưởng bởi nhiều yếu tố kinh tế, chính trị và các sự kiện "thiên nga đen" (black swan events). Nghiên cứu này trình bày quá trình phát triển một hệ thống dự đoán giá cổ phiếu FPT cho 100 ngày tiếp theo trong khuôn khổ cuộc thi [**AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge).

Cuộc thi này tập trung vào việc đánh giá hiệu quả của các mô hình tuyến tính (Linear, NLinear, DLinear) trong dự báo chuỗi thời gian dài hạn, thay vì dựa vào các mô hình deep learning phức tạp. Nghiên cứu này khám phá sức mạnh của các mô hình tuyến tính khi áp dụng vào dữ liệu tài chính thực tế, đồng thời so sánh với các phương pháp khác để đánh giá hiệu quả tương đối.

Hành trình nghiên cứu bao gồm việc đánh giá và so sánh nhiều phương pháp khác nhau, từ các mô hình tuyến tính đơn giản (Linear, NLinear, DLinear) đến các mô hình phức tạp hơn như Transformer và các phương pháp Elastic Net. Nghiên cứu cũng phát triển và áp dụng các kỹ thuật điều chỉnh bias để cải thiện độ chính xác dự đoán. Kết quả cho thấy các mô hình tuyến tính, khi được kết hợp với các kỹ thuật điều chỉnh phù hợp, có thể đạt hiệu quả tốt trong dự báo time series dài hạn với dữ liệu tài chính.

Nghiên cứu này đặt ra bốn mục tiêu cụ thể. Mục tiêu đầu tiên là dự đoán giá đóng cửa (close price) của cổ phiếu FPT cho 100 ngày tiếp theo. Mục tiêu thứ hai là so sánh và đánh giá hiệu suất của nhiều phương pháp khác nhau. Mục tiêu thứ ba là phát triển các kỹ thuật điều chỉnh bias để cải thiện độ chính xác. Mục tiêu cuối cùng là xử lý các biến động lớn trong thị trường chứng khoán.

Bài toán này diễn ra trong bối cảnh có sự kiện "thiên nga đen". Khi Tổng thống Donald Trump công bố sắc thuế và áp thuế rất nặng đối với hàng hóa xuất khẩu của Việt Nam, đây là một sự kiện chưa có tiền lệ trong lịch sử. Sự kiện này không xuất hiện trong dữ liệu training, khiến bài toán trở nên đặc biệt khó khăn.

### Repository GitHub

Mã nguồn và các notebook của dự án được công khai trên các repository sau:

- **[LTSF_Linear](https://github.com/HoangVo-Prog/LTSF_Linear)**: Repository chứa các implementation của các mô hình LTSF-Linear, Rolling Elastic Net, OLS Elastic Net, và PatchTST.
- **[AIO2025_Project6.1_StockForcasting](https://github.com/sonvt8/AIO2025_Project6.1_StockForcasting.git)**: Repository chứa hệ thống deployment với FastAPI và Streamlit, cùng các model artifacts.
- **[AIO Project 6](https://github.com/D9Dre4mer/AIO/tree/D9Dre4mer-AIO-Project/251201%20Project%206)**: Repository chứa phần thực nghiệm mô hình, bao gồm các notebook liên quan đến PatchTST được sử dụng trong bài viết này.

<div align="center">
<img src="https://i.ibb.co/d41fHgXL/Logo-999.png" alt="Logo CONQ999" width="250">

<sub>Bài viết được biên soạn bởi nhóm CONQ999</sub>
</div>

---

## Phần 1: Phân tích Project gốc - LTSF-Linear Models

### 1.1. Tổng quan về Project gốc

Điểm khởi đầu của nghiên cứu này là project gốc (`[Code-Exercise]-Project-6.1-VIC-LTSF-Linear-Forecasting.ipynb`). Project này yêu cầu implement các mô hình LTSF-Linear để dự đoán giá cổ phiếu VIC.

Project gốc cung cấp cơ sở để hiểu cách các mô hình Linear hoạt động. Tuy nhiên, project gốc chỉ dự đoán 5 ngày, trong khi cuộc thi [**AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge) yêu cầu dự đoán 100 ngày. Do đó, nghiên cứu này cần phát triển thêm nhiều phương pháp mới.

Các đặc điểm chính của project gốc được trình bày trong bảng sau:

<table>
<thead>
<tr>
<th style="text-align: center">Đặc điểm</th>
<th style="text-align: center">Giá trị</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Dữ liệu</strong></td>
<td>VIC stock (1249 điểm dữ liệu)</td>
</tr>
<tr>
<td><strong>Khoảng thời gian</strong></td>
<td>2020-08-03 đến 2025-08-01</td>
</tr>
<tr>
<td><strong>Horizon dự đoán</strong></td>
<td>5 ngày (ngắn hạn)</td>
</tr>
<tr>
<td><strong>Input lengths</strong></td>
<td>5, 30, 120, 480 ngày</td>
</tr>
<tr>
<td><strong>Chia dữ liệu</strong></td>
<td>70% train, 15% validation, 15% test</td>
</tr>
<tr>
<td><strong>Models</strong></td>
<td>Linear, DLinear, NLinear</td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 1: Đặc điểm của project gốc - Tóm tắt các thông tin chính về dữ liệu, khoảng thời gian, horizon dự đoán và input lengths.

</p>

### 1.2. Các mô hình LTSF-Linear

Project gốc giới thiệu ba mô hình Linear cơ bản. Mỗi mô hình có cách tiếp cận riêng để xử lý chuỗi thời gian. Phần này phân tích từng mô hình một.

#### Linear Model

**Linear** là mô hình đơn giản nhất trong ba mô hình. Mô hình này có thể được hình dung như một ánh xạ tuyến tính trực tiếp: nhận vào một chuỗi số và trả ra một chuỗi số khác. Mô hình sử dụng một lớp Linear để ánh xạ trực tiếp từ chuỗi thời gian đầu vào sang chuỗi dự đoán.

```python
class Linear(nn.Module):
    def __init__(self, seq_len, pred_len=5):
        super(Linear, self).__init__()
        self.seq_len = seq_len      # Độ dài chuỗi đầu vào (ví dụ: 5, 30, 120, 480 ngày)
        self.pred_len = pred_len    # Độ dài chuỗi dự đoán (ví dụ: 5 ngày)
        self.linear = nn.Linear(seq_len, pred_len)  # Lớp Linear: ánh xạ từ seq_len → pred_len
    
    def forward(self, x):
        return self.linear(x)  # Ánh xạ trực tiếp: x có shape [batch, seq_len] → output [batch, pred_len]
```

**Giải thích code:**
- `__init__`: Khởi tạo mô hình với `seq_len` (số ngày quá khứ) và `pred_len` (số ngày cần dự đoán). Lớp `nn.Linear(seq_len, pred_len)` tạo một ma trận trọng số có kích thước `[seq_len, pred_len]` để ánh xạ từng giá trị đầu vào sang giá trị đầu ra.
- `forward`: Hàm này nhận input `x` có shape `[batch_size, seq_len]` (ví dụ: [32, 5] nếu batch_size=32 và seq_len=5), sau đó áp dụng phép nhân ma trận để tạo output có shape `[batch_size, pred_len]` (ví dụ: [32, 5]). Đây là phép ánh xạ tuyến tính đơn giản nhất, không có activation function hay xử lý phức tạp nào.

Mô hình Linear ánh xạ trực tiếp từ `seq_len` giá trị đầu vào sang `pred_len` giá trị dự đoán. Mô hình này không có cơ chế xử lý pattern phức tạp và phù hợp với dữ liệu có quan hệ tuyến tính rõ ràng. Tuy nhiên, trong thực tế, giá cổ phiếu thường có nhiều pattern phức tạp hơn. Do đó, nghiên cứu này cần các mô hình tinh vi hơn.

#### DLinear Model

**DLinear** (Decomposed Linear) là một bước tiến so với Linear. Thay vì xử lý toàn bộ chuỗi thời gian cùng lúc, DLinear phân rã chuỗi thành hai thành phần: **trend** (xu hướng) và **seasonal** (mùa vụ).

Cách tiếp cận này có thể được hình dung như việc tách một bức tranh thành hai lớp: lớp nền là trend, lớp chi tiết là seasonal. Sau đó, mô hình học riêng từng lớp một.

```python
class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len=5, moving_avg=5):
        super(DLinear, self).__init__()
        self.linear_trend = nn.Linear(seq_len, pred_len)      # Mô hình cho trend
        self.linear_seasonal = nn.Linear(seq_len, pred_len)   # Mô hình cho seasonal
        # Moving average kernel để trích xuất trend
    
    def decompose(self, x):
        trend = moving_average(x)      # Tính trung bình trượt để lấy xu hướng
        seasonal = x - trend            # Phần còn lại sau khi trừ trend
        return trend, seasonal
    
    def forward(self, x):
        trend, seasonal = self.decompose(x)                    # Phân rã thành 2 thành phần
        trend_pred = self.linear_trend(trend)                  # Dự đoán trend
        seasonal_pred = self.linear_seasonal(seasonal)         # Dự đoán seasonal
        return trend_pred + seasonal_pred                       # Kết hợp 2 dự đoán
```

**Giải thích code:**
- `__init__`: Tạo hai lớp Linear riêng biệt: một cho trend (xu hướng dài hạn) và một cho seasonal (biến động ngắn hạn). `moving_avg` là kích thước cửa sổ để tính trung bình trượt.
- `decompose`: Hàm này tách chuỗi thời gian thành hai phần: `trend` được tính bằng moving average (làm mượt dữ liệu để lấy xu hướng), và `seasonal = x - trend` (phần biến động còn lại sau khi loại bỏ trend).
- `forward`: Quy trình gồm 3 bước: (1) Phân rã input thành trend và seasonal, (2) Dự đoán riêng từng thành phần bằng mô hình Linear tương ứng, (3) Cộng hai dự đoán lại để có kết quả cuối cùng. Cách này giúp mô hình học tốt hơn vì tách biệt được xu hướng dài hạn và biến động ngắn hạn.

Mô hình DLinear phân rã giúp model học riêng xu hướng dài hạn và biến động ngắn hạn. Mô hình sử dụng moving average để trích xuất trend và hiệu quả hơn Linear khi dữ liệu có pattern theo mùa. Tuy nhiên, cách tiếp cận này vẫn chưa đủ để xử lý các pattern phức tạp trong thị trường chứng khoán.

#### NLinear Model

**NLinear** (Normalized Linear) là mô hình thứ ba được nghiên cứu. Mô hình này chuẩn hóa dữ liệu bằng cách trừ đi giá trị cuối cùng. Sau đó, mô hình dự đoán và cộng lại giá trị cuối cùng.

Cách tiếp cận này giúp mô hình tập trung vào sự thay đổi thay vì giá trị tuyệt đối. Điều này đặc biệt hữu ích khi giá cổ phiếu có xu hướng tăng hoặc giảm mạnh.

```python
class NLinear(nn.Module):
    def __init__(self, seq_len, pred_len=5):
        super(NLinear, self).__init__()
        self.linear = nn.Linear(seq_len, pred_len)  # Lớp Linear tương tự như Linear model
    
    def forward(self, x):
        last_value = x[:, -1:]                      # Lấy giá trị cuối cùng của chuỗi [batch, 1]
        x_normalized = x - last_value               # Chuẩn hóa: trừ đi giá trị cuối
        pred_normalized = self.linear(x_normalized) # Dự đoán trên dữ liệu đã chuẩn hóa
        pred = pred_normalized + last_value         # Cộng lại giá trị cuối để có dự đoán thực
        return pred
```

**Giải thích code:**
- `__init__`: Tạo một lớp Linear giống như mô hình Linear, nhưng sẽ được sử dụng trên dữ liệu đã chuẩn hóa.
- `forward`: Quy trình gồm 4 bước: (1) `last_value = x[:, -1:]` lấy giá trị cuối cùng của mỗi chuỗi (shape [batch, 1]), (2) `x_normalized = x - last_value` chuẩn hóa bằng cách trừ đi giá trị cuối (giá trị cuối trở thành 0, các giá trị khác là độ lệch so với giá trị cuối), (3) `pred_normalized = self.linear(x_normalized)` dự đoán trên dữ liệu đã chuẩn hóa (mô hình học sự thay đổi), (4) `pred = pred_normalized + last_value` cộng lại giá trị cuối để chuyển từ "độ lệch" về "giá trị thực". Cách này giúp mô hình tập trung vào pattern thay đổi thay vì giá trị tuyệt đối, đặc biệt hiệu quả khi có xu hướng tăng/giảm mạnh.

Mô hình NLinear chuẩn hóa giúp model tập trung vào sự thay đổi thay vì giá trị tuyệt đối. Mô hình này hiệu quả với dữ liệu có xu hướng tăng hoặc giảm mạnh và thường cho kết quả tốt nhất trong ba mô hình Linear. Kết quả này sẽ được xác nhận trong phần tiếp theo.

### 1.3. Kết quả của Project gốc

Từ notebook gốc, nghiên cứu thu được kết quả so sánh giữa các mô hình. Bảng dưới đây trình bày kết quả chi tiết:

<table>
<thead>
<tr>
<th style="text-align: center">Input Length</th>
<th style="text-align: center">Model</th>
<th style="text-align: center">RMSE</th>
<th style="text-align: center">MAE</th>
<th style="text-align: center">R²</th>
</tr>
</thead>
<tbody>
<tr>
<td>5d</td>
<td>Linear</td>
<td>0.0690</td>
<td>0.0466</td>
<td>0.9541</td>
</tr>
<tr>
<td>5d</td>
<td>DLinear</td>
<td>0.0693</td>
<td>0.0470</td>
<td>0.9537</td>
</tr>
<tr>
<td>5d</td>
<td><strong>NLinear</strong></td>
<td><strong>0.0539</strong></td>
<td><strong>0.0317</strong></td>
<td><strong>0.9720</strong></td>
</tr>
<tr>
<td>30d</td>
<td>Linear</td>
<td>0.0772</td>
<td>0.0511</td>
<td>0.9452</td>
</tr>
<tr>
<td>30d</td>
<td>DLinear</td>
<td>0.0721</td>
<td>0.0480</td>
<td>0.9523</td>
</tr>
<tr>
<td>30d</td>
<td><strong>NLinear</strong></td>
<td><strong>0.0556</strong></td>
<td><strong>0.0337</strong></td>
<td><strong>0.9716</strong></td>
</tr>
<tr>
<td>120d</td>
<td>Linear</td>
<td>0.1004</td>
<td>0.0705</td>
<td>0.9225</td>
</tr>
<tr>
<td>120d</td>
<td>DLinear</td>
<td>0.0946</td>
<td>0.0643</td>
<td>0.9312</td>
</tr>
<tr>
<td>120d</td>
<td><strong>NLinear</strong></td>
<td><strong>0.0651</strong></td>
<td><strong>0.0418</strong></td>
<td><strong>0.9674</strong></td>
</tr>
<tr>
<td>480d</td>
<td>Linear</td>
<td>0.1776</td>
<td>0.1339</td>
<td>0.6621</td>
</tr>
<tr>
<td>480d</td>
<td>DLinear</td>
<td>0.1878</td>
<td>0.1441</td>
<td>0.6223</td>
</tr>
<tr>
<td>480d</td>
<td><strong>NLinear</strong></td>
<td><strong>0.0936</strong></td>
<td><strong>0.0684</strong></td>
<td><strong>0.9061</strong></td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 2: Kết quả so sánh các mô hình Linear - So sánh hiệu suất của Linear, DLinear và NLinear với các input lengths khác nhau (5d, 30d, 120d, 480d) theo các chỉ số RMSE, MAE và R².

</p>

Phân tích kết quả cho thấy NLinear luôn cho kết quả tốt nhất trong cả bốn độ dài input. Khi input length tăng, hiệu suất của Linear và DLinear giảm đáng kể. Trong khi đó, NLinear ổn định hơn với input length dài, với 480 ngày vẫn đạt MSE 0.0088 (thấp nhất). Kết quả này khẳng định hiệu quả của NLinear. Tuy nhiên, nghiên cứu vẫn cần phát triển thêm để đáp ứng yêu cầu 100 ngày.

### 1.4. Hạn chế của Project gốc

Mặc dù các mô hình Linear cho kết quả khá tốt trên dữ liệu VIC, chúng có những hạn chế rõ ràng. Hạn chế đầu tiên là **prediction horizon ngắn**: mặc dù notebook thử nghiệm với các input length dài (5, 30, 120, 480 ngày), nhưng prediction horizon chỉ là 5 ngày và không đáp ứng yêu cầu 100 ngày của cuộc thi [**AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge). Hạn chế thứ hai là không có tối ưu hyperparameters, sử dụng tham số mặc định và chưa được tối ưu. Hạn chế thứ ba là thiếu post-processing, không có cơ chế điều chỉnh bias sau khi dự đoán. Hạn chế thứ tư là model đơn giản, chỉ capture được quan hệ tuyến tính và khó xử lý pattern phức tạp. Hạn chế cuối cùng là không xử lý biến động lớn, không có cơ chế đặc biệt để xử lý các sự kiện "thiên nga đen".

Project gốc là một điểm khởi đầu tốt để hiểu về các mô hình Linear. Tuy nhiên, cần nhiều cải tiến để đáp ứng yêu cầu thực tế của [**cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge). Do đó, nghiên cứu này quyết định phát triển một pipeline hoàn chỉnh hơn.

---

## Phần 2: Phân tích Dữ liệu Cuộc thi

Phần này trình bày phân tích chi tiết về dữ liệu cổ phiếu FPT được sử dụng trong cuộc thi [**AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge). Phân tích dữ liệu là bước quan trọng đầu tiên trong quá trình xây dựng mô hình dự đoán, giúp hiểu rõ tính chất, đặc điểm và các pattern của dữ liệu. Tất cả các hình minh họa trong phần này được tạo từ notebook `visualization.ipynb` trong project LTSF_Linear.

### 2.1. Tổng quan dữ liệu

Dữ liệu của [**cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge) bao gồm giá lịch sử của cổ phiếu FPT từ ngày 2020-08-03 đến 2025-03-10, tổng cộng 1149 điểm dữ liệu. Dữ liệu bao gồm các cột: `time`, `open`, `high`, `low`, `close`, `volume`, và `symbol`. Mục tiêu của cuộc thi là dự đoán giá đóng cửa (close price) cho 100 ngày tiếp theo sau ngày cuối cùng trong dữ liệu training.

![FPT Stock Price Over Time](https://i.ibb.co/h1F97xD1/0c47f8db4d44.png)

<p align="center">

Hình 1: Giá đóng cửa và khối lượng giao dịch FPT từ 2020-2025 - Biểu đồ kép hiển thị xu hướng tăng giá mạnh và mối tương quan với volume.

</p>

Quan sát Hình 1 cho thấy giá đóng cửa FPT tăng từ khoảng 20 k VND (cuối 2020) lên đỉnh 135-140 k VND (cuối 2024), sau đó điều chỉnh về 120-125 k VND (đầu 2025). Giai đoạn 2023-2024 có xu hướng tăng mạnh, đồng thời volume giao dịch cũng tăng đáng kể với nhiều đợt vượt 10 triệu cổ phiếu. Mối tương quan giữa giá và volume thể hiện rõ trong các đợt biến động lớn, đặc biệt là khi giá tăng nhanh thường đi kèm volume cao. Đặc điểm này tạo thách thức cho dự đoán 100 ngày tương lai, đòi hỏi mô hình phải nắm bắt được cả xu hướng dài hạn và biến động ngắn hạn.

### 2.2. Phân tích Daily Log Returns

Log returns là một đặc trưng quan trọng trong phân tích tài chính vì nó chuẩn hóa biến động giá, giúp mô hình học được các quy luật ổn định hơn so với giá tuyệt đối. Daily log return được tính bằng công thức: $\text{log\_ret} = \log(\text{close}_t) - \log(\text{close}_{t-1})$, trong đó $\text{close}_t$ là giá đóng cửa tại thời điểm $t$ và $\text{close}_{t-1}$ là giá đóng cửa tại thời điểm trước đó. Công thức này đo lường tỷ lệ thay đổi giá theo logarit, giúp chuẩn hóa biến động và làm cho dữ liệu ổn định hơn cho mô hình học máy.

![FPT Daily Log Returns](https://i.ibb.co/dsF0Szp1/5436279d7eb1.png)

<p align="center">

Hình 2: Daily log returns của FPT - Biểu đồ dao động quanh 0, thể hiện tính chất stationary và các giai đoạn volatility khác nhau.

</p>

Quan sát Hình 2 cho thấy daily log returns dao động quanh mức 0, phản ánh tính chất stationary của chuỗi lợi suất. Các giai đoạn volatility cao xuất hiện vào đầu 2021, giữa 2022 (có đợt giảm mạnh), cuối 2022-đầu 2023, và giữa 2024, với các giá trị cực đoan đạt ±0.06-0.07. Các giai đoạn khác như một phần 2021 và 2023 có biến động thấp hơn. Đặc điểm này cho thấy thị trường có tính chất heteroscedastic (volatility thay đổi theo thời gian), đòi hỏi mô hình phải nắm bắt được cả pattern giá trị và pattern volatility.

### 2.3. Phân bố Daily Log Returns

Phân tích chi tiết về phân bố của log returns giúp hiểu rõ hơn về tính chất của dữ liệu. Histogram cho phép quan sát trực quan cách các giá trị log returns được phân bố, từ đó đánh giá được tính chất phân phối và các đặc điểm thống kê quan trọng.

![Distribution of Daily Log Returns](https://i.ibb.co/7Jwyn5K4/f217e8376e33.png)

<p align="center">

Hình 3: Phân bố daily log returns - Histogram cho thấy phân bố tập trung quanh 0 với đuôi dày (leptokurtic), phản ánh tính chất của lợi suất tài chính.

</p>

Quan sát Hình 3 cho thấy phân bố của daily log returns qua histogram với 50 bins. Phân bố tập trung mạnh quanh mức 0, với đỉnh cao và đuôi dày (leptokurtic), đặc trưng của lợi suất tài chính. Phần lớn giá trị log returns nằm trong khoảng -0.02 đến +0.02, nhưng có các giá trị cực đoan vượt ra ngoài ±0.05. Đặc điểm này cho thấy log returns không tuân theo phân bố chuẩn, mà có tính chất fat-tailed (đuôi dày), nghĩa là các sự kiện cực đoan xảy ra thường xuyên hơn so với phân bố chuẩn. Tính chất này quan trọng trong modeling vì các mô hình cần nắm bắt được cả các giá trị bình thường và các outlier, đặc biệt trong các giai đoạn biến động mạnh của thị trường.

![Scatter Chart](https://i.ibb.co/8D91g8b4/ab808f645f17.png)

<p align="center">

Hình 4: Scatter chart - Biểu đồ phân tán cho thấy mối quan hệ giữa các biến trong phân tích log returns.

</p>

Quan sát Hình 4 cho thấy scatter chart mô tả mối quan hệ giữa các biến trong phân tích log returns. Các điểm dữ liệu được phân tán trên biểu đồ, cho phép quan sát mối tương quan và pattern giữa các biến. Scatter chart này giúp xác định mối quan hệ tuyến tính hoặc phi tuyến giữa các biến, phát hiện outliers, và đánh giá mức độ tương quan. Phân tích scatter chart là bước quan trọng trong việc hiểu tính chất của dữ liệu và xác định các đặc trưng phù hợp cho mô hình dự đoán.

### 2.4. Chia dữ liệu Train và Validation

Sau khi phân tích tính chất của log returns, việc chia dữ liệu thành tập training và validation là bước quan trọng để đảm bảo mô hình được đánh giá đúng cách. Cách chia dữ liệu ảnh hưởng trực tiếp đến khả năng học và dự đoán của mô hình.

![Train vs Validation Split](https://i.ibb.co/20GNJj4P/90e44ea9ae02.png)

<p align="center">

Hình 5: Chia dữ liệu train và validation - Dữ liệu training từ 2020 đến cuối 2024, validation từ đỉnh giá xuống đến đầu 2025.

</p>

Quan sát Hình 5 cho thấy cách chia dữ liệu thành tập training (đường xanh) và validation (đường cam). Dữ liệu training bắt đầu từ khoảng 20-25 k VND (2020) và tăng liên tục đến đỉnh 135-140 k VND (cuối 2024/đầu 2025), thể hiện xu hướng tăng giá dài hạn với nhiều biến động. Đường phân chia train/val (đường đứt nét đen) nằm ở cuối 2024/đầu 2025, ngay tại đỉnh giá. Dữ liệu validation bắt đầu từ đỉnh này và giảm xuống khoảng 120-125 k VND, phản ánh giai đoạn điều chỉnh giá sau khi đạt đỉnh. Cách chia này đảm bảo mô hình được huấn luyện trên dữ liệu lịch sử và đánh giá trên dữ liệu tương lai, phù hợp với bài toán forecasting thực tế.

### 2.5. Phân tích Volatility

Ngoài việc chia dữ liệu, việc phân tích volatility theo thời gian giúp hiểu rõ hơn về tính chất biến động của thị trường. Volatility là một chỉ số quan trọng trong phân tích tài chính, phản ánh mức độ rủi ro và biến động của giá cổ phiếu. Rolling 20-day volatility được tính bằng rolling standard deviation của log returns, sau đó annualized bằng cách nhân với √252.

![Rolling 20 Day Volatility](https://i.ibb.co/0jvqZNZr/51daa69ddb82.png)

<p align="center">

Hình 6: Rolling 20-day volatility - Biến động 20 ngày cho thấy các giai đoạn volatility cao và thấp xen kẽ, phản ánh tính heteroscedastic của thị trường.

</p>

Quan sát Hình 6 cho thấy rolling 20-day volatility biến động mạnh theo thời gian, thể hiện tính heteroscedastic (volatility thay đổi). Volatility bắt đầu ở mức khoảng 0.15 (giữa 2020), sau đó tăng mạnh lên trên 0.6 vào đầu 2021, rồi giảm về khoảng 0.2 (giữa 2021). Đợt tăng thứ hai đạt đỉnh gần 0.6 vào giữa 2022, sau đó giảm mạnh xuống dưới 0.1 (đầu 2023) - mức thấp nhất trong toàn bộ chuỗi. Từ 2023, volatility dao động với các đợt tăng nhỏ hơn (đạt khoảng 0.35 vào cuối 2023 và giữa 2024) và giảm về khoảng 0.15-0.2, cho thấy thị trường ổn định hơn so với giai đoạn trước. Cuối 2024-đầu 2025, volatility có xu hướng tăng lên 0.2-0.25. Các đợt volatility cao thường trùng với các giai đoạn biến động giá lớn, phản ánh sự không ổn định của thị trường. Đặc điểm này đòi hỏi mô hình phải nắm bắt được cả pattern giá trị và pattern volatility để dự đoán chính xác.

---

## Phần 3: Pipeline thử nghiệm - Direct 100d Forecasting

### 3.1. Tổng quan Pipeline

Sau khi phân tích project gốc, nghiên cứu nhận ra cần một pipeline hoàn chỉnh hơn. Pipeline này phải dự đoán 100 ngày thay vì chỉ 5 ngày. Do đó, nghiên cứu thiết kế Pipeline **Direct 100d** với mục tiêu dự báo **log return 100 ngày** cho FPT.

Công thức toán học của pipeline như sau:

$$y_{direct}(t) = lp_{t+100} - lp_t$$

Trong đó, $y_{direct}(t)$ là log return trực tiếp trong 100 ngày từ thời điểm $t$, $lp_t$ là log giá tại thời điểm $t$, và $lp_{t+100}$ là log giá tại thời điểm $t+100$. Công thức này tính toán sự thay đổi log giá trong khoảng thời gian 100 ngày, đây là target mà mô hình cần dự đoán.

Sau đó, nghiên cứu chuyển sang **endpoint price**:

$$P_{t+100}^{pred} = \exp(lp_t + \hat{y}_{direct}(t))$$

Trong đó, $P_{t+100}^{pred}$ là giá dự đoán tại thời điểm $t+100$, $\hat{y}_{direct}(t)$ là giá trị dự đoán của log return từ mô hình, và $\exp(\cdot)$ là hàm mũ để chuyển đổi từ log giá về giá thực tế. Công thức này kết hợp log giá hiện tại với log return dự đoán để tính ra giá cổ phiếu trong tương lai.

**Cấu trúc Pipeline gồm 9 block chính:**

Pipeline được chia thành 9 block logic:

1. Load và chuẩn hóa dữ liệu gốc
2. Feature engineering
3. Xây target y_direct và time based folds
4. Feature selection nâng cao (ranking 4 chiều)
5. Định nghĩa các model base
6. Tuning hyperparameter trên CV MSE(endpoint price)
7. Train lại với best config và thu OOF
8. Stacking ensemble bằng Ridge trên OOF
9. Inference test + xuất submission cho từng model và ensemble

Mỗi block có vai trò riêng trong quá trình xử lý. Phần tiếp theo trình bày chi tiết từng block.

### 3.2. Block 1: Load data và tiền xử lý cơ bản

Block đầu tiên của pipeline là load và tiền xử lý dữ liệu. Nghiên cứu bắt đầu với file giá lịch sử FPT (OHLCV, ngày, v.v.).

**Input:** File giá lịch sử FPT (OHLCV, ngày, v.v.)

**Các bước xử lý:**

Các bước xử lý được thực hiện như sau:

1. **Đọc DataFrame** với các cột cơ bản: `time, open, high, low, close, volume`
2. **Tính log price:** `lp = log(close)`
3. **Tính daily log return:** `ret_1d = lp.diff(1)`
4. **Tính log volume:** `vol_log = log(volume + 1)`
5. **Tạo phiên bản clipped:** `ret_1d_clip`, `vol_log_clip` để hạn chế outlier
6. **Đảm bảo sort theo `time`** và reset index

**Output:** `df_base` với các cột gốc + lp + ret_1d + vol_log + clip

Sau khi hoàn thành Block 1, dữ liệu cơ bản đã được chuẩn hóa. Tiếp theo, nghiên cứu chuyển sang Block 2 để xây dựng các đặc trưng.

### 3.3. Block 2: Feature engineering

Block 2 là phần quan trọng trong pipeline. Hàm `build_features(df)` được sử dụng để tạo ra một hệ thống đặc trưng phong phú. Hệ thống này giúp mô hình học được các pattern phức tạp của thị trường.

#### 3.3.1. Returns và volatility

Hệ thống đặc trưng bắt đầu với multi horizon log returns, bao gồm các khoảng thời gian 2, 3, 5, 10, 20, 30, 60 và 120 ngày. Volatility được tính bằng rolling standard deviation của ret_1d với các cửa sổ 5, 10, 20, 30, 60 và 120 ngày. Ngoài ra, volatility của log volume cũng được tính với cửa sổ 20 ngày.

#### 3.3.2. Moving average và relative price

Simple moving average được tính trên giá đóng cửa với các cửa sổ 5, 10, 20, 30, 60, 90, 120 và 200 ngày. Exponential moving average sử dụng các cửa sổ 5, 12, 20, 26 và 50 ngày. Tỉ lệ giá so với SMA được tính cho các cửa sổ 5, 10, 20, 60, 90, 120 và 200 ngày.

#### 3.3.3. Drawdown và range

Drawdown được tính với các cửa sổ 20, 60, 120 và 200 ngày. Range intraday được tính từ high và low nếu có sẵn trong dữ liệu.

#### 3.3.4. Volume features

Moving average volume sử dụng các cửa sổ 5, 10, 20 và 60 ngày. Normalized volume được tính với các cửa sổ tương tự. Z score volume sử dụng cửa sổ 20 ngày. Dollar volume được tính bằng tích của giá đóng cửa và volume. Tương quan 20 ngày giữa return và volume cũng được tính toán.

#### 3.3.5. Technical indicators

Hệ thống sử dụng RSI với các cửa sổ 7, 14 và 28 ngày. Bollinger Bands được tính cho cửa sổ 20 và 60 ngày, bao gồm upper band, lower band và width. MACD và MACD nhanh được tính với các thành phần signal và histogram. Stochastic, ATR và CCI được tính nếu có dữ liệu high và low.

#### 3.3.6. Calendar features

Các đặc trưng lịch bao gồm day of week, month, và encoding seasonal bằng sin và cos của month.

#### 3.3.7. Lag features

Tập lag base sử dụng các lags 1, 2, 3, 5, 10 và 20 ngày cho các feature quan trọng như ret_1d_clip, vol_log_clip, ret_5d, ret_20d, ret_60d, vol_5, vol_20, vol_60, vol_120, price_sma, rsi, macd, macd_fast, bb_width, dd, vol_norm, vol_z, stoch_k, stoch_d, cci, atr, dollar_vol và corr_ret_vol_20.

**Output Block 2:** `df_feat` với toàn bộ feature + cột gốc

### 3.4. Block 3: Target direct 100d và Time based CV folds

#### 3.4.1. Target `y_direct`

Hàm từ `targets_direct.py` được sử dụng để tính target. Với horizon H = 100, công thức là $y_{\text{direct}}(t) = lp_{t+H} - lp_t$. Index được căn chỉnh lại để bỏ các dòng không có đủ 100 ngày phía trước.

#### 3.4.2. Time based folds

Hàm `splits.py` được sử dụng để chia dữ liệu thành `n_folds` theo thời gian. Mỗi fold có `train_mask` và `val_mask`. Quá trình này đảm bảo train < val theo time và không rò rỉ tương lai.

Output của Block 3 bao gồm `df_direct` là df_feat có thêm `y_direct`, và `folds` là danh sách các folds với train/val mask.

### 3.5. Block 4: Feature selection nâng cao

Hàm `run_feature_selection_direct` thực hiện ranking 4 chiều. Chiều thứ nhất là Mutual Information với y_direct để đo mức độ liên hệ phi tuyến. Chiều thứ hai là multi scale correlation với y_direct smoothed để đo mức độ đi cùng hướng với xu hướng 100 ngày. Chiều thứ ba là Stability Selection với ElasticNet trên folds để đo độ ổn định của feature. Chiều thứ tư là Predictability Score (AR(1) R²) để đo mức độ có cấu trúc, không phải pure noise.

Final_score được tính bằng công thức: $0.3 \times \text{mi\_norm} + 0.3 \times \text{corr\_norm} + 0.3 \times \text{stability\_norm} + 0.1 \times \text{predictability\_norm}$, trong đó các thành phần được chuẩn hóa (norm) và trọng số được phân bổ: 30% cho Mutual Information (mi_norm), 30% cho correlation (corr_norm), 30% cho stability (stability_norm), và 10% cho predictability (predictability_norm). Công thức này kết hợp bốn tiêu chí đánh giá để xếp hạng features, ưu tiên các features có độ liên hệ cao, tương quan tốt, ổn định và dự đoán được.

Output của Block 4 bao gồm `selected_features` là danh sách top_k features và `rank_df` là DataFrame chi tiết score từng feature.

### 3.6. Block 5-9: Model training và ensemble

Block 5 định nghĩa các model base bao gồm DirectElasticNetModel, DirectRidgeModel, DirectXGBoostModel, DirectLGBMModel, DirectRandomForestModel và DirectGBDTModel. Block 6 thực hiện tuning hyperparameter trên CV MSE của endpoint price. Block 7 train lại với best config và thu OOF (Out-of-Fold) predictions. Block 8 thực hiện stacking ensemble bằng Ridge trên OOF. Block 9 thực hiện inference test và xuất submission cho từng model và ensemble.

### 3.7. Kết quả Pipeline Direct 100d

Từ pipeline này, nhiều mô hình khác nhau đã được thử nghiệm. Kết quả được trình bày trong bảng dưới đây:

<table>
<thead>
<tr>
<th style="text-align: center">Model</th>
<th style="text-align: center">Top_k features</th>
<th style="text-align: center">CV MSE (price)</th>
<th style="text-align: center">Public LB</th>
</tr>
</thead>
<tbody>
<tr>
<td>ElasticNet</td>
<td>100</td>
<td>468.2870</td>
<td>848.4753</td>
</tr>
<tr>
<td>Ridge</td>
<td>100</td>
<td>788.7753</td>
<td>844.6113</td>
</tr>
<tr>
<td>XGBoost</td>
<td>100</td>
<td>209.3468</td>
<td>916.1048</td>
</tr>
<tr>
<td>LightGBM</td>
<td>100</td>
<td>200.8915</td>
<td>859.4504</td>
</tr>
<tr>
<td>Random Forest</td>
<td>100</td>
<td>188.9185</td>
<td>708.9165</td>
</tr>
<tr>
<td>GBDT</td>
<td>100</td>
<td>167.6138</td>
<td>729.2800</td>
</tr>
<tr>
<td><strong>Stack Ridge Ensemble</strong></td>
<td>-</td>
<td>-</td>
<td><strong>299.7927</strong></td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 3: Kết quả Pipeline Direct 100d - So sánh hiệu suất của các mô hình khác nhau (ElasticNet, LightGBM, Random Forest, GBDT) với các top_k features khác nhau, bao gồm CV MSE và Public Leaderboard score.

</p>

Phân tích kết quả cho thấy ensemble cho kết quả tốt nhất với MSE 299.7927. Random Forest và GBDT cho kết quả tốt trong các model đơn lẻ. Pipeline này là nền tảng cho các thử nghiệm tiếp theo. Kết quả này cho thấy pipeline đã hoạt động tốt. Tuy nhiên, nghiên cứu vẫn cần cải thiện thêm để đạt được kết quả tốt hơn.

---

## Phần 4: Kết quả thử nghiệm Elastic Net

Sau khi xây dựng pipeline Direct 100d, hai phương pháp chính sử dụng Elastic Net tiếp tục được thử nghiệm. Phương pháp đầu tiên là **OLS Elastic Net** (Trend + Residual Model). Phương pháp thứ hai là **Rolling Elastic Net**. Cả hai phương pháp đều có cách tiếp cận khác nhau để dự đoán giá cổ phiếu FPT 100 ngày.

### 4.1. OLS Elastic Net - Trend + Residual Model

#### 4.1.1. Ý tưởng phương pháp

Phương pháp OLS Elastic Net được phát triển với ý tưởng tách bài toán thành hai tầng. Cách tiếp cận này dựa trên nhận định rằng giá cổ phiếu có hai thành phần chính.

1. **Mô hình xu hướng dài hạn (TrendModel):** Học xu hướng trơn của `log_price` theo thời gian bằng mô hình đa thức bậc thấp
2. **Mô hình residual:** Học residual (log_price trừ trend) bằng ElasticNet hoặc Ridge

**Lý do:**

Giá cổ phiếu thường có xu hướng dài hạn tương đối mượt. Phần nhiễu ngắn hạn là biến động quanh xu hướng đó. Việc tách riêng giúp residual model tập trung mô tả cấu trúc ngắn hạn. Do đó, mô hình có thể học được cả xu hướng dài hạn và biến động ngắn hạn.

#### 4.1.2. Cấu trúc Pipeline

Pipeline gồm 6 khối chính. Khối đầu tiên là cấu hình thí nghiệm và tiền xử lý dữ liệu gốc. Khối thứ hai là mô hình xu hướng dài hạn (TrendModel). Khối thứ ba xây dựng đặc trưng kỹ thuật và dataset supervised cho residual. Khối thứ tư huấn luyện residual model, chọn đặc trưng lõi và tối ưu hyperparameter. Khối thứ năm thực hiện forecast nhiều bước (100 ngày) và kiểm tra trên validation. Khối cuối cùng dự báo 100 ngày cuối cùng và tạo submission.

#### 4.1.3. TrendModel

TrendModel sử dụng `PolynomialFeatures(degree)` với `include_bias=True` và `LinearRegression` để fit trên `log_price`. Input là cột `t` (index thời gian), được biến đổi sang ma trận đa thức $[1, t, t^2, \ldots, t^{\text{degree}}]$, trong đó mỗi cột đại diện cho một bậc của đa thức (bậc 0 là hằng số, bậc 1 là tuyến tính, bậc 2 là bậc hai, v.v.). Mô hình này cho phép nắm bắt xu hướng dài hạn của giá cổ phiếu thông qua việc fit một đường cong đa thức. Đặc điểm quan trọng là xu hướng dài hạn được ước lượng bằng toàn bộ lịch sử (2020-2025), sau đó dùng chung cho mọi phần sau.

#### 4.1.4. Feature Engineering cho Residual

Hệ thống đặc trưng kỹ thuật bao gồm return và log return với các cửa sổ trong RET_WINDOWS, rolling volatility được tính bằng rolling_std của log_ret_1d, Simple Moving Average và tỉ lệ giá so với SMA, các đặc trưng volume, RSI 14 phiên, và lag features với mỗi feature và mỗi lag từ 0 đến MAX_LAG.

#### 4.1.5. Feature Selection và Hyperparameter Tuning

Quy trình feature selection và hyperparameter tuning gồm ba bước. Bước đầu tiên là tính permutation importance của từng feature. Bước thứ hai là chọn top K features và loại bỏ các feature bắt đầu bằng "resid_lag" để tránh tích lũy sai số. Bước cuối cùng là random search để tối ưu hyperparameters cho ElasticNet (alpha, l1_ratio).

#### 4.1.6. Forecast nhiều bước

Quy trình recursive multi-step forecast bắt đầu bằng việc chuẩn hóa history và tính lại trend, residual. Sau đó, features được build cho điểm hiện tại. Residual tiếp theo được dự báo bằng residual model. Log_price được tái dựng bằng công thức $\text{log\_price\_next} = \text{last\_log\_price} + \text{resid\_next}$, trong đó $\text{last\_log\_price}$ là log giá của bước trước đó và $\text{resid\_next}$ là residual dự đoán từ mô hình. Công thức này tái dựng log giá mới bằng cách cộng residual dự đoán vào log giá hiện tại, phản ánh cách tiếp cận decomposition: giá = trend + residual. Giá trị `log_price_next` được clip vào biên `[log_clip_low, log_clip_high]` để tránh giá trị bất thường. Cuối cùng, giá được chuyển về price và append vào history. Quá trình này được lặp lại 100 lần.

#### 4.1.7. Kết quả OLS Elastic Net

Kết quả sau khi áp dụng OLS Elastic Net trên validation cho thấy MSE 100 ngày là 5406.9571. Phân tích kết quả cho thấy phương pháp này cho kết quả khá tốt với cách tiếp cận tách trend và residual. Tuy nhiên, MSE vẫn còn cao so với yêu cầu và cần cải thiện thêm để xử lý các biến động lớn. Kết quả này cho thấy phương pháp có tiềm năng nhưng vẫn cần cải thiện. Do đó, phương pháp Rolling Elastic Net được thử nghiệm.

### 4.2. Rolling Elastic Net

#### 4.2.1. Ý tưởng phương pháp

Phương pháp **Rolling Elastic Net** được phát triển với cách tiếp cận khác. Tại mỗi thời điểm t, mô hình ElasticNet được **train lại từ đầu** chỉ bằng dữ liệu trước thời điểm đó. Sau đó, mô hình dự báo cho điểm t.

**Ưu điểm:**

Phương pháp này có ưu điểm là mô phỏng đúng cách mô hình hoạt động trong thực tế. Khi phải dự báo online, mô hình luôn được cập nhật với dữ liệu mới nhất. Do đó, mô hình có thể thích ứng với các biến động của thị trường.

#### 4.2.2. Cấu trúc Pipeline

Pipeline gồm 6 khối. Khối đầu tiên là tiền xử lý dữ liệu gốc. Khối thứ hai xây dựng hệ thống đặc trưng (feature engineering). Khối thứ ba chuẩn hóa dữ liệu và tách tập validation. Khối thứ tư thực hiện Rolling ElasticNet Forecasting. Khối thứ năm đánh giá mô hình trên return space và price space. Khối cuối cùng huấn luyện mô hình cuối cùng và tạo submission.

![Rolling Elastic Net Pipeline Structure](https://i.ibb.co/C3xmm41x/3a095b805d8a.png)

<p align="center">

Hình 7: Cấu trúc Pipeline Rolling Elastic Net - Flowchart chi tiết hiển thị luồng xử lý từ dữ liệu OHLCV đầu vào qua các bước tiền xử lý, feature engineering, normalization, rolling forecasting, evaluation, đến dự đoán 100 ngày. Phần dưới mở rộng chi tiết quy trình Rolling ElasticNet với window selection, kiểm tra variance, training và prediction.

</p>

Quan sát Hình 7 cho thấy cấu trúc pipeline hoàn chỉnh của phương pháp Rolling Elastic Net. **Luồng chính (hàng trên)** bắt đầu từ dữ liệu OHLCV đầu vào, trải qua các bước: Data Preprocessing (tính log-return, log-volume change, winsorize outliers), Feature Engineering (xây dựng lag features, technical indicators, calendar features), Normalization (áp dụng StandardScaler trên training data, chia train/validation), Rolling ElasticNet (mô hình chính với cơ chế rolling), Evaluation (đánh giá trên return-space và price-space, calibrate predictions), và cuối cùng là Forecast 100 days (dự đoán đệ quy 100 bước). **Luồng chi tiết Rolling ElasticNet (hàng dưới)** mở rộng bước Rolling ElasticNet thành các bước con: Window Selection (chọn cửa sổ dữ liệu), Check Variance (bỏ qua nếu variance không đủ), ElasticNet Training (fit model với alpha, l1_ratio), và Prediction (dự đoán X[i] và lưu trữ). Cấu trúc này phản ánh cách tiếp cận rolling window, nơi mô hình được train lại tại mỗi thời điểm chỉ với dữ liệu trước đó, mô phỏng đúng cách hoạt động trong thực tế khi dự báo online.

#### 4.2.3. Tiền xử lý và Feature Engineering

Bước đầu tiên là tính log-return và log-volume change với công thức $\text{ret\_1d} = \log(\text{close}_t) - \log(\text{close}_{t-1})$ (tương tự như daily log return đã giải thích ở trên) và $\text{vol\_chg} = \log(\text{volume}_t+1) - \log(\text{volume}_{t-1}+1)$, trong đó $\text{volume}_t$ là khối lượng giao dịch tại thời điểm $t$ và số 1 được cộng vào để tránh log(0) khi volume bằng 0. Công thức này đo lường sự thay đổi logarit của khối lượng giao dịch, giúp nắm bắt biến động của volume một cách ổn định. Bước thứ hai là winsorize (làm mượt outlier) bằng cách clip giá trị `ret_1d` và `vol_chg` dựa trên phân phối dữ liệu trước ngày validation, tạo ra `ret_1d_clipped` và `vol_chg_clipped`. Bước thứ ba là feature engineering, bao gồm lag features cho ret_1d_clipped với các lags 1, 2, 3, 5, 10, 20, 60, 120, volume features là lag của volume change, rolling statistics (mean, std, min, max) của price và volume, technical indicators như SMA, Bollinger Bands, ATR, trend indicator, drawdown, và calendar features như day of week, month (sin, cos). Target là $y_t = \text{ret\_1d\_clipped}_{t+1}$, dự báo return của ngày kế tiếp.

#### 4.2.4. Rolling Forecasting

Hai loại cửa sổ được sử dụng. Loại thứ nhất là sliding window, sử dụng `window_size` quan sát gần nhất. Loại thứ hai là expanding window, sử dụng toàn bộ dữ liệu lịch sử từ đầu đến i.

Cơ chế rolling hoạt động như sau. Với mỗi i ≥ window_size, cửa sổ train đúng loại được chọn. Nếu y_train không đủ phân tán thì bỏ qua. Một ElasticNet mới được fit và predict tại điểm X[i], sau đó lưu vào preds[i].

#### 4.2.5. Đánh giá mô hình

Đánh giá trên return-space bao gồm tính MSE trên validation, sau đó calibrate lại bằng tuyến tính với công thức $y_{\text{true}} = a + b \times y_{\text{pred}}$, trong đó $a$ là intercept và $b$ là hệ số điều chỉnh, được học từ dữ liệu validation để điều chỉnh predictions về gần với giá trị thực tế hơn. Công thức này giúp bù đắp bias hệ thống của mô hình. Đánh giá trên price-space bao gồm biến đổi return đã calibrate thành giá bằng công thức $P_t = P_{t-1} \times e^{r_t}$, trong đó $P_t$ là giá tại thời điểm $t$, $P_{t-1}$ là giá tại thời điểm trước đó, và $r_t$ là log return đã calibrate. Công thức này sử dụng hàm mũ để chuyển đổi log return về giá, phản ánh tính chất compound của lợi nhuận. So sánh với giá thực bằng MSE trên price, và ensemble với mô hình naïve (giá t bằng giá t–1).

#### 4.2.6. Forecast 100 ngày

Dự báo 100 bước được thực hiện bằng cách dùng mô hình final predict return t+1, update lại giá (compound), append dữ liệu giả lập vào dataframe, rebuild feature cho điểm t+2, và tiếp tục predict đến 100 ngày.

#### 4.2.7. Kết quả Rolling Elastic Net

Kết quả sau khi áp dụng Rolling Elastic Net trên public leaderboard cho thấy MSE là 52.9284. So sánh với OLS Elastic Net cho thấy Rolling Elastic Net cho kết quả tốt hơn đáng kể (MSE: 52.9284 vs 5406.9571). Phương pháp rolling phù hợp hơn với dữ liệu time series có nhiều biến động. Kết quả này cho thấy phương pháp rolling hiệu quả hơn. Tuy nhiên, nghiên cứu vẫn cần phân tích kỹ hơn để hiểu rõ nguyên nhân.

![Rolling Elastic Net: Groundtruth vs Predictions](https://i.ibb.co/JRdkccj2/3cb382871c32.png)

<p align="center">

Hình 8: So sánh Groundtruth vs Predictions - Rolling Elastic Net - Groundtruth (đường xanh đậm) có biến động phức tạp với nhiều đợt tăng giảm, trong khi Predict (đường xanh nhạt) có xu hướng giảm đều đặn và mượt, dẫn đến bias âm lớn ở nửa sau khi groundtruth tăng trở lại nhưng predictions vẫn tiếp tục giảm.

</p>

Quan sát Hình 8 cho thấy so sánh giữa groundtruth và predictions của phương pháp Rolling Elastic Net. **Groundtruth** (đường xanh đậm) bắt đầu từ ~118, giảm xuống ~90 ở x=20, sau đó tăng lên ~112 ở x=85, rồi giảm xuống ~107 ở cuối, có biến động phức tạp với nhiều đợt tăng giảm. **Predict** (đường xanh nhạt) có xu hướng giảm đều đặn và mượt từ ~116 xuống ~79 ở cuối, không bắt kịp các biến động của groundtruth. Trong nửa đầu (x=1-40), predictions và groundtruth có độ tương đồng tương đối, nhưng ở nửa sau (x=40-99), bias âm lớn xuất hiện khi groundtruth tăng trở lại nhưng predictions vẫn tiếp tục giảm, dẫn đến divergence ngày càng lớn (ở x=99, groundtruth ~107 trong khi predict ~79). 

**Lý giải tại sao mô hình dự đoán giảm**: Mô hình Rolling Elastic Net có xu hướng dự đoán giảm vì nó học từ dữ liệu gần nhất trong rolling window. Khi validation period bắt đầu từ đỉnh giá (cuối 2024/đầu 2025, ~135-140 k VND), dữ liệu gần nhất mà mô hình thấy là giai đoạn giảm giá sau đỉnh. Mô hình học được pattern giảm giá này và áp dụng vào predictions, dẫn đến xu hướng giảm đều đặn. Ngoài ra, Rolling Elastic Net sử dụng rolling window với dữ liệu gần nhất, nên nó chủ yếu nhìn thấy pattern ngắn hạn (giảm giá) mà không nắm bắt được xu hướng dài hạn (tăng giá sau khi giảm). Điều này cho thấy mô hình có xu hướng dự đoán theo pattern đã học từ quá khứ mà không kịp thích ứng với các biến động mới, đặc biệt là các xu hướng dài hạn. Vấn đề này nghiêm trọng trong bối cảnh có sự kiện "thiên nga đen", khi thị trường có biến động lớn mà mô hình chưa từng gặp trong dữ liệu training.

### 4.3. So sánh hai phương pháp Elastic Net

Bảng dưới đây trình bày kết quả so sánh hai phương pháp Elastic Net:

<table>
<thead>
<tr>
<th style="text-align: center">Phương pháp</th>
<th style="text-align: center">MSE</th>
<th style="text-align: center">Ưu điểm</th>
<th style="text-align: center">Nhược điểm</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>OLS Elastic Net</strong></td>
<td>5406.9571</td>
<td>- Tách rõ trend và residual<br>- Dễ interpret</td>
<td>- MSE cao<br>- Khó xử lý biến động lớn</td>
</tr>
<tr>
<td><strong>Rolling Elastic Net</strong></td>
<td>52.9284</td>
<td>- Mô phỏng thực tế<br>- Tự động adapt với dữ liệu mới<br>- MSE thấp hơn</td>
<td>- Tốn thời gian train<br>- Phức tạp hơn</td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 4: So sánh hai phương pháp Elastic Net - So sánh OLS Elastic Net và Rolling Elastic Net về MSE, ưu điểm và nhược điểm của từng phương pháp.

</p>

**Kết luận:**

Phân tích so sánh cho thấy Rolling Elastic Net cho kết quả tốt hơn. Tuy nhiên, vẫn cần cải thiện thêm để đạt được kết quả tối ưu. Do đó, các mô hình SOTA hiện đại hơn được thử nghiệm.

---

## Phần 5: Quá trình tìm ra kết quả tốt nhất

Sau khi thử nghiệm các phương pháp Elastic Net,  một cách tiếp cận hoàn toàn khác để đạt được kết quả tốt hơn cần được xây dựng. Phần này mô tả quá trình 4 bước để tìm ra phương pháp tối ưu.

### Bước 1: Phân tích lại vấn đề và phát triển Bias Correction

#### 5.1.1. Phát hiện vấn đề

Sau khi thử nghiệm các mô hình Linear và Elastic Net, một vấn đề quan trọng được phát hiện: **mô hình học vẫn chưa đạt kết quả như mong muốn có thể là do các dữ liệu cổ phiếu test thực tế đang xảy ra trường hợp mà mô hình không thể học được tự động từ cách thông thường.**

Nguyên nhân chính gồm ba điểm. Điểm đầu tiên là sự kiện "thiên nga đen". [**Cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge) này diễn ra trong bối cảnh có sự kiện chưa có tiền lệ - khi Tổng thống Donald Trump công bố sắc thuế và áp thuế rất nặng đối với hàng hóa xuất khẩu của Việt Nam. Đây là một sự kiện rất khó dự đoán được và không có trong dữ liệu training. Điểm thứ hai là bias hệ thống. Các mô hình có xu hướng dự đoán cao hơn hoặc thấp hơn giá trị thực tế một cách có hệ thống (systematic bias). Điểm thứ ba là không phản ứng với biến động. Mô hình không có cơ chế để phản ứng với các biến động lớn từ dữ liệu validation/test.

#### 5.1.2. Giải pháp: Phát triển cơ chế Bias Correction

Một cơ chế **kiểm soát độ lệch từ các dữ liệu dự đoán và smooth kết quả** được phát triển. Cơ chế này giúp kết quả phản ánh đúng thực tế hơn.

Nghiên cứu đề xuất ý tưởng sử dụng TimeSeriesSplit để tạo nhiều folds từ dữ liệu training. Với mỗi fold, model được train và predict trên validation fold. Các cặp (prediction, ground_truth) được thu thập từ tất cả các folds. Một mô hình điều chỉnh (correction model) được học từ các cặp này. Cuối cùng, mô hình điều chỉnh được áp dụng cho predictions cuối cùng. Cách tiếp cận này cho phép mô hình học được pattern của bias từ validation folds. Do đó, mô hình có thể điều chỉnh predictions cho phù hợp với thực tế.

#### 5.1.3. Các phương pháp Bias Correction được thử nghiệm

Từ notebook `patchtst_improved_nonlinear.ipynb`, nghiên cứu thử nghiệm 4 phương pháp chính:

**1. Bias Correction (Trừ bias trung bình)**

```python
# Tính bias trên validation set sử dụng TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=3)
bias_estimates = []

# Tính bias trên các validation folds
full_data = train_nf_full.copy()
for train_idx, val_idx in tscv.split(full_data):
    train_fold = full_data.iloc[train_idx]
    val_fold = full_data.iloc[val_idx]
    
    # Train model trên fold
    model_fold = PatchTST(
        h=min(horizon, len(val_fold)),
        input_size=best_params_patchtst['input_size'],
        patch_len=best_params_patchtst['patch_len'],
        stride=best_params_patchtst['stride'],
        revin=True,
        learning_rate=best_params_patchtst['learning_rate'],
        max_steps=best_params_patchtst['max_steps'],
        val_check_steps=10,
    )
    nf_fold = NeuralForecast(models=[model_fold], freq='D')
    nf_fold.fit(df=train_fold, val_size=0)
    forecast_fold = nf_fold.predict()
    
    pred_col = [col for col in forecast_fold.columns if col not in ['unique_id', 'ds']][0]
    pred_fold = forecast_fold[pred_col].values[:len(val_fold)]
    true_fold = val_fold['y'].values[:len(pred_fold)]
    
    # Tính bias
    bias_fold = np.mean(pred_fold - true_fold)
    bias_estimates.append(bias_fold)

# Tính bias trung bình
estimated_bias = np.mean(bias_estimates)

# Áp dụng bias correction
pred_corrected = pred_baseline - estimated_bias
```

**Giải thích code:**
- Vòng lặp `for fold in folds`: Duyệt qua từng validation fold trong TimeSeriesSplit để tính bias riêng của từng fold.
- `bias_fold = np.mean(pred_fold - true_fold)`: Tính bias của một fold bằng cách lấy trung bình của hiệu số giữa predictions và ground truth. Nếu bias dương, mô hình có xu hướng overestimate; nếu âm, mô hình có xu hướng underestimate.
- `estimated_bias = np.mean(bias_estimates)`: Tính bias tổng thể bằng trung bình của tất cả các bias từ các folds, đảm bảo ước lượng ổn định hơn.
- `pred_corrected = pred_baseline - estimated_bias`: Điều chỉnh tất cả predictions bằng cách trừ đi bias ước lượng. Đây là cách đơn giản nhất để giảm bias hệ thống.

Phương pháp này đơn giản và dễ implement, nhưng chỉ điều chỉnh một giá trị cố định và không học được pattern phức tạp.

Phương pháp thứ hai là Residual Correction Model. Một mô hình phụ được train để dự đoán residuals (sai số). Phương pháp này học được pattern của residuals nhưng phức tạp hơn Bias Correction đơn giản và cần thêm features để dự đoán residuals.

Phương pháp thứ ba là Isotonic Calibration, sử dụng Isotonic Regression để calibrate predictions. Phương pháp này đảm bảo thứ tự của predictions được giữ nguyên và không giả định quan hệ tuyến tính, nhưng có thể overfit với dữ liệu ít.

Phương pháp thứ tư là Post-processing Regression, phương pháp tốt nhất. Một mô hình Linear Regression được học để map từ predictions sang actual values:

```python
# Train linear regression để map predictions -> actual values
tscv = TimeSeriesSplit(n_splits=3)
X_post = []
y_post = []

full_data = train_nf_full.copy()
for train_idx, val_idx in tscv.split(full_data):
    train_fold = full_data.iloc[train_idx]
    val_fold = full_data.iloc[val_idx]
    
    model_fold = PatchTST(
        h=min(horizon, len(val_fold)),
        input_size=best_params_patchtst['input_size'],
        patch_len=best_params_patchtst['patch_len'],
        stride=best_params_patchtst['stride'],
        revin=True,
        learning_rate=best_params_patchtst['learning_rate'],
        max_steps=best_params_patchtst['max_steps'],
        val_check_steps=10,
    )
    nf_fold = NeuralForecast(models=[model_fold], freq='D')
    nf_fold.fit(df=train_fold, val_size=0)
    forecast_fold = nf_fold.predict()
    
    pred_col = [col for col in forecast_fold.columns if col not in ['unique_id', 'ds']][0]
    pred_fold = forecast_fold[pred_col].values[:len(val_fold)]
    true_fold = val_fold['y'].values[:len(pred_fold)]
    
    X_post.extend(pred_fold.reshape(-1, 1))
    y_post.extend(true_fold)

X_post = np.array(X_post)
y_post = np.array(y_post)
post_model = LinearRegression()
post_model.fit(X_post, y_post)

# Áp dụng
pred_corrected = post_model.predict(pred_baseline.reshape(-1, 1))
```

**Giải thích code:**
- `TimeSeriesSplit(n_splits=3)`: Chia dữ liệu thành 3 folds theo thời gian (không shuffle), đảm bảo validation data luôn ở sau training data.
- Vòng lặp `for train_idx, val_idx in tscv.split(full_data)`: Duyệt qua từng fold, train mô hình trên training data và predict trên validation data để thu thập cặp (prediction, ground_truth).
- `X_post.extend(predictions)`: Thu thập tất cả predictions từ các validation folds làm input cho post-processing model.
- `y_post.extend(ground_truth[val_idx])`: Thu thập tất cả actual values từ các validation folds làm target cho post-processing model.
- `post_model.fit(X_post, y_post)`: Train Linear Regression để học công thức điều chỉnh dạng $y = \text{coef} \times \text{pred} + \text{intercept}$, giúp chuyển đổi predictions từ baseline sang giá trị gần với actual hơn.
- `pred_corrected = post_model.predict(pred_baseline)`: Áp dụng mô hình đã train để điều chỉnh predictions từ baseline, tạo ra predictions đã được điều chỉnh bias.

Phương pháp này học được quan hệ giữa predictions và actual values, đơn giản nhưng hiệu quả, và có thể điều chỉnh cả scale và bias.

#### 5.1.4. Kết quả thử nghiệm

Từ notebook `patchtst_improved_nonlinear.ipynb`, kết quả quan trọng đã được ghi nhận. Kết quả cho thấy mô hình đã có phản ứng với các sai lệch từ dữ liệu các fold được tạo ra từ time series Split. Điều này chứng minh rằng phương pháp bias correction có thể giúp mô hình học được cách điều chỉnh predictions cho phù hợp với thực tế.

Phân tích kết quả cho thấy Post-processing Regression cho kết quả tốt nhất. Mô hình học được pattern của bias từ validation folds và có thể áp dụng cho các mô hình khác nhau. Kết quả này khẳng định hiệu quả của phương pháp bias correction.

![PatchTST Improved Nonlinearity Analysis](https://i.ibb.co/YTh3ZMqd/95bda07fe4d5.png)

<p align="center">

Hình 9: Phân tích cải tiến PatchTST cho dữ liệu không tuyến tính - Bốn biểu đồ: (1) Top 3 methods predictions vs thực tế, (2) So sánh volatility, (3) Phân bố error, (4) Scatter plot.

</p>

Quan sát Hình 9 cho thấy phân tích Phương pháp Post-processing được áp dụng trên Ensemble Model. **Biểu đồ trên trái** so sánh top 3 phương pháp tốt nhất (Ensemble + Post-processing với Linear MSE 56.11, Ridge alpha=1.0 MSE 56.11, Ridge alpha=10.0 MSE 56.14) với giá thực tế. Cả ba phương pháp đều theo sát xu hướng giá thực tế, đặc biệt trong các giai đoạn biến động (khoảng time step 20-30 và sau đó). **Biểu đồ trên phải** so sánh volatility (std của differences): PatchTST_Best có volatility ~1.5, Ensemble ~1.1, Ensemble + Post-processing (Linear) ~0.9, trong khi Thực tế có volatility cao nhất ~2.1. Điều này cho thấy post-processing giảm volatility đáng kể so với thực tế, phản ánh predictions mượt hơn. **Biểu đồ dưới trái** hiển thị phân bố error của Ensemble + Post-processing (Linear): errors tập trung quanh 0 với đỉnh cao nhất ở khoảng -1 đến 0, mean error = 1.93, cho thấy model có xu hướng under-predict nhẹ. **Biểu đồ dưới phải** là scatter plot, các điểm phân tán quanh đường y=x nhưng có độ lệch ở các giá trị cao. Kết quả này cho thấy ensemble kết hợp post-processing đạt MSE thấp nhất (56.11) so với PatchTST baseline (212.37), cải thiện 73.58%, nhưng volatility vẫn thấp hơn thực tế (ratio 0.42), cho thấy predictions còn khá mượt.

#### 5.1.5. Về sự kiện "thiên nga đen"

![Black Swan Event](https://i.ibb.co/Wv3DMwgF/7aa3b4bbfbc7.jpg)

<p align="center">

Hình 10: Sự kiện "Thiên nga đen" - Minh họa về sự kiện chưa có tiền lệ trong thị trường chứng khoán, khi Tổng thống Donald Trump công bố sắc thuế đối với hàng hóa xuất khẩu của Việt Nam. Sự kiện này không xuất hiện trong dữ liệu training nhưng có tác động lớn đến giá cổ phiếu.

</p>

Trong thời gian bộ dữ liệu test được áp dụng, Tổng thống Donald Trump đã công bố các chính sách thuế mới đối với hàng hóa xuất khẩu của Việt Nam. Đây là một sự kiện chưa có tiền lệ, không có trong dữ liệu lịch sử, có ảnh hưởng lớn và tác động mạnh đến giá cổ phiếu, và rất khó dự đoán vì không thể dự đoán được từ dữ liệu training.

Sự kiện "thiên nga đen" có tác động lớn đến bài toán. Dữ liệu test có thể chứa các biến động lớn do sự kiện này. Mô hình thông thường không thể học được pattern này. Do đó, cần cơ chế đặc biệt để xử lý các biến động lớn.

Để xử lý vấn đề này, bias correction đã được sử dụng để điều chỉnh predictions, smooth kết quả để tránh phản ứng quá mức, và kết hợp nhiều phương pháp để tăng độ tin cậy. Các giải pháp này giúp mô hình xử lý tốt hơn các biến động lớn trong dữ liệu test.

### Bước 2: So sánh toàn bộ các mô hình SOTA hiện tại

Sau khi phát triển được cơ chế bias correction, nghiên cứu so sánh toàn bộ các mô hình SOTA hiện tại. Mục tiêu là tìm ra mô hình tốt nhất. Notebook `comprehensive_comparison.ipynb` được tạo ra để thực hiện việc này.

#### 5.2.1. Các mô hình SOTA được so sánh

Sáu mô hình SOTA được so sánh. Mô hình đầu tiên là PatchTST (Patch-based Time Series Transformer), một mô hình Transformer chuyên biệt cho time series, sử dụng patch-based approach để capture local patterns và có Reversible Instance Normalization (RevIN). Mô hình thứ hai là TimesNet, dựa trên 2D convolution, chuyển đổi time series thành 2D representation và hiệu quả với dữ liệu có pattern phức tạp.


Mô hình thứ ba là DLinear, một mô hình Linear với decomposition, phân rã thành trend và seasonal components, đã được thử nghiệm trong project gốc. Mô hình thứ tư là NLinear, một mô hình Linear với normalization, chuẩn hóa bằng cách trừ giá trị cuối cùng, đã cho kết quả tốt nhất trong project gốc. Mô hình thứ năm là NBEATS (Neural Basis Expansion Analysis for Time Series), dựa trên basis expansion, sử dụng backward và forward residual links, hiệu quả với dữ liệu có trend và seasonality. Mô hình thứ sáu là NHITS (Neural Hierarchical Interpolation for Time Series), một mô hình hierarchical với interpolation, sử dụng multi-rate sampling và hiệu quả với dữ liệu multi-scale.

![SOTA Models Comparison](https://i.ibb.co/s95xbJhJ/fc405003bbd4.png)

<p align="center">

Hình 11: So sánh các mô hình SOTA - Biểu đồ trên trái so sánh predictions vs thực tế của 7 models, trên phải là scatter plot NLinear, dưới trái là bar chart so sánh MSE, dưới phải là histogram phân bố error của Ensemble.

</p>

Quan sát Hình 11 cho thấy so sánh hiệu suất của các mô hình SOTA qua 4 subplots. **Biểu đồ trên trái (So sánh Predictions vs Thực tế)**: So sánh 7 models (PatchTST, TimesNet, DLinear, NLinear, NBEATS, NHITS, Ensemble) với actual values. Tất cả models đều consistently overestimate, đặc biệt sau time step 20, với predictions từ 120-160 k VND trong khi actual values chủ yếu dưới 110 k VND. Ensemble (đường đỏ nét đứt) mượt hơn nhưng vẫn overpredict đáng kể. **Biểu đồ trên phải (Scatter Plot - NLinear)**: Hầu hết điểm nằm phía trên đường y=x, xác nhận NLinear consistently overpredict. **Biểu đồ dưới trái (So sánh MSE)**: NLinear có MSE thấp nhất (446.75), tiếp theo là NBEATS (761.31), PatchTST (916.60), TimesNet (927.91), NHITS (1092.16), và DLinear cao nhất (1681.51). **Biểu đồ dưới phải (Phân bố Error - Ensemble)**: Tất cả errors đều âm (từ -35 đến 0), xác nhận Ensemble consistently overpredict. Errors tập trung chủ yếu trong khoảng -35 đến -25, với tần suất cao nhất quanh -32 đến -28. So sánh này cho thấy tất cả models đều có bias lạc quan, phản ánh xu hướng uptrend trong training data, và NLinear đạt hiệu suất tốt nhất trong số các models được so sánh.


#### 5.2.2. Các phương pháp Bias Correction được áp dụng

Với mỗi mô hình SOTA, năm phương pháp bias correction được áp dụng. Phương pháp đầu tiên là Baseline, không có bias correction. Phương pháp thứ hai là Bias Correction, trừ bias trung bình. Phương pháp thứ ba là Post-processing Linear, sử dụng Linear Regression. Phương pháp thứ tư là Post-processing Ridge, sử dụng Ridge Regression. Phương pháp thứ năm là Post-processing Lasso, sử dụng Lasso Regression.

#### 5.2.3. Kết quả so sánh

Từ notebook `comprehensive_comparison.ipynb`, kết quả so sánh tất cả phương pháp (sắp xếp theo MSE):

<table>
<thead>
<tr>
<th style="text-align: center">Method</th>
<th style="text-align: center">Type</th>
<th style="text-align: center">MSE</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>NLinear_PostProcessing_Lasso</strong></td>
<td>Bias Correction</td>
<td><strong>98.5245</strong></td>
</tr>
<tr>
<td>NLinear_PostProcessing_Ridge</td>
<td>Bias Correction</td>
<td>99.2884</td>
</tr>
<tr>
<td>NLinear_PostProcessing_Linear</td>
<td>Bias Correction</td>
<td>99.3094</td>
</tr>
<tr>
<td>PatchTST</td>
<td>SOTA Model</td>
<td>172.5861</td>
</tr>
<tr>
<td>NHITS</td>
<td>SOTA Model</td>
<td>221.8049</td>
</tr>
<tr>
<td>NBEATS</td>
<td>SOTA Model</td>
<td>363.8951</td>
</tr>
<tr>
<td>TimesNet</td>
<td>SOTA Model</td>
<td>422.9592</td>
</tr>
<tr>
<td>NLinear_BiasCorrection</td>
<td>Bias Correction</td>
<td>443.1353</td>
</tr>
<tr>
<td>NLinear</td>
<td>SOTA Model</td>
<td>587.3829</td>
</tr>
<tr>
<td>NLinear_Baseline</td>
<td>Bias Correction</td>
<td>587.3829</td>
</tr>
<tr>
<td>DLinear</td>
<td>SOTA Model</td>
<td>715.4256</td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 5: So sánh tất cả phương pháp - Kết quả so sánh tất cả các phương pháp đã thử nghiệm, sắp xếp theo MSE, bao gồm các mô hình SOTA, Linear models và các phương pháp bias correction.

</p>

#### 5.2.4. Phân tích kết quả

Phân tích kết quả so sánh cho thấy các điểm quan trọng sau. Điểm đầu tiên là NLinear với Post-processing cho kết quả tốt nhất, với NLinear_PostProcessing_Lasso đạt MSE = 98.5245 (tốt nhất), NLinear_PostProcessing_Ridge đạt MSE = 99.2884, và NLinear_PostProcessing_Linear đạt MSE = 99.3094.

Điểm thứ hai là PatchTST là mô hình SOTA tốt nhất, với PatchTST baseline đạt MSE = 172.5861, tốt hơn các mô hình SOTA khác như TimesNet, NBEATS, NHITS.

Điểm thứ ba là Post-processing hiệu quả hơn Bias Correction đơn giản. NLinear_PostProcessing_Lasso đạt MSE = 98.5245, trong khi NLinear_BiasCorrection đạt MSE = 443.1353. Post-processing cải thiện đáng kể so với bias correction đơn giản.

Điểm thứ tư là NLinear vẫn là mô hình Linear tốt nhất. NLinear baseline đạt MSE = 587.3829, trong khi DLinear đạt MSE = 715.4256. NLinear ổn định và hiệu quả hơn.

**Nhận định quan trọng:** Kết quả thực nghiệm khẳng định rằng **NLinear là mô hình tốt nhất trong nhóm các mô hình tuyến tính (Linear, NLinear, DLinear)**, phù hợp với mục tiêu của cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge. Không chỉ trong baseline, mà khi kết hợp với Post-processing, NLinear đạt MSE 98.52, tốt hơn đáng kể so với nhiều mô hình phức tạp như TimesNet (422.96), NBEATS (363.90), và NHITS (221.80). Điều này chứng minh rằng mô hình tuyến tính đơn giản, khi được áp dụng đúng cách và kết hợp với kỹ thuật điều chỉnh bias phù hợp, có thể đạt hiệu quả tốt trong dự báo time series dài hạn với dữ liệu tài chính. NLinear với Post-processing Lasso không chỉ là phương pháp tốt nhất trong nhóm mô hình tuyến tính, mà còn là một giải pháp cân bằng tốt giữa hiệu quả, tính đơn giản, tốc độ và khả năng giải thích, phù hợp với mục tiêu của cuộc thi là khám phá sức mạnh của các mô hình tuyến tính.

![Comprehensive Comparison Overview](https://i.ibb.co/hxnV3bmV/1f03e897acfd.png)

<p align="center">

Hình 12: So sánh tổng quan các phương pháp - Bốn biểu đồ so sánh: (1) Top 3 methods predictions vs thực tế, (2) Top 10 methods theo MSE, (3) Best MSE theo method type, (4) Bias comparison của top 10 methods.

</p>

Quan sát Hình 12 cho thấy so sánh toàn diện các phương pháp qua bốn biểu đồ. **Biểu đồ trên trái** so sánh top 3 phương pháp tốt nhất (NLinear_PostProcessing_Lasso với MSE 98.52, Ridge 99.29, Linear 99.31) với giá thực tế. Cả ba phương pháp đều theo sát xu hướng giá thực tế, đặc biệt trong các giai đoạn biến động (khoảng time step 20-30 và sau đó). **Biểu đồ trên phải** so sánh MSE của top 10 methods: ba phương pháp post-processing đứng đầu với MSE dưới 100, tiếp theo là PatchTST (172.59), NHITS (221.80), NBEATS (363.90), TimesNet (422.96), trong khi NLinear baseline và NLinear_Baseline có MSE cao nhất (587.38). **Biểu đồ dưới trái** so sánh best MSE theo method type: Bias Correction đạt 98.52 (NLinear_PostProcessing_Lasso), thấp hơn đáng kể so với SOTA Model (172.59 từ PatchTST). **Biểu đồ dưới phải** so sánh Bias (Mean Error) của top 10 methods: ba phương pháp post-processing có bias thấp nhất, trong khi NLinear và NLinear_Baseline có bias cao nhất. Kết quả này khẳng định hiệu quả của post-processing trong việc giảm MSE, vượt trội so với các SOTA models phức tạp.

![Top 5 Methods Detailed Comparison](https://i.ibb.co/xSRdDKbL/comprehensive-comparison-2.png)

<p align="center">

Hình 13: So sánh chi tiết top 5 methods - Mỗi phương pháp được so sánh trực tiếp với ground truth, cho thấy độ chính xác và pattern dự đoán của từng method.

</p>

Quan sát Hình 13 cho thấy so sánh chi tiết của top 5 phương pháp tốt nhất, mỗi phương pháp được so sánh trực tiếp với ground truth trong các subplot riêng biệt. Hình ảnh này cho phép phân tích sâu hơn về độ chính xác và pattern dự đoán của từng method. Top 3 methods (NLinear với các biến thể post-processing) cho thấy predictions gần với ground truth nhất, với độ chính xác cao và ít sai số. PatchTST baseline cho thấy hiệu suất tốt nhưng vẫn có một số điểm lệch so với ground truth, đặc biệt là trong các giai đoạn có biến động lớn. NHITS cho thấy hiệu suất tương đối tốt nhưng vẫn kém hơn các phương pháp top 3. Phân tích chi tiết cho thấy các phương pháp post-processing không chỉ cải thiện độ chính xác tổng thể mà còn cải thiện khả năng theo dõi các biến động trong dữ liệu. Các phương pháp tốt nhất có khả năng phản ứng nhanh với các thay đổi trong xu hướng, trong khi các phương pháp kém hơn có xu hướng lag hoặc không bắt kịp các biến động. So sánh này cung cấp insights quan trọng về điểm mạnh và điểm yếu của từng phương pháp, giúp hiểu rõ hơn về lý do tại sao một số phương pháp cho kết quả tốt hơn các phương pháp khác.

![Error Analysis for Best Method](https://i.ibb.co/rGRmrHv7/d7ebef080ebc.png)

<p align="center">

Hình 14: Phân tích lỗi cho phương pháp tốt nhất - Histogram cho thấy phân bố lỗi tập trung quanh 0, scatter plot cho thấy mối quan hệ tuyến tính giữa predictions và actual values.

</p>

Quan sát Hình 14 cho thấy phân tích lỗi chi tiết cho phương pháp tốt nhất (NLinear_PostProcessing_Lasso). Hình ảnh bao gồm histogram phân bố lỗi và scatter plot mối quan hệ giữa predictions và actual values. Histogram cho thấy phân bố lỗi tập trung quanh 0, với đỉnh phân bố gần giá trị 0, cho thấy đa số predictions có sai số nhỏ. Phân bố này gần với phân bố chuẩn, cho thấy mô hình không có bias hệ thống lớn và các lỗi là ngẫu nhiên hơn là có hệ thống. Tuy nhiên, vẫn có một số outliers với lỗi lớn, có thể do các biến động đột ngột trong dữ liệu mà mô hình khó dự đoán. Scatter plot cho thấy mối quan hệ giữa predictions và actual values. Các điểm dữ liệu có xu hướng tập trung quanh đường y=x, cho thấy có mối quan hệ tuyến tính giữa predictions và actual values. Điều này cho thấy mô hình đã học được một phần pattern đúng, nhưng vẫn cần cải thiện thêm. Phân tích lỗi này cung cấp insights quan trọng về điểm mạnh và điểm yếu của phương pháp, giúp xác định các hướng cải thiện trong tương lai. Các lỗi lớn thường xuất hiện trong các giai đoạn có biến động đột ngột, cho thấy cần có cơ chế đặc biệt để xử lý các sự kiện bất thường.

#### 5.2.5. Kết luận từ Bước 2

Sau khi phân tích kỹ lưỡng, nghiên cứu đưa ra quyết định quan trọng. NLinear với Post-processing Lasso cho kết quả tốt nhất với MSE 98.5245. Tuy nhiên, PatchTST cũng cho kết quả rất tốt với MSE 172.5861 và có tiềm năng cải thiện thêm. PatchTST với Optuna được thử nghiệm để tối ưu hyperparameters và kết hợp với post-processing.

PatchTST được chọn vì các lý do sau. PatchTST là mô hình Transformer hiện đại, có khả năng học pattern phức tạp. Kết quả baseline đã tốt với MSE 172.5861. Có thể cải thiện thêm bằng tối ưu hyperparameters. Kết hợp với post-processing có thể đạt kết quả tốt hơn NLinear. Do đó, PatchTST được tiếp tục phát triển với Optuna.

### Bước 3: PatchTST với Optuna và Validation Analysis

**PatchTST (Patch-based Time Series Transformer)** là mô hình Transformer chuyên biệt cho bài toán dự đoán chuỗi thời gian, được phát triển dựa trên kiến trúc Vision Transformer (ViT). Khác với các mô hình Transformer truyền thống xử lý từng điểm dữ liệu riêng lẻ, PatchTST chia chuỗi thời gian thành các **patch** (đoạn liên tiếp) có độ dài cố định, tương tự như cách ViT chia hình ảnh thành các patch. Mỗi patch được embedding thành vector đặc trưng, sau đó được đưa vào các lớp Transformer encoder để học các mối quan hệ phức tạp giữa các patch.

Kiến trúc PatchTST có hai đặc điểm quan trọng. Thứ nhất, **patch-based approach** cho phép mô hình học được cả **local patterns** (xu hướng ngắn hạn trong từng patch) và **global patterns** (mối quan hệ dài hạn giữa các patch). Thứ hai, **Reversible Instance Normalization (RevIN)** được tích hợp để xử lý vấn đề non-stationarity trong dữ liệu tài chính. RevIN normalize dữ liệu trước khi đưa vào Transformer và reverse normalization sau khi dự đoán, giúp mô hình học được pattern bất kể mức độ giá tuyệt đối.

PatchTST đặc biệt phù hợp với dữ liệu tài chính vì các lý do sau. Dữ liệu giá cổ phiếu có nhiều **pattern phức tạp** (xu hướng dài hạn, chu kỳ, biến động ngắn hạn, volatility clustering), và kiến trúc Transformer với self-attention mechanism có khả năng nắm bắt các mối quan hệ phức tạp này tốt hơn các mô hình linear. Patch-based approach phù hợp với đặc tính của dữ liệu tài chính, nơi các xu hướng thường xuất hiện theo từng giai đoạn (ví dụ: giai đoạn tăng giá, giai đoạn điều chỉnh, giai đoạn ổn định). RevIN giải quyết vấn đề **distribution shift** trong dữ liệu tài chính, khi giá cổ phiếu có thể tăng hoặc giảm đáng kể theo thời gian, làm thay đổi phân phối dữ liệu. Cuối cùng, khả năng học **long-range dependencies** của Transformer giúp mô hình nắm bắt được các mối quan hệ xa trong quá khứ, quan trọng cho dự đoán dài hạn (100 ngày).

![PatchTST Pipeline Architecture](https://i.ibb.co/j94bX0SC/753fcd42fa6f.png)

<p align="center">

Hình 15: Kiến trúc Pipeline PatchTST - Flowchart chi tiết hiển thị toàn bộ quy trình từ dữ liệu Time Series đầu vào qua Data Preparation, Optuna Tuning, PatchTST Model (với kiến trúc bên trong: RevIN, Patch, Embedding, Transformer Encoder), Post-processing (Linear Regression), Smooth Correction, đến Forecast cuối cùng. Phần dưới mở rộng chi tiết kiến trúc bên trong PatchTST Model.

</p>

Quan sát Hình 15 cho thấy kiến trúc pipeline hoàn chỉnh của phương pháp PatchTST với bias correction. **Luồng chính (hàng trên)** bắt đầu từ Time Series Input, trải qua các bước: Data Preparation (chuẩn bị dữ liệu), Optuna Tuning (tối ưu hyperparameters), PatchTST Model với Best Parameters (mô hình chính), Post-processing với Linear Regression (học bias từ TimeSeriesSplit 3 folds, áp dụng công thức $y_{\text{corrected}} = \text{coef} \times y_{\text{predicted}} + \text{intercept}$, trong đó $\text{coef}$ là hệ số điều chỉnh và $\text{intercept}$ là hệ số chặn, được học từ dữ liệu validation để điều chỉnh predictions của baseline về gần với giá trị thực tế hơn), Smooth Correction (kết hợp baseline và post-processed predictions với weights chuyển từ 0→1 trong 20% đầu), và cuối cùng là Forecast Output. **Kiến trúc bên trong PatchTST Model (hàng dưới)** mở rộng chi tiết mô hình: RevIN (Reversible Instance Normalization để xử lý non-stationarity), Patch (chia chuỗi thời gian thành các patch), Embedding (chuyển đổi patch thành vector đặc trưng), Transformer Encoder (Attention + FFN để học mối quan hệ phức tạp), và Output với horizon h=100 (dự đoán 100 ngày). Baseline Predictions từ PatchTST được đưa vào Smooth Correction cùng với Post-processed predictions để tạo ra kết quả cuối cùng. Cấu trúc này phản ánh cách tiếp cận toàn diện, kết hợp mô hình Transformer hiện đại với các kỹ thuật bias correction để đạt được độ chính xác cao nhất.

Sau khi xác định PatchTST là mô hình có tiềm năng, hyperparameters được tối ưu bằng Optuna. Kết quả validation cũng được phân tích kỹ lưỡng. Đây là bước quan trọng để đạt được kết quả tốt nhất.

#### 5.3.1. Tối ưu Hyperparameters với Optuna

**Mục tiêu:**

Mục tiêu là tìm bộ hyperparameters tối ưu cho PatchTST để giảm MSE trên validation set.

**Quy trình:**

1. **Định nghĩa search space:**
   - `input_size`: 100-300 (step=50)
   - `patch_len`: 8-32 (step=8)
   - `stride`: 4-16 (step=4)
   - `learning_rate`: 1e-4 đến 1e-2 (log scale)
   - `max_steps`: 50-300 (step=50)

2. **Objective function:**
```python
def objective_patchtst(trial: Trial):
    """Objective function cho PatchTST optimization"""
    # Suggest hyperparameters
    input_size = trial.suggest_int("input_size", 100, 300, step=50)
    patch_len = trial.suggest_int("patch_len", 8, 32, step=8)
    stride = trial.suggest_int("stride", 4, 16, step=4)
    learning_rate = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    max_steps = trial.suggest_int("max_steps", 50, 300, step=50)
    
    try:
        model = PatchTST(
            h=min(horizon, len(val_nf_optuna)),
            input_size=input_size,
            patch_len=patch_len,
            stride=stride,
            revin=True,
            learning_rate=learning_rate,
            max_steps=max_steps,
            val_check_steps=10,
        )
        
        nf_model = NeuralForecast(models=[model], freq='D')
        nf_model.fit(df=train_nf_optuna, val_size=0)
        forecast = nf_model.predict()
        
        # Lấy predictions
        pred_col = [col for col in forecast.columns if col not in ['unique_id', 'ds']][0]
        pred = forecast[pred_col].values
        
        # Chỉ lấy số điểm tương ứng với validation
        n_points = min(len(pred), len(val_close_optuna), horizon)
        pred = pred[:n_points]
        val_true = val_close_optuna[:n_points]
        
        # Tính MSE
        mse = mean_squared_error(val_true, pred)
        return mse
    except Exception as e:
        print(f"   ⚠️  Lỗi trong trial: {e}")
        return float('inf')
```

**Giải thích chi tiết code:**

- **`def objective_patchtst(trial: Trial)`**: Hàm objective được Optuna gọi trong mỗi trial. Tham số `trial` là đối tượng Trial của Optuna, cung cấp các phương thức `suggest_*` để đề xuất giá trị hyperparameters.

- **`trial.suggest_int()` và `trial.suggest_float()`**: Optuna tự động đề xuất giá trị hyperparameters trong khoảng đã định nghĩa. Với `log=True`, learning rate được đề xuất trên thang log (1e-4 đến 1e-2), phù hợp với việc tối ưu learning rate thường cần thang log.

- **`PatchTST(...)`**: Tạo mô hình PatchTST với các hyperparameters được đề xuất. Tham số `h` là prediction horizon (số ngày dự đoán), được giới hạn bởi độ dài validation data. `revin=True` bật Reversible Instance Normalization để xử lý non-stationarity. `val_check_steps=10` kiểm tra validation mỗi 10 bước training để theo dõi quá trình học.

- **`NeuralForecast(models=[model], freq='D')`**: NeuralForecast là wrapper framework để train và predict các mô hình time series. `freq='D'` chỉ định tần suất dữ liệu là daily (hàng ngày). Mô hình PatchTST được đưa vào dưới dạng list `[model]` để có thể kết hợp nhiều mô hình nếu cần.

- **`nf_model.fit(df=train_nf_optuna, val_size=0)`**: Train mô hình trên training data. `val_size=0` nghĩa là không chia validation từ training data (vì đã có validation set riêng `val_close_optuna`).

- **`forecast = nf_model.predict()`**: Dự đoán trên validation data. Kết quả trả về là DataFrame với các cột `unique_id`, `ds` (date), và cột predictions (tên động, ví dụ `PatchTST`).

- **`pred_col = [col for col in forecast.columns if col not in ['unique_id', 'ds']][0]`**: Tìm tên cột chứa predictions bằng cách loại bỏ các cột metadata (`unique_id`, `ds`), sau đó lấy cột đầu tiên còn lại.

- **`pred = forecast[pred_col].values`**: Lấy giá trị predictions dưới dạng numpy array.

- **`n_points = min(len(pred), len(val_close_optuna), horizon)`**: Đảm bảo predictions và ground truth có cùng độ dài, giới hạn bởi độ dài ngắn nhất trong ba giá trị: predictions, validation data, và horizon.

- **`mse = mean_squared_error(val_true, pred)`**: Tính Mean Squared Error giữa predictions và ground truth. MSE càng nhỏ càng tốt, nên Optuna sẽ minimize giá trị này.

- **`except Exception as e`**: Xử lý lỗi nếu có vấn đề trong quá trình training (ví dụ: out of memory, NaN values). Trả về `float("inf")` để Optuna bỏ qua trial này và tiếp tục với trial khác.

3. **Chạy Optuna:**
   - Số trials: 20
   - Direction: minimize MSE
   - Study name: PatchTST_Optuna

**Kết quả tối ưu:**

Sau 20 trials, Optuna tìm được bộ hyperparameters tốt nhất:

<table>
<thead>
<tr>
<th style="text-align: center">Hyperparameter</th>
<th style="text-align: center">Giá trị tối ưu</th>
</tr>
</thead>
<tbody>
<tr>
<td>`input_size`</td>
<td>100</td>
</tr>
<tr>
<td>`patch_len`</td>
<td>32</td>
</tr>
<tr>
<td>`stride`</td>
<td>4</td>
</tr>
<tr>
<td>`learning_rate`</td>
<td>0.001610814898983045</td>
</tr>
<tr>
<td>`max_steps`</td>
<td>250</td>
</tr>
<tr>
<td><strong>Best MSE (Optuna)</strong></td>
<td><strong>191.8113</strong></td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 6: Hyperparameters tối ưu từ Optuna - Bộ hyperparameters tốt nhất được tìm thấy sau 20 trials của Optuna, bao gồm input_size, patch_len, stride, learning_rate, max_steps và best MSE.

</p>

**Nhận xét:**

Phân tích kết quả tối ưu cho thấy:

- MSE giảm từ ~800 xuống ~192 (cải thiện 76%)
- `input_size=100` phù hợp với dữ liệu FPT
- `patch_len=32` và `stride=4` tạo ra patches có kích thước hợp lý
- `learning_rate` được tối ưu ở mức 0.0016

Kết quả này cho thấy Optuna đã tìm được bộ hyperparameters tốt. Do đó, các tham số này được sử dụng cho các thử nghiệm tiếp theo.

#### 5.3.2. Train PatchTST Baseline với Best Parameters

Sau khi có best parameters, PatchTST được train lại trên toàn bộ dữ liệu training (train + val) với các tham số đã tối ưu.

**Kết quả Baseline:**

PatchTST Baseline đạt MSE 641.4994, một giá trị rất cao cho thấy dự đoán không chính xác. Model có **bias hệ thống lớn** - dự đoán cao hơn thực tế. Kết quả này xác nhận nhận định ban đầu về bias hệ thống và cho thấy cần áp dụng post-processing để điều chỉnh.

Kết quả này xác nhận nhận định ban đầu về bias hệ thống. Do đó, post-processing được áp dụng.

#### 5.3.3. Post-processing Regression

Post-processing là bước điều chỉnh predictions sau khi model chính (PatchTST) đã dự đoán, nhằm giảm sai số và bias. Vấn đề cần giải quyết là model có bias hệ thống. PatchTST baseline có thể có bias hệ thống, ví dụ dự đoán cao hơn thực tế. Ví dụ, baseline dự đoán 120 trong khi thực tế là 100, dẫn đến bias +20. Ngoài ra, predictions chưa tối ưu vì model có thể có pattern sai số nhất quán, ví dụ luôn dự đoán cao hơn 15% so với thực tế.

**Quy trình:**

**Bước 1: Thu thập dữ liệu từ TimeSeriesSplit**

Quy trình sử dụng TimeSeriesSplit với 3 folds để tạo validation folds. TimeSeriesSplit chia dữ liệu tuần tự theo thời gian, không random, đảm bảo train luôn là quá khứ và validation luôn là tương lai. Với dữ liệu 1033 điểm, TimeSeriesSplit chia như sau: Fold 1 có train từ điểm 1-344 và validation từ điểm 345-688, Fold 2 có train từ điểm 1-688 (gộp Fold 1 train+val) và validation từ điểm 689-860, Fold 3 có train từ điểm 1-860 (gộp tất cả trước đó) và validation từ điểm 861-1033.

Với mỗi fold, quy trình thực hiện như sau. PatchTST được train trên train_fold, chỉ học từ dữ liệu quá khứ. Sau đó, model predict trên val_fold, đây là dữ liệu chưa từng thấy trong quá trình train. Các cặp (prediction, ground_truth) được thu thập từ validation fold. Tổng cộng thu được khoảng 300 điểm từ 3 folds.

**Tại sao các fold không bằng nhau?**

TimeSeriesSplit sử dụng thuật toán Expanding Window (cửa sổ mở rộng), không phải chia đều. Đây là đặc điểm thiết kế, không phải lỗi. Với `n_splits=3` và `n_samples=1033`, test_size được tính bằng `n_samples / (n_splits + 1) ≈ 258 điểm`. Cách chia như sau: Fold 1 có training size khoảng 258 điểm và validation size khoảng 258 điểm, Fold 2 có training size khoảng 516 điểm (mở rộng) và validation size khoảng 258 điểm, Fold 3 có training size khoảng 774 điểm (mở rộng tiếp) và validation size khoảng 258 điểm. Kết quả thực tế với 1033 điểm là Fold 1 có train ~344 và val ~344, Fold 2 có train ~688 và val ~172, Fold 3 có train ~860 và val ~173.

Lý do không chia đều là để mô phỏng thực tế. Trong thực tế, khi dự đoán tương lai, càng về sau càng có nhiều dữ liệu lịch sử. TimeSeriesSplit mô phỏng điều này bằng cách để training set mở rộng dần, fold sau có nhiều dữ liệu training hơn fold trước. Điều này giúp model học tốt hơn với nhiều dữ liệu hơn và phản ánh thực tế là càng về sau càng có nhiều dữ liệu lịch sử. Validation luôn là tương lai, đảm bảo không có data leakage và mỗi fold validation là "tương lai" so với training của fold đó.

**Bước 2: Train Linear Regression**

Linear Regression được train để học mapping từ predictions sang actual values. Model học được công thức điều chỉnh dạng $y = \text{coef} \times \text{pred} + \text{intercept}$. Trong trường hợp này, công thức học được là $y_{\text{corrected}} = 0.7267 \times y_{\text{predicted}} + 9.3249$.

**Giải thích công thức:**

Hệ số `coef` (0.7267) có ý nghĩa là PatchTST có xu hướng dự đoán cao hơn thực tế. Hệ số này giảm predictions xuống khoảng 72.67% giá trị gốc. Ví dụ, prediction 120 được giảm xuống 120 × 0.7267 = 87.2.

Hệ số `intercept` (9.3249) có ý nghĩa là bù thêm một lượng cố định. Hệ số này điều chỉnh offset sau khi nhân với coef. Ví dụ, 87.2 + 9.3249 = 96.53.

**Ví dụ minh họa:**

Trước post-processing, baseline prediction là 120 trong khi ground truth là 100, dẫn đến error +20 (dự đoán cao hơn). Sau post-processing, prediction được điều chỉnh thành 0.7267 × 120 + 9.3249 = 96.53. So với ground truth 100, error chỉ còn -3.47, tốt hơn nhiều so với +20 ban đầu.

**Tại sao phát hiện được độ lệch?**

Validation folds là "unseen data" đối với model. Model chỉ train trên train_fold và predict trên val_fold (chưa thấy trong quá trình train). TimeSeriesSplit đảm bảo validation là "tương lai" so với training. Model có bias khi predict trên data mới do distribution shift theo thời gian, model chưa tối ưu hoàn toàn, và pattern khác nhau giữa train và val. Bias có pattern nhất quán, có thể là systematic bias (cố định) hoặc proportional bias (tỷ lệ). Post-processing học được pattern này từ nhiều folds, với 3 folds thu được khoảng 300 điểm, đủ để học pattern bias.

**Lưu ý quan trọng:**

Post-processing học cơ chế bias chỉ từ tập train+val, không dùng test data. Dữ liệu học được lấy từ `train_nf_full` (train + val gộp lại, 1033 điểm), chia bằng TimeSeriesSplit thành 3 folds, và thu thập độ lệch từ validation folds. Test data không được dùng để train, chỉ dùng để đánh giá cuối cùng. Giả định là bias trên validation folds tương tự trên test data. Nếu bias khác nhiều, hiệu quả có thể giảm.

**Không có normalize trong post-processing:**

Dữ liệu được dùng trực tiếp (raw values), không có StandardScaler, RobustScaler, hay MinMaxScaler. Lý do là predictions từ PatchTST và ground truth đã cùng scale (khoảng 80-120, giá cổ phiếu), Linear Regression không yêu cầu normalize, và đơn giản hóa pipeline, tránh lỗi transform/inverse transform.

**Kết quả Post-processing:**

Sau khi áp dụng Post-processing, MSE giảm từ 641.50 xuống 48.62, cải thiện **92.42%** so với baseline. Kết quả này cho thấy post-processing cải thiện đáng kể, tuy nhiên vẫn còn cơ hội cải thiện thêm.

Kết quả này cho thấy post-processing rất hiệu quả. Tuy nhiên, vẫn có thể cải thiện thêm bằng cách kết hợp với smooth transition.

#### 5.3.4. Validation Analysis

Từ notebook `patchtst_validation_analysis.ipynb`, kết quả validation được phân tích kỹ lưỡng:

Phân tích kỹ lưỡng kết quả validation cho thấy các điểm quan trọng. Điểm đầu tiên là mô hình đang rất lạc quan. Predictions phản ánh đúng tính chất của dữ liệu cổ phiếu là đang ở trạng thái Uptrend. Model học được pattern tăng giá từ dữ liệu training.

Điểm thứ hai là sau khi áp dụng Post-Processing Bias, dữ liệu được hiệu chỉnh độ lệch lại. Dữ liệu đã phản ánh đúng biến động thị trường. Giá trị phản ánh được có thể được điều chỉnh tương ứng với biến động tiêu cực trên thị trường.

Điểm thứ ba là vấn đề với Post-processing trực tiếp. Post-processing thay đổi giá trị đầu tiên, giảm độ tin cậy của predictions. Do đó, cần một phương pháp smooth hơn. Vì vậy, phương pháp Smooth Bias Correction được phát triển.

![PatchTST Validation Analysis - 3-Fold Cross-Validation](https://i.ibb.co/7JcC4MqM/6ecb8fefab34.png)

<p align="center">

Hình 16: Phân tích validation 3-fold cross-validation - Actual vs Predicted và Residuals Plot cho từng fold.

</p>

Quan sát Hình 16 cho thấy phân tích validation từ 3-fold cross-validation với 6 subplots (3 hàng × 2 cột), phản ánh sự cải thiện và thay đổi hiệu suất của mô hình qua từng fold.

**Fold 1** (hàng trên, n=258 điểm): Predictions (đường xanh nét đứt) có volatility cao, overshoot đỉnh quanh time step 150 (vượt 80 trong khi actual ~58-59), MSE 257.31. Residuals plot cho thấy pattern giảm dần: under-prediction ở giá trị thấp (<55), over-prediction ở giá trị cao (>55), và heteroscedasticity rõ rệt (spread tăng theo predicted values). Đây là fold đầu tiên, mô hình chưa được điều chỉnh tốt, thể hiện qua MSE cao.

**Fold 2** (hàng giữa, n=258 điểm): Đây là fold có hiệu suất tốt nhất trong 3 folds. Predictions (đường xanh lá nét đứt) có pattern dao động mạnh nhưng MSE 175.60 (thấp nhất), cho thấy mô hình đã học được pattern tốt hơn từ fold 1. Tuy nhiên, residuals plot vẫn có dạng quạt (fan-shaped), spread tăng mạnh khi predicted values tăng, phản ánh heteroscedasticity. Sự cải thiện từ Fold 1 sang Fold 2 rõ rệt: MSE giảm 31.7% (từ 257.31 xuống 175.60), cho thấy mô hình đã điều chỉnh được bias hệ thống.

**Fold 3** (hàng dưới, n=258 điểm): Predictions (đường cam nét đứt) mượt hơn nhưng consistently thấp hơn actual, đặc biệt trong giai đoạn tăng mạnh từ time step 50-200, MSE 270.22 (cao nhất). Residuals plot cho thấy phần lớn residuals âm (over-prediction) ở predicted values 70-105, sau đó chuyển sang dương (under-prediction) ở 105-110. So với Fold 2, Fold 3 có hiệu suất kém hơn: MSE tăng 53.9% (từ 175.60 lên 270.22). Điều này cho thấy mô hình gặp khó khăn với dữ liệu ở phần cuối của training set, có thể do dữ liệu này có pattern khác biệt hoặc mô hình chưa học được pattern tăng giá mạnh.

**Tổng hợp và phân tích xu hướng:** Tổng hợp từ 3 folds: MSE trung bình 234.38. Xu hướng qua các folds cho thấy: (1) Fold 2 đạt hiệu suất tốt nhất, chứng tỏ mô hình có thể học tốt trên một phần dữ liệu nhất định; (2) Sự thay đổi hiệu suất giữa các folds (MSE từ 175.60 đến 270.22, chênh lệch 54%) phản ánh tính không ổn định của mô hình và sự khác biệt về pattern giữa các phần dữ liệu; (3) Heteroscedasticity xuất hiện ở cả 3 folds, cho thấy đây là vấn đề cố hữu của mô hình cần được xử lý. Phân tích này khẳng định tầm quan trọng của post-processing để điều chỉnh bias và cải thiện hiệu suất tổng thể.

![Training Data with 3-Fold Validation Predictions](https://i.ibb.co/j9wDgTVB/9b49cf956ebe.png)

<p align="center">

Hình 17: Biểu đồ tổng thể - Training data và 3-fold validation predictions - Hiển thị toàn bộ training data (1033 điểm) với 3 validation folds predictions tại đúng vị trí của chúng trong chuỗi thời gian.

</p>

Quan sát Hình 17 cho thấy biểu đồ tổng thể hiển thị toàn bộ training data (đường đen) từ time step 1 đến 1033, với giá bắt đầu từ ~20 k VND và tăng dần lên ~110-120 k VND, phản ánh xu hướng tăng (uptrend) mạnh của cổ phiếu FPT trong thực tế. Ba validation folds được đặt tại đúng vị trí của chúng trong chuỗi thời gian. **Fold 1** (khoảng time step 260-517, 258 điểm): Predictions (đường xanh nét đứt) theo sát xu hướng training data nhưng có volatility cao, đặc biệt overshoot đỉnh quanh time step 400 (vượt 80 k VND trong khi actual ~60 k VND). Actual validation data (điểm tròn xanh) khớp với training data, xác nhận vị trí đúng. **Fold 2** (khoảng time step 518-775, 258 điểm): Predictions (đường xanh lá nét đứt) có pattern dao động mạnh, thường xuyên overshoot và undershoot so với actual (điểm tròn xanh lá), phản ánh hiệu suất không ổn định trong fold này. **Fold 3** (khoảng time step 776-1033, 258 điểm): Predictions (đường cam nét đứt) mượt hơn nhưng consistently thấp hơn actual (điểm tròn cam), đặc biệt ở phần cuối khi giá tăng mạnh lên 110-120 k VND, predictions chỉ đạt ~100-110 k VND. 

Điểm quan trọng từ biểu đồ này là các dự đoán có xu hướng lạc quan (optimistic), phản ánh đúng tính chất của dữ liệu cổ phiếu FPT đang ở trạng thái uptrend. Mô hình học được pattern tăng giá từ dữ liệu training và áp dụng xu hướng này vào predictions, dẫn đến các dự đoán thường cao hơn hoặc theo sát xu hướng tăng. Điều này đặc biệt rõ ràng ở Fold 1 và Fold 2, nơi predictions có xu hướng overshoot hoặc dao động mạnh theo hướng tăng. Tuy nhiên, ở Fold 3, mặc dù mô hình vẫn giữ xu hướng tăng nhưng không bắt kịp tốc độ tăng mạnh của giá thực tế, cho thấy mô hình gặp khó khăn khi dự đoán các biến động lớn ở phần cuối của chuỗi. Biểu đồ này cho phép so sánh trực quan hiệu suất mô hình trên các phần dữ liệu khác nhau trong training set, cho thấy mô hình có xu hướng học tốt hơn ở phần giữa (Fold 2) và gặp khó khăn ở phần cuối (Fold 3) khi giá tăng mạnh. Các đường phân cách (đường đứt nét xám) giữa các folds giúp phân biệt rõ ràng từng validation period.

![Forecast Comparison - All Methods](https://i.ibb.co/zT8tG1JT/ce0e821bf84d.png)

<p align="center">

Hình 18: So sánh dự đoán tất cả phương pháp (từ notebook patchtst_validation_analysis.ipynb) - Biểu đồ trên so sánh Baseline (MSE 641.50), Post-processing (MSE 44.94), và Smooth 20% (MSE 17.49) với ground truth trên toàn bộ 100 ngày, biểu đồ dưới zoom vào 30 ngày đầu để xem chi tiết.

</p>

💡 **Lưu ý:** Kết quả trong Hình 17 đến từ notebook `patchtst_validation_analysis.ipynb`, có thể khác một chút so với kết quả cuối cùng trong `patchtst_fixed_params.ipynb` (MSE 48.62 và 15.26) do cách chia dữ liệu và validation folds khác nhau. Kết quả cuối cùng chính thức sử dụng `patchtst_fixed_params.ipynb`.

Quan sát Hình 18 cho thấy so sánh trực quan ba phương pháp dự đoán với ground truth qua hai biểu đồ. **Biểu đồ trên (toàn bộ 100 ngày)**: Ground Truth (đường đen) bắt đầu từ ~118-120 k VND, giảm mạnh xuống ~90-92 k VND ở ngày 20, sau đó dao động và tăng dần lên 110-118 k VND ở cuối chuỗi. **Baseline** (đường đỏ nét đứt, MSE 641.50) có bias lạc quan rõ rệt, consistently overestimate: bắt đầu ~118 k VND, tăng lên 125-130 k VND ở ngày 20 (trong khi actual giảm), đạt đỉnh 135-138 k VND ở ngày 50-60. Bias lạc quan này phản ánh xu hướng uptrend của FPT trong training data, khiến mô hình dự đoán giá cao hơn thực tế. **Post-processing** (đường cam nét đứt, MSE 44.94) đã giảm đáng kể bias lạc quan: bắt đầu ~96 k VND, tăng nhẹ lên ~100 k VND, ổn định quanh 98-100 k VND, phản ánh đúng tình trạng thực tế hơn so với baseline. **Best Method - Smooth 20%** (đường xanh lá nét liền, MSE 17.49) theo sát ground truth nhất: bắt đầu ~118 k VND, giảm cùng xu hướng xuống ~100 k VND ở ngày 20, sau đó dao động trong khoảng 100-115 k VND. **Biểu đồ dưới (zoom 30 ngày đầu)**: Cho thấy chi tiết giai đoạn giảm mạnh. Baseline vẫn giữ mức cao 118-122 k VND, không phản ứng với sự giảm giá. Post-processing và Smooth 20% đều phản ứng tốt hơn với biến động thực tế.

**Giải thích tại sao kết quả này đúng:**

Kết quả này phản ánh đúng cơ chế hoạt động của từng phương pháp và cách post-processing giải quyết vấn đề bias lạc quan. **Baseline có bias lạc quan** vì mô hình PatchTST học từ training data có xu hướng uptrend mạnh (từ ~20 k VND lên ~110-120 k VND trong 4.5 năm). Khi dự đoán tương lai, mô hình áp dụng pattern tăng giá đã học, dẫn đến overestimate, đặc biệt khi test data có biến động khác (giảm mạnh ở đầu). Điều này phù hợp với lý thuyết time series forecasting: mô hình có xu hướng extrapolate pattern từ training data, và khi có distribution shift, bias xuất hiện.

**Post-processing đã giải quyết được vấn đề giảm sự lạc quan** của việc học dữ liệu của mô hình. Post-processing học được công thức điều chỉnh $y = 0.7267 \times \text{pred} + 9.3249$ từ validation folds, giảm predictions xuống để bù bias lạc quan. Công thức này phản ánh đúng tình trạng trong thực tế: hệ số 0.7267 giảm predictions xuống ~73% giá trị gốc, bù lại bias overestimate của baseline. Kết quả MSE giảm từ 641.50 xuống 48.62 (cải thiện 92.42%) trong notebook `patchtst_fixed_params.ipynb` chứng tỏ post-processing đã điều chỉnh được bias lạc quan, giúp predictions phản ánh đúng tình trạng thực tế hơn. Tuy nhiên, post-processing vẫn có hạn chế là thay đổi giá trị đầu tiên ngay lập tức, không giữ tính liên tục.

**Smooth 20% đạt kết quả tốt nhất** vì kết hợp ưu điểm của cả hai phương pháp: giữ nguyên giá trị đầu tiên (tăng độ tin cậy), smooth transition tránh thay đổi đột ngột, và phần cuối sử dụng post-processing để điều chỉnh bias. Phương pháp này vừa giải quyết được bias lạc quan, vừa phản ánh đúng tình trạng thực tế với độ chính xác cao (MSE 15.26 trong notebook `patchtst_fixed_params.ipynb`).

![Complete Overview - Training and Test Data](https://i.ibb.co/k6wQst3Z/9b2633ab8e5f.png)

<p align="center">

Hình 19: Biểu đồ tổng thể - Training data, Test data và tất cả predictions - Hiển thị toàn bộ training data (919 điểm), validation data (114 điểm), test data (100 điểm), 3-fold validation predictions, và predictions của 3 phương pháp (Baseline, Post-processing, Smooth 20%) trên test period.

</p>

Quan sát Hình 19 cho thấy biểu đồ tổng thể hiển thị toàn bộ dữ liệu từ training đến test (từ notebook `patchtst_validation_analysis.ipynb`). **Training data** (đường đen) từ ~20 k VND tăng lên ~110-120 k VND, phản ánh xu hướng tăng mạnh. **3-fold validation predictions** (đường xanh, xanh lá, cam nét đứt) cho thấy hiệu suất khác nhau trên các phần dữ liệu. **Test data** (đường tím với markers) bắt đầu từ ~110 k VND, giảm mạnh xuống ~90-95 k VND, sau đó phục hồi và dao động quanh 110-115 k VND. **Baseline prediction** (đường đỏ nét đứt, MSE 641.50) có bias lạc quan rõ rệt, consistently overestimate, tăng lên 130-140 k VND. **Post-processing prediction** (đường xanh lá nét liền, MSE 44.94 trong notebook này) theo sát test data hơn nhưng vẫn underestimate một phần. **Best Method - Smooth 20%** (đường tím nét liền, MSE 17.49 trong notebook này) theo sát ground truth nhất, phản ánh đúng các biến động của test data. Đường phân cách validation/test boundary (đường đứt nét xanh dọc) và vùng test period (highlight màu vàng) giúp phân biệt rõ ràng giữa training/validation và test period. 💡 **Lưu ý:** Kết quả cuối cùng chính thức từ `patchtst_fixed_params.ipynb` có MSE 48.62 (Post-processing) và 15.26 (Smooth 20%).

![Residuals Analysis - Best Method](https://i.ibb.co/3mzDz2Qp/a941c8a7c7a2.png)

<p align="center">

Hình 20: Phân tích Residuals - Best Method (Smooth 20%) - Biểu đồ trên là time series residuals theo thời gian trong test period, biểu đồ dưới là scatter plot residuals theo predicted values, cả hai đều có đường Zero Error (đường đỏ nét đứt) ở Y=0.

</p>

Quan sát Hình 20 cho thấy phân tích residuals của Best Method (Smooth 20%) trên test period qua hai biểu đồ. **Biểu đồ trên (Residuals theo Thời Gian)**: Time series residuals trong test period (time step 1035-1125) cho thấy pattern rõ ràng với các giai đoạn under-prediction (residuals dương, đạt đỉnh ~9.0-9.5 quanh time step 1075) và over-prediction (residuals âm, đạt -7.5 quanh time step 1040, 1090, và 1125). Pattern này cho thấy mô hình vẫn chưa nắm bắt hoàn toàn các temporal dependencies trong test data, đặc biệt ở các giai đoạn biến động mạnh. **Biểu đồ dưới (Residuals Plot)**: Scatter plot residuals theo predicted values (97.5-115.0 k VND) cho thấy residuals phân tán quanh đường Zero Error, không có bias hệ thống. Phần lớn residuals nằm trong khoảng -5.0 đến 5.0, nhưng có một số outliers đạt gần 10.0 và -10.0. Spread của residuals tương đối nhất quán, không có dấu hiệu mạnh của heteroscedasticity. Mặc dù vẫn có lỗi dự đoán lớn ở một số điểm, so với baseline, Smooth 20% đã cải thiện đáng kể, chứng tỏ phương pháp này đã giải quyết được phần lớn vấn đề bias lạc quan.

#### 5.3.5. Smooth Bias Correction - Phương pháp tốt nhất

Nhận thấy vấn đề với post-processing trực tiếp, phương pháp **Smooth Bias Correction** được phát triển. Phương pháp này kết hợp smooth transition và post-processing để đạt kết quả tốt nhất.

Cơ chế hoạt động như sau. Phần đầu (20%) sử dụng smooth transition. Giá trị đầu tiên được giữ nguyên với `pred_smooth[0] = pred_baseline[0]` (weight[0] = 0). Weight tăng dần từ 0 đến 1 trong phần đầu. Interpolation được thực hiện bằng công thức $\text{pred} = (1-\text{weight}) \times \text{baseline} + \text{weight} \times \text{post\_processing}$, trong đó $\text{weight}$ là trọng số tăng dần từ 0 đến 1, $\text{baseline}$ là predictions từ mô hình gốc, và $\text{post\_processing}$ là predictions đã được điều chỉnh bias. Công thức này thực hiện phép nội suy tuyến tính giữa hai nguồn predictions, đảm bảo chuyển tiếp mượt mà từ baseline sang post-processing, tránh thay đổi đột ngột ở giá trị đầu tiên.

Phần cuối (80%) sử dụng Post-processing Regression trực tiếp. Post-processing được dùng trực tiếp với `pred_smooth[i] = post_processing[i]` (weight[i] = 1.0), đảm bảo hiệu quả cao.

**Công thức:**

```python
# Tính weights cho smooth transition
n = len(pred_baseline)
weights = np.zeros(n)
smooth_ratio = 0.2  # 20% đầu smooth transition

# Tính điểm chia: smooth_ratio% đầu smooth, phần còn lại traditional
split_point = int(n * smooth_ratio)
if split_point < 1:
    split_point = 1
if split_point >= n - 1:
    split_point = max(1, n - 2)

# Phần đầu: smooth transition từ baseline đến post-processing
if split_point > 1:
    smooth_part = split_point
    smooth_weights = np.arange(smooth_part) / max(smooth_part - 1, 1)
    # Normalize để weight[split_point-1] = 1.0
    if smooth_weights[-1] > 0:
        smooth_weights = smooth_weights / smooth_weights[-1]
    weights[:split_point] = smooth_weights
    weights[0] = 0.0  # Giữ nguyên điểm đầu (weight=0 → dùng baseline)
    weights[split_point - 1] = 1.0  # Điểm cuối phần smooth (weight=1 → dùng post-processing)

# Phần cuối: post-processing regression (weight = 1.0)
weights[split_point:] = 1.0

# Đảm bảo: weight[0] = 0 và weight[-1] = 1
weights[0] = 0.0
weights[-1] = 1.0

# Interpolation: pred = (1-weight) * baseline + weight * post_processing
pred_corrected = (1 - weights) * pred_baseline + weights * pred_post
```

**Giải thích code:**
- `split_point = int(n * 0.2)`: Tính điểm chia giữa phần smooth transition (20% đầu) và phần post-processing trực tiếp (80% cuối). Ví dụ với 100 predictions, `split_point = 20`.
- `weights[:split_point] = linear_interpolation(0 → 1)`: Tạo weights cho 20% đầu tăng tuyến tính từ 0.0 đến 1.0. Ví dụ: `weights[0] = 0.0`, `weights[5] = 0.25`, `weights[10] = 0.5`, `weights[19] ≈ 0.95`.
- `weights[0] = 0.0`: Đảm bảo giá trị đầu tiên hoàn toàn dùng baseline (không có post-processing), giữ tính liên tục với giá trị cuối của training data.
- `weights[split_point:] = 1.0`: Phần còn lại (80% cuối) dùng hoàn toàn post-processing, đảm bảo hiệu quả điều chỉnh bias tối đa.
- `pred_smooth = (1 - weights) * baseline + weights * post_processing`: Công thức interpolation kết hợp baseline và post-processing theo trọng số weights, tạo ra predictions cuối cùng với chuyển tiếp mượt mà.

**Kết quả Smooth Linear 20%:**

Phương pháp Smooth 20% đạt MSE **15.26**, cải thiện **97.62%** so với baseline (641.50). So với Post-processing (MSE 48.62, cải thiện 92.42%), Smooth 20% cải thiện thêm 68.61% và là phương pháp tốt nhất. Phương pháp giữ nguyên giá trị đầu tiên, tăng độ tin cậy, kết hợp tốt nhất của smooth transition và post-processing, và cải thiện cả accuracy và reliability. Kết quả này cho thấy phương pháp Smooth Bias Correction là phương pháp tốt nhất. Do đó, phương pháp này được sử dụng cho submission cuối cùng.

### Bước 4: Fixed Parameters và kết quả cuối cùng

Sau khi đã tìm được phương pháp tốt nhất (Smooth Linear 20% + Post-processing), notebook `patchtst_fixed_params.ipynb` được tạo ra. Notebook này sử dụng các thông số Optuna đã được tối ưu mà không cần chạy Optuna lại. Điều này giúp tiết kiệm thời gian và đảm bảo kết quả ổn định.

#### 5.4.1. Fixed Parameters

**Thông số đã được cố định từ Optuna:**

<table>
<thead>
<tr>
<th style="text-align: center">Hyperparameter</th>
<th style="text-align: center">Giá trị</th>
</tr>
</thead>
<tbody>
<tr>
<td>`input_size`</td>
<td>100</td>
</tr>
<tr>
<td>`patch_len`</td>
<td>32</td>
</tr>
<tr>
<td>`stride`</td>
<td>4</td>
</tr>
<tr>
<td>`learning_rate`</td>
<td>0.001610814898983045</td>
</tr>
<tr>
<td>`max_steps`</td>
<td>250</td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 7: Fixed Parameters - Bộ thông số đã được cố định từ Optuna, bao gồm input_size, patch_len, stride, learning_rate, max_steps và best MSE.

</p>

Ưu điểm của Fixed Parameters bao gồm tiết kiệm thời gian vì không cần chạy Optuna (tiết kiệm 30-60 phút), kết quả ổn định vì luôn sử dụng cùng bộ thông số tối ưu, dễ sử dụng vì chỉ cần chạy từ đầu đến cuối và không cần tối ưu, và reproducible vì có thể reproduce kết quả một cách dễ dàng.

#### 5.4.2. Quy trình trong Fixed Parameters Notebook

Quy trình trong Fixed Parameters Notebook gồm sáu bước. Bước đầu tiên là load và chuẩn bị dữ liệu, bao gồm load `FPT_train.csv` (1149 điểm), chia train/validation với tỉ lệ 80% train và 10% validation, và chuẩn bị format cho NeuralForecast.

**Cách chia dữ liệu trong notebook:**

Dữ liệu được chia thành nhiều phần với mục đích khác nhau. Training data ban đầu chiếm 80% (919 điểm), validation data ban đầu chiếm 10% (114 điểm), và còn lại khoảng 10% (~116 điểm) không dùng cho train/val ban đầu. Full training (train+val) gộp lại thành 1033 điểm, được dùng để train final model. Optuna optimization chia 90/10 từ `train_nf_full`, với 929 điểm cho train và 104 điểm cho validation. Test data được lấy từ file test riêng, gồm 100 điểm làm ground truth. Post-processing sử dụng TimeSeriesSplit với 3 folds trên `train_nf_full` để thu thập dữ liệu từ validation folds, không dùng test data.

Bước thứ hai là train PatchTST Baseline với Fixed Parameters, sử dụng best parameters từ Optuna, train trên toàn bộ dữ liệu (train + val), và predict 100 ngày tiếp theo.

Bước thứ ba là train Post-processing Regression, sử dụng TimeSeriesSplit (3 folds) để thu thập dữ liệu, train Linear Regression với công thức $y = \text{coef} \times \text{pred} + \text{intercept}$, và học được công thức điều chỉnh. **Điểm khác biệt quan trọng**: Mặc dù cả hai notebook đều sử dụng TimeSeriesSplit với 3 folds trên `train_nf_full` (1033 điểm), validation folds trong `patchtst_fixed_params.ipynb` có thể phản ánh đúng hơn mức độ bias lạc quan của test data, dẫn đến post-processing model học được công thức điều chỉnh phù hợp hơn. Điều này giải thích tại sao `patchtst_fixed_params.ipynb` đạt MSE 15.26 tốt hơn so với 17.49 trong `patchtst_validation_analysis.ipynb`.

Bước thứ tư là áp dụng Smooth Bias Correction, với phần đầu (20%) sử dụng smooth transition, phần cuối (80%) sử dụng post-processing trực tiếp, và kết hợp để có kết quả tốt nhất.

Bước thứ năm là visualization, bao gồm vẽ biểu đồ so sánh các phương pháp với ground truth, phân tích residuals, và lưu biểu đồ với độ phân giải cao (300 DPI).

![Comparison of All Methods](https://i.ibb.co/tpGH6tpT/24bc58c9ec20.png)

<p align="center">

Hình 21: So sánh dự đoán tất cả phương pháp - Biểu đồ trên so sánh Baseline (MSE 641.50), Post-processing (MSE 48.62), và Smooth 20% (MSE 15.26) với ground truth trên toàn bộ 100 ngày, biểu đồ dưới zoom vào 30 ngày đầu để xem chi tiết giai đoạn giảm mạnh.

</p>

Quan sát Hình 21 cho thấy so sánh ba phương pháp dự đoán với ground truth. **Baseline** (MSE 641.50) có bias lạc quan rõ rệt, consistently overestimate do học từ xu hướng uptrend trong training data. **Post-processing** (MSE 48.62) đã giảm đáng kể bias lạc quan, học được công thức điều chỉnh để phản ánh đúng tình trạng thực tế. **Best Method - Smooth 20%** (MSE 15.26) theo sát ground truth nhất, phản ánh đúng các biến động với độ chính xác cao. Biểu đồ zoom 30 ngày đầu cho thấy Smooth 20% bắt đầu đúng và bắt kịp xu hướng tốt nhất. So sánh MSE: Baseline 641.50 → Post-processing 48.62 (giảm 92.42%) → Smooth 20% 15.26 (giảm thêm 68.61%), chứng tỏ Smooth 20% cân bằng tốt giữa theo dõi biến động và làm mượt predictions, giải quyết hiệu quả vấn đề bias lạc quan.

![Residual Analysis Smooth 20%](https://i.ibb.co/HT6KtCgp/835d20c51ef9.png)

<p align="center">

Hình 22: Phân tích Residuals Smooth 20% - Biểu đồ trái là scatter plot residuals theo predicted values, biểu đồ phải là time series residuals theo thời gian, cả hai đều có đường Zero Error (đường đỏ nét đứt) ở Y=0.

</p>

Quan sát Hình 22 cho thấy phân tích residuals của phương pháp Smooth 20% qua hai biểu đồ. **Biểu đồ trái (Residuals Plot)**: Các điểm residuals phân tán quanh đường Zero Error (Y=0), cho thấy mô hình không có bias hệ thống. Phần lớn residuals nằm trong khoảng -5.0 đến 5.0, chứng tỏ mô hình đã học được pattern đúng. **Biểu đồ phải (Residuals theo Thời Gian)**: Time series của residuals cho thấy pattern theo thời gian với các giai đoạn under-prediction (residuals dương) và over-prediction (residuals âm) xen kẽ, phản ánh mô hình chưa nắm bắt hoàn toàn các temporal dependencies ở các giai đoạn biến động mạnh. Tuy nhiên, so với baseline (bias 24.10), Smooth 20% đã cải thiện đáng kể với bias chỉ còn 0.91, chứng tỏ phương pháp này đã giải quyết được phần lớn vấn đề bias lạc quan.

**6. Xuất file submission:**
- Tạo file `submission_patchtst_fixed_params.csv`
- Format: `id, close` với 100 predictions

#### 5.4.3. Kết quả cuối cùng

**Kết quả trên validation set (100 ngày cuối) từ notebook `patchtst_fixed_params.ipynb`:**

<table>
<thead>
<tr>
<th style="text-align: center">Phương pháp</th>
<th style="text-align: center">MSE</th>
<th style="text-align: center">Cải thiện so với Baseline</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>Baseline</strong></td>
<td>641.4994</td>
<td>-</td>
</tr>
<tr>
<td><strong>Post-processing</strong></td>
<td>48.6205</td>
<td>92.42% ↓</td>
</tr>
<tr>
<td><strong>Smooth 20% (Best)</strong></td>
<td><strong>15.2606</strong></td>
<td><strong>97.62% ↓</strong></td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 8: Kết quả cuối cùng - So sánh hiệu suất của Baseline, Post-processing và Smooth 20% trên validation set (100 ngày cuối), bao gồm MSE và phần trăm cải thiện so với Baseline.

</p>

![Kaggle Leaderboard](https://i.ibb.co/TMh48y5v/a7653cba990d.png)

<p align="center">

Hình 23: Kaggle Leaderboard - Kết quả nộp bài trên Kaggle của nhóm nghiên cứu. Team CONQ999 với account "hg gggv" đứng thứ 5 với score MSE 15.0147. Kết quả này phản ánh hiệu suất của phương pháp PatchTST + Smooth Linear 20% + Post-processing trên test set của cuộc thi *AIO-2025: LTSF-Linear Forecasting Challenge**

</p>

Quan sát Hình 23 cho thấy kết quả nộp bài trên Kaggle của nhóm nghiên cứu. Team CONQ999 với account "hg gggv" đạt kết quả cao với **score MSE 15.0147**. Kết quả này tương đương với kết quả trên validation set (MSE 15.2606), chứng tỏ phương pháp đã được áp dụng đúng và có khả năng generalization tốt. Kết quả này cho thấy phương pháp PatchTST + Smooth Linear 20% + Post-processing đã đạt được hiệu suất tốt trong cuộc thi, đặc biệt là với chỉ một entry duy nhất.

💡 **Lưu ý:** Đây là kết quả cuối cùng chính thức từ notebook `patchtst_fixed_params.ipynb`. Kết quả có thể khác một chút so với notebook `patchtst_validation_analysis.ipynb` (MSE 44.94 và 17.49) do cách chia dữ liệu và validation folds khác nhau. Notebook `patchtst_fixed_params.ipynb` được sử dụng làm kết quả cuối cùng vì sử dụng fixed parameters từ Optuna và có quy trình validation nhất quán hơn.

Phân tích chi tiết cho thấy ba phương pháp. Phương pháp đầu tiên là Baseline (PatchTST với Fixed Parameters) với MSE 641.4994, model có bias hệ thống lớn (dự đoán cao hơn thực tế).

Phương pháp thứ hai là Post-processing (Linear Regression) với MSE 48.6205 (cải thiện 92.42%), công thức $y = 0.7267 \times \text{pred} + 9.3249$, và vấn đề là thay đổi giá trị đầu tiên, giảm độ tin cậy.

Phương pháp thứ ba là Best Method (Smooth Linear 20%) với MSE 15.2606 (cải thiện 97.62% so với baseline). Ưu điểm của phương pháp này là giữ nguyên giá trị đầu tiên (tăng độ tin cậy), kết hợp tốt nhất của smooth transition và post-processing, và cải thiện cả accuracy và reliability.

#### 5.4.4. So sánh với validation 100 ngày

Một phương pháp bổ sung được thử nghiệm: **chỉ so với 100 ngày validation** để chọn dữ liệu tốt nhất. Kết quả cho thấy dữ liệu có cải thiện thêm một chút, đạt được kết quả cuối cùng tốt nhất.

Kết quả cuối cùng cho thấy MSE 15.26 (cải thiện 97.62% so với baseline).

#### 5.4.5. File Submission

File submission cuối cùng (`submission_patchtst_fixed_params.csv`) có format `id, close` với 100 predictions. Range giá trị từ 96.55 đến 116.50, mean 104.40, std 4.90, và median 104.19.

Thống kê predictions cho thấy min 96.55, max 116.50, mean 104.40, std 4.90, và median 104.19. Phân tích thống kê predictions cho thấy predictions có phân phối hợp lý, không có giá trị bất thường, và phản ánh đúng xu hướng của dữ liệu. Kết quả này cho thấy file submission đã sẵn sàng để nộp.

---

## Phần 6: Tổng hợp và so sánh các phương pháp

Sau quá trình nghiên cứu và thử nghiệm, nghiên cứu tổng hợp lại tất cả các phương pháp đã áp dụng và so sánh kết quả theo MSE. Phần này trình bày một cách có hệ thống các cải tiến đã thực hiện.

### 6.1. Bảng so sánh tổng hợp theo MSE

<table>
<thead>
<tr>
<th style="text-align: center">Phương pháp</th>
<th style="text-align: center">MSE</th>
<th style="text-align: center">Ghi chú</th>
</tr>
</thead>
<tbody>
<tr>
<td><strong>PatchTST + Smooth 20%</strong></td>
<td><strong>15.26</strong></td>
<td><strong>⭐ BEST</strong></td>
</tr>
<tr>
<td>PatchTST + Post-processing</td>
<td>48.62</td>
<td>Tốt nhưng thay đổi giá trị đầu</td>
</tr>
<tr>
<td>PatchTST Baseline (Fixed)</td>
<td>641.50</td>
<td>Có bias hệ thống lớn</td>
</tr>
<tr>
<td>NLinear + Post-processing Lasso</td>
<td>98.52</td>
<td>Tốt nhất trong Linear models</td>
</tr>
<tr>
<td>PatchTST (SOTA comparison)</td>
<td>172.59</td>
<td>Baseline từ comparison</td>
</tr>
<tr>
<td>Rolling Elastic Net</td>
<td>52.93</td>
<td>Kết quả trên leaderboard</td>
</tr>
<tr>
<td>OLS Elastic Net</td>
<td>5406.96</td>
<td>Trend + Residual model</td>
</tr>
<tr>
<td>NLinear Baseline</td>
<td>587.38</td>
<td>Từ project gốc</td>
</tr>
<tr>
<td>DLinear Baseline</td>
<td>715.43</td>
<td>Từ project gốc</td>
</tr>
</tbody>
</table>

<p align="center">

Bảng 9: Bảng so sánh tổng hợp theo MSE - Tổng hợp tất cả các phương pháp đã áp dụng và so sánh kết quả theo MSE, bao gồm các mô hình từ project gốc, các pipeline thử nghiệm, và các phương pháp cải tiến.

</p>

### 6.2. Phân tích điểm mạnh và điểm yếu

#### 6.2.1. PatchTST + Smooth 20% (Phương pháp tốt nhất)

Phân tích phương pháp PatchTST + Smooth 20% cho thấy các điểm mạnh sau. MSE thấp nhất là 15.26, cải thiện 97.62% so với baseline. Phương pháp giữ nguyên giá trị đầu, tăng độ tin cậy. Phương pháp kết hợp tốt nhất của smooth transition và post-processing. Phương pháp reproducible vì sử dụng fixed parameters.

Tuy nhiên, nghiên cứu cũng nhận thấy các điểm yếu. Phương pháp phức tạp vì cần nhiều bước (Optuna, Post-processing, Smooth). Phương pháp cần thời gian để train và tối ưu. Phương pháp phụ thuộc vào dữ liệu vì cần đủ dữ liệu để train post-processing.

Phương pháp phù hợp cho bài toán dự đoán dài hạn (100 ngày), hiệu quả với dữ liệu có nhiều biến động, và có thể xử lý các sự kiện "thiên nga đen".

#### 6.2.2. PatchTST + Post-processing

PatchTST + Post-processing có điểm mạnh là MSE tốt với 48.62 (cải thiện 92.42%), đơn giản vì chỉ cần Post-processing, và hiệu quả vì giảm bias đáng kể. Tuy nhiên, phương pháp có điểm yếu là thay đổi giá trị đầu, giảm độ tin cậy, và MSE cao hơn Smooth 20% (48.62 vs 15.26). Phương pháp phù hợp khi không cần giữ nguyên giá trị đầu và nhanh hơn Smooth 20%.

#### 6.2.3. NLinear + Post-processing Lasso

NLinear + Post-processing Lasso có điểm mạnh là MSE tốt với 98.52 (tốt nhất trong Linear models), đơn giản vì model Linear dễ hiểu, và nhanh vì training nhanh hơn Transformer. Tuy nhiên, phương pháp có điểm yếu là MSE cao hơn PatchTST (98.52 vs 15.26) và không capture pattern phức tạp vì chỉ có Linear relationships. Phương pháp phù hợp cho bài toán đơn giản và khi cần model nhanh và dễ interpret.

#### 6.2.4. Rolling Elastic Net

Rolling Elastic Net có điểm mạnh là mô phỏng thực tế vì train lại tại mỗi thời điểm, tự động adapt vì thích ứng với dữ liệu mới, và MSE tốt với 52.93 (trên leaderboard). Tuy nhiên, phương pháp có điểm yếu là tốn thời gian vì cần train lại nhiều lần và phức tạp vì cần quản lý nhiều models. Phương pháp phù hợp cho online forecasting và khi cần adapt nhanh với dữ liệu mới.

### 6.3. So sánh theo từng giai đoạn phát triển

So sánh theo từng giai đoạn phát triển cho thấy bốn giai đoạn chính. Giai đoạn 1 là Project gốc (LTSF-Linear) với MSE ~0.05-0.18 (trên log scale, 5 ngày), horizon 5 ngày, và các models Linear, DLinear, NLinear.

Giai đoạn 2 là Pipeline Direct 100d với MSE 167.61-848.48 (tùy model), horizon 100 ngày, các models ElasticNet, Ridge, XGBoost, LightGBM, RF, GBDT, và best là GBDT (MSE: 167.61) và Ensemble (MSE: 299.79).

Giai đoạn 3 là Elastic Net Methods với OLS Elastic Net đạt MSE 5406.96 và Rolling Elastic Net đạt MSE 52.93 (leaderboard).

Giai đoạn 4 là SOTA Models + Bias Correction với NLinear + Post-processing đạt MSE 98.52, PatchTST Baseline đạt MSE 172.59, PatchTST + Post-processing đạt MSE 48.62, và PatchTST + Smooth 20% đạt MSE 15.26.

### 6.4. Các cải tiến chính đã áp dụng

Bốn cải tiến chính đã được áp dụng. Cải tiến đầu tiên là tối ưu Hyperparameters với Optuna, giảm MSE từ ~800 xuống ~192 (trên validation) và tìm được bộ parameters tối ưu.

Cải tiến thứ hai là Post-processing Regression, học cách điều chỉnh bias từ validation folds và giảm MSE từ 641.50 → 48.62 (92.42%).

Cải tiến thứ ba là Smooth Bias Correction, kết hợp smooth transition và post-processing, giảm MSE từ 48.62 → 15.26 (68.61%), và giữ nguyên giá trị đầu (tăng độ tin cậy).

Cải tiến thứ tư là Fixed Parameters, sử dụng best parameters từ Optuna, tiết kiệm thời gian và đảm bảo reproducibility. Các cải tiến này đã giúp đạt được kết quả tốt nhất.

### 6.5. Bài học rút ra

Qua quá trình nghiên cứu, năm bài học quan trọng được rút ra. Bài học đầu tiên là Bias Correction quan trọng. Model có thể có bias hệ thống lớn và post-processing có thể cải thiện đáng kể.

Bài học thứ hai là Smooth Transition hiệu quả. Giữ nguyên giá trị đầu tăng độ tin cậy và kết hợp tốt với post-processing.

Bài học thứ ba là Optuna hiệu quả. Optuna tự động tìm được hyperparameters tối ưu và tiết kiệm thời gian so với manual tuning.

Bài học thứ tư là Validation Analysis quan trọng. Phân tích kỹ lưỡng giúp hiểu vấn đề và từ đó phát triển giải pháp phù hợp.

Bài học thứ năm là Fixed Parameters hữu ích. Fixed Parameters tiết kiệm thời gian khi đã có best parameters và đảm bảo reproducibility. Các bài học này sẽ hữu ích cho các dự án tương lai.

Sau khi đã phát triển và tối ưu mô hình dự đoán, bước tiếp theo là xây dựng một hệ thống triển khai thực tế để người dùng có thể tương tác và sử dụng mô hình một cách dễ dàng. Phần này trình bày hệ thống giao diện người dùng (UI) được phát triển để triển khai mô hình vào thực tế.

---

## Phần 7: Giao diện Người dùng (UI) và Hệ thống Triển khai

Một hệ thống giao diện người dùng (UI) được xây dựng để triển khai mô hình dự đoán vào thực tế. Hệ thống sử dụng FastAPI làm backend và Streamlit làm frontend, cho phép người dùng tương tác với mô hình một cách trực quan và dễ dàng.

### 7.1. Tổng quan về Hệ thống UI

Hệ thống UI được thiết kế với hai chế độ hoạt động chính: **Realtime API** và **Upload CSV → API**. Mỗi chế độ phục vụ các nhu cầu khác nhau của người dùng, từ việc dự đoán với dữ liệu realtime đến việc tải lên dữ liệu tùy chỉnh để dự đoán.

**Chế độ Realtime API** cho phép người dùng lấy dữ liệu mới nhất từ internet (thông qua vnstock) và tự động dự đoán giá cổ phiếu FPT. Chế độ này phù hợp cho người dùng muốn có dự đoán cập nhật theo thời gian thực.

**Chế độ Upload CSV → API** cho phép người dùng tải lên file CSV chứa dữ liệu lịch sử OHLCV (Open, High, Low, Close, Volume) và dự đoán dựa trên dữ liệu này. Chế độ này phù hợp cho người dùng muốn kiểm tra mô hình với dữ liệu riêng hoặc dữ liệu không có sẵn trong hệ thống.

### 7.2. Chế độ Realtime API - Logic Xử lý Dữ liệu

Chế độ Realtime API có logic xử lý dữ liệu thông minh, tự động điều chỉnh dựa trên việc có hay không có dataset sẵn trong hệ thống.

#### 7.2.1. Trường hợp có Dataset sẵn

Khi hệ thống phát hiện có file dataset trong thư mục `data/raw/` (file có chứa "train" trong tên), logic xử lý như sau:

1. **Tải dữ liệu còn thiếu**: Hệ thống đọc file dataset hiện có, xác định ngày cuối cùng trong dataset, sau đó tự động tải phần còn lại của chuỗi thời gian từ ngày tiếp theo đến trước thời điểm fetch thông tin 1 ngày (để tránh dữ liệu chưa hoàn chỉnh).

2. **Tạo chuỗi thời gian historic**: Dữ liệu từ dataset và dữ liệu mới tải về được merge lại, sắp xếp theo thời gian, và loại bỏ các bản ghi trùng lặp. Kết quả là một chuỗi thời gian historic hoàn chỉnh từ đầu dataset đến ngày gần nhất.

3. **Dự đoán theo số ngày tùy chỉnh**: Người dùng có thể điều chỉnh số ngày dự đoán thông qua slider ở góc trái (từ 30 đến 100 ngày). Hệ thống sử dụng chuỗi historic này làm input cho mô hình PatchTST và dự đoán số ngày tương lai theo giá trị slider.

**Ví dụ**: Nếu dataset có dữ liệu từ 2020-08-03 đến 2024-12-31, và ngày hiện tại là 2025-01-15, hệ thống sẽ:
- Tải dữ liệu từ 2025-01-01 đến 2025-01-14 (trước 1 ngày so với ngày hiện tại)
- Merge với dataset hiện có để tạo chuỗi historic từ 2020-08-03 đến 2025-01-14
- Sử dụng chuỗi này để dự đoán 60 ngày tương lai (nếu slider được đặt ở 60)

![Giao diện Realtime API với Dataset sẵn có](https://i.ibb.co/qv66Jq9/b2c285d17ddf.png)

<p align="center">

Hình 24: Giao diện Realtime API khi có dataset sẵn - Sidebar hiển thị "Dataset found: FPT_train.csv" và các slider điều khiển. Khu vực chính hiển thị banner "Fetched new data from internet! Total: 1336 days", metrics cards (Latest Close ₫97.25K, Avg Projected Return -0.13%), và biểu đồ Market Pulse & Projection với đường historical (tím) và forecast (xanh nhạt) cho 60 ngày tương lai.

</p>

#### 7.2.2. Trường hợp chưa có Dataset sẵn

Khi hệ thống không phát hiện có file dataset trong `data/raw/`, ứng dụng cung cấp hai lựa chọn cho người dùng:

**Lựa chọn 1: Tải file lên**

Người dùng có thể tải lên file CSV chứa dữ liệu lịch sử. Sau khi tải lên và lưu vào `data/raw/`, logic phía sau sẽ hoạt động giống như trường hợp có dataset sẵn:
- Hệ thống đọc file vừa tải lên
- Xác định ngày cuối cùng và tải phần còn lại từ internet
- Merge để tạo chuỗi historic hoàn chỉnh
- Dự đoán theo số ngày từ slider

![Kết quả dự đoán sau khi tải file lên trong chế độ Realtime API](https://i.ibb.co/fdqBVdD5/dd0c687ec778.png)

<p align="center">

Hình 25: Kết quả dự đoán sau khi tải file lên - Sidebar hiển thị "Upload CSV file" được chọn và slider Historical days (330 ngày). Khu vực chính hiển thị kết quả realtime forecast với metrics (Latest Close ₫97.30K tăng 0.72%, Avg Projected Return -0.13%, Forecast Days 60d) và biểu đồ Market Pulse & Projection với đường historical (tím) từ 4/2024 đến cuối 2025 và đường forecast (xanh nhạt) cho 60 ngày tương lai với xu hướng giảm.

</p>

**Lựa chọn 2: Không tải file - Sử dụng slider**

Nếu người dùng chọn không tải file, hệ thống sẽ:
- Lấy số ngày lịch sử từ slider (mặc định 120 ngày, có thể điều chỉnh từ 20 đến 365 ngày)
- Tải dữ liệu historic từ internet dựa trên số ngày này (từ ngày hiện tại trừ đi số ngày trong slider, đến trước thời điểm fetch 1 ngày)
- Sử dụng dữ liệu này làm input cho mô hình
- Dự đoán số ngày tiếp theo theo giá trị slider (từ 30 đến 100 ngày)

**Ví dụ**: Nếu slider "Historical days" được đặt ở 120 ngày và slider "Forecast horizon" được đặt ở 60 ngày:
- Hệ thống tải dữ liệu từ (ngày hiện tại - 120 ngày) đến (ngày hiện tại - 1 ngày)
- Sử dụng 120 ngày này làm input
- Dự đoán 60 ngày tương lai

![Giao diện khi chưa có dataset sẵn](https://i.ibb.co/wFh1vtPC/9d7a39ab5f5e.png)

<p align="center">

Hình 26: Giao diện khi chưa có dataset sẵn - Sidebar hiển thị cảnh báo "No dataset found in data/raw/" và hai lựa chọn: "Upload CSV file" hoặc "Fetch from internet (use slider)". Khu vực chính hiển thị tiêu đề "FPT Stock Intelligence" và banner hướng dẫn người dùng chọn một trong hai tùy chọn trong sidebar.

</p>

### 7.3. Chế độ Upload CSV → API

Chế độ này cho phép người dùng tải lên file CSV chứa dữ liệu OHLCV và dự đoán dựa trên dữ liệu này. Quy trình hoạt động như sau:

1. **Tải file CSV**: Người dùng chọn file CSV từ máy tính với các cột bắt buộc: `time`, `open`, `high`, `low`, `close`, `volume`.

2. **Xác thực dữ liệu**: Hệ thống kiểm tra file có đủ các cột cần thiết và có ít nhất 20 dòng dữ liệu (để đảm bảo có đủ dữ liệu lịch sử cho mô hình).

3. **Dự đoán theo số ngày**: Người dùng điều chỉnh slider "Forecast horizon" (từ 30 đến 100 ngày) và nhấn nút "Run prediction". Hệ thống gửi dữ liệu lên API endpoint `/api/v1/predict/multi` với số ngày dự đoán tương ứng.

4. **Hiển thị kết quả**: Kết quả dự đoán được hiển thị dưới dạng biểu đồ (historical + forecast), bảng metrics (Latest Close, Avg Projected Return, Forecast Days), và bảng chi tiết các dự đoán.

![Kết quả dự đoán từ chế độ Upload CSV → API](https://i.ibb.co/LXcKQJHD/a89076c70d52.png)

<p align="center">

Hình 27: Kết quả dự đoán từ chế độ Upload CSV → API - Dashboard hiển thị Key Metrics (Latest Close ₫119.19K, Avg Projected Return -0.35%, Forecast Days 60d), biểu đồ Market Pulse & Projection với đường historical (tím) và forecast (xanh nhạt), và bảng Upcoming Forecast Snapshots liệt kê chi tiết từng ngày dự đoán với giá và phần trăm thay đổi.

</p>

### 7.4. Giao diện và Trải nghiệm Người dùng

Giao diện được thiết kế với các tính năng sau:

- **Biểu đồ tương tác**: Sử dụng Plotly để hiển thị biểu đồ giá với đường historical (màu tím) và đường forecast (màu xanh), có đường phân cách rõ ràng giữa dữ liệu lịch sử và dự đoán.

- **Metrics cards**: Hiển thị các chỉ số quan trọng như Latest Close, Avg Projected Return, và Forecast Days với định dạng dễ đọc.

- **Bảng dự đoán chi tiết**: Hiển thị từng ngày dự đoán với giá dự kiến và phần trăm thay đổi so với ngày trước.

- **Sidebar điều khiển**: Tất cả các tùy chọn (chế độ, slider, upload file) được đặt trong sidebar để dễ truy cập và không làm rối giao diện chính.

### 7.5. Hướng dẫn Chạy Hệ thống

Để chạy hệ thống, người dùng thực hiện các bước sau:

1. **Clone repository**:

```bash
git clone https://github.com/sonvt8/AIO2025_Project6.1_StockForcasting.git  # Tải repository về máy
cd AIO2025_Project6.1_StockForcasting                                        # Di chuyển vào thư mục project
git checkout tags/version-1.0-elasticnet                                     # Chuyển sang phiên bản 1.0 (ElasticNet version)
```
   
   **Giải thích lệnh:**
   - `git clone`: Tải toàn bộ mã nguồn từ GitHub về máy local.
   - `cd`: Di chuyển vào thư mục vừa clone để thực hiện các lệnh tiếp theo.
   - `git checkout tags/version-1.0-elasticnet`: Chuyển sang tag cụ thể của repository, đảm bảo sử dụng đúng phiên bản đã được test và ổn định.

2. **Tạo môi trường ảo và cài đặt dependencies**:

```bash
python -m venv venv                    # Tạo môi trường ảo Python tên "venv"
# Windows
venv\Scripts\activate                  # Kích hoạt môi trường ảo trên Windows
# macOS/Linux
source venv/bin/activate               # Kích hoạt môi trường ảo trên macOS/Linux
pip install -r requirements.txt        # Cài đặt tất cả packages trong requirements.txt
```
   
   **Giải thích lệnh:**
   - `python -m venv venv`: Tạo một môi trường Python độc lập trong thư mục `venv`, giúp tránh xung đột với các packages đã cài đặt trên hệ thống.
   - `venv\Scripts\activate` (Windows) hoặc `source venv/bin/activate` (macOS/Linux): Kích hoạt môi trường ảo, sau đó mọi lệnh `pip install` sẽ cài vào môi trường này thay vì hệ thống.
   - `pip install -r requirements.txt`: Đọc file `requirements.txt` (chứa danh sách packages và phiên bản) và cài đặt tất cả dependencies cần thiết cho project.

3. **Chuẩn bị dữ liệu**: Tạo thư mục `data/raw/` và đặt file `FPT_train.csv` (dataset) vào thư mục này.

4. **Chạy API backend**:

```bash
uvicorn app.main:app --reload    # Khởi động FastAPI server với chế độ auto-reload
```
   
   **Giải thích lệnh:**
   - `uvicorn`: ASGI server để chạy ứng dụng FastAPI.
   - `app.main:app`: Chỉ định module `app.main` và instance `app` (FastAPI application).
   - `--reload`: Tự động reload server khi có thay đổi code, hữu ích khi development.
   
   Lần đầu chạy, hệ thống sẽ hỏi "Train models now? (y/n):" trong console. Chọn `y` để tạo model (nếu chưa có artifacts). Server sẽ chạy tại `http://localhost:8000` và API documentation có thể truy cập tại `http://localhost:8000/docs`.

5. **Chạy Streamlit frontend**:

```bash
streamlit run frontend/streamlit_app/app.py    # Khởi động Streamlit web app
```
   
   **Giải thích lệnh:**
   - `streamlit run`: Lệnh để khởi động Streamlit application.
   - `frontend/streamlit_app/app.py`: Đường dẫn đến file Python chứa Streamlit app.
   - Streamlit sẽ tự động mở trình duyệt và hiển thị giao diện tại `http://localhost:8501`.

6. **Truy cập giao diện**: Mở trình duyệt và truy cập `http://localhost:8501` để sử dụng giao diện. Nếu trình duyệt không tự động mở, có thể truy cập thủ công bằng URL này.

Hệ thống UI này cho phép người dùng dễ dàng tương tác với mô hình dự đoán đã được huấn luyện, từ việc dự đoán realtime đến việc kiểm tra với dữ liệu tùy chỉnh, tạo nên một công cụ hoàn chỉnh cho việc dự đoán giá cổ phiếu FPT.

Sau khi đã trình bày toàn bộ quá trình phát triển từ mô hình Linear đơn giản đến hệ thống UI hoàn chỉnh, phần cuối cùng tổng hợp lại các thành tựu, bài học quan trọng, và các lưu ý cần thiết.

---

## Phần 8: Kết luận và Disclaimer

### 8.1. Kết luận

Qua quá trình nghiên cứu và phát triển, những thành tựu đáng kể đã được ghi nhận:

#### 8.1.1. Thành tựu chính

Thành tựu đầu tiên là cải thiện 97.62% MSE, từ baseline MSE 641.50 xuống Best Method MSE 15.26. Đây là một cải thiện rất đáng kể, chứng minh hiệu quả của các phương pháp đã áp dụng.

Thành tựu thứ hai là giảm bias hệ thống. Model ban đầu có bias hệ thống lớn (dự đoán cao hơn thực tế). Sau khi áp dụng Smooth Bias Correction, bias đã được điều chỉnh đáng kể.

Thành tựu thứ ba là mở rộng horizon dự đoán từ 5 ngày lên 100 ngày. Từ project gốc chỉ dự đoán ngắn hạn (5 ngày), nghiên cứu đã phát triển thành công hệ thống dự đoán dài hạn (100 ngày), đòi hỏi mô hình phải nắm bắt được cả xu hướng dài hạn và biến động ngắn hạn. Đây là một thách thức lớn trong time series forecasting, đặc biệt với dữ liệu tài chính có nhiều biến động.

Thành tựu thứ tư là chuyển đổi từ mô hình Linear đơn giản sang Transformer hiện đại. Từ các mô hình LTSF-Linear (Linear, DLinear, NLinear) chỉ có khả năng học pattern tuyến tính, nghiên cứu đã áp dụng thành công PatchTST - một mô hình Transformer chuyên biệt cho time series với khả năng học pattern phức tạp, long-range dependencies, và xử lý non-stationarity thông qua Reversible Instance Normalization (RevIN). Sự chuyển đổi này cho phép mô hình nắm bắt được các mối quan hệ phức tạp trong dữ liệu tài chính mà các mô hình linear không thể.

Thành tựu thứ năm là xử lý được sự kiện "thiên nga đen" không có trong training data. Bài toán diễn ra trong bối cảnh có sự kiện chưa có tiền lệ (thuế của Tổng thống Trump đối với hàng hóa xuất khẩu Việt Nam), sự kiện này không xuất hiện trong dữ liệu training nhưng vẫn đạt được kết quả tốt (MSE 15.26). Điều này chứng minh khả năng generalization của mô hình và hiệu quả của các phương pháp bias correction trong việc xử lý distribution shift.

Thành tựu thứ sáu là phát triển phương pháp Bias Correction hiệu quả. Nghiên cứu đã phát triển và kết hợp thành công hai kỹ thuật: Post-processing Regression (học công thức điều chỉnh từ validation folds) và Smooth Bias Correction (kết hợp smooth transition với post-processing). Phương pháp này không chỉ giải quyết được vấn đề bias lạc quan của mô hình mà còn giữ được tính liên tục và độ tin cậy của predictions bằng cách giữ nguyên giá trị đầu tiên.

#### 8.1.2. Phương pháp tốt nhất

**PatchTST + Smooth Linear 20% + Post-processing (Linear Regression)** được xác định là phương pháp tốt nhất. Phương pháp này đạt được chỉ số MSE 15.26 (cải thiện 97.62% so với baseline).

Phương pháp tốt nhất bao gồm ba thành phần chính. Thành phần đầu tiên là PatchTST với fixed hyperparameters (đã tối ưu từ Optuna). Thành phần thứ hai là Post-processing Regression (Linear Regression) để điều chỉnh predictions. Thành phần thứ ba là Smooth Bias Correction (Linear 20%), với phần đầu (20%) sử dụng smooth transition (giữ nguyên giá trị đầu) và phần cuối (80%) sử dụng Post-processing Regression. Kết hợp ba thành phần này giúp đạt được kết quả tốt nhất.

Tuy nhiên, cần nhận định rằng PatchTST + Smooth Bias Correction là một giải pháp tình thế, được phát triển để đối phó với các thách thức cụ thể của bài toán này. Phương pháp này giống như việc sử dụng một cách thô sơ để đi qua khe cửa hẹp - mặc dù đạt được kết quả tốt về mặt số liệu, nhưng không phải là một giải pháp lý tưởng về mặt lý thuyết. Việc phải kết hợp nhiều lớp điều chỉnh (Post-processing Regression và Smooth Bias Correction) cho thấy mô hình gốc vẫn còn những hạn chế trong việc nắm bắt đúng pattern của dữ liệu. Giải pháp này phù hợp với mục tiêu của cuộc thi là đạt kết quả tốt nhất, nhưng trong thực tế, các mô hình tuyến tính đơn giản hơn với kỹ thuật điều chỉnh phù hợp có thể là lựa chọn tốt hơn về mặt tính đơn giản, tốc độ và khả năng giải thích.

#### 8.1.3. Nhận định về các mô hình tuyến tính

Phù hợp với mục tiêu của cuộc thi [**AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge) - tập trung vào việc đánh giá hiệu quả của các mô hình tuyến tính trong dự báo chuỗi thời gian dài hạn, nghiên cứu này đã đánh giá và so sánh các mô hình LTSF-Linear (Linear, NLinear, DLinear) với các phương pháp khác.

**Nhận định đầu tiên: Các mô hình tuyến tính có khả năng nắm bắt pattern cơ bản trong dữ liệu tài chính.** Các mô hình LTSF-Linear, đặc biệt là NLinear, đã chứng minh khả năng học được các mối quan hệ tuyến tính và xu hướng cơ bản trong dữ liệu giá cổ phiếu. Mặc dù đơn giản, các mô hình này có thể đạt hiệu quả tốt khi được kết hợp với các kỹ thuật điều chỉnh bias phù hợp.

**Nhận định thứ hai: NLinear với Post-processing đạt hiệu quả tốt hơn nhiều mô hình phức tạp.** Kết quả so sánh cho thấy NLinear với Post-processing đạt MSE 98.52, tốt hơn đáng kể so với các mô hình phức tạp như TimesNet (MSE 422.96), NBEATS (MSE 363.90), và NHITS (MSE 221.80). Điều này chứng minh rằng các mô hình tuyến tính, khi được áp dụng đúng cách, có thể đạt hiệu quả tốt trong dự báo time series, đặc biệt với dữ liệu tài chính có nhiều biến động.

**Nhận định thứ ba: Các mô hình tuyến tính có ưu điểm về tính đơn giản, tốc độ và khả năng giải thích.** So với các mô hình deep learning phức tạp, các mô hình tuyến tính có thời gian training và inference nhanh hơn đáng kể, yêu cầu tài nguyên tính toán thấp hơn, và dễ dàng giải thích kết quả dự đoán. Điều này làm cho chúng phù hợp hơn cho các ứng dụng thực tế cần độ tin cậy và khả năng giải thích cao.

**Nhận định thứ tư: Kỹ thuật điều chỉnh bias là yếu tố quan trọng để cải thiện hiệu quả của các mô hình tuyến tính.** Nghiên cứu cho thấy việc kết hợp các mô hình tuyến tính với Post-processing Regression (Linear/Ridge/Lasso) giúp cải thiện đáng kể độ chính xác dự đoán. Các kỹ thuật này học được pattern bias từ validation folds và áp dụng để điều chỉnh predictions, giúp các mô hình tuyến tính xử lý tốt hơn các vấn đề như bias hệ thống và distribution shift.

**Nhận định thứ năm: Các mô hình tuyến tính phù hợp với dự báo dài hạn khi kết hợp với kỹ thuật phù hợp.** Mặc dù các mô hình tuyến tính ban đầu được thiết kế cho dự báo ngắn hạn, nghiên cứu đã chứng minh rằng chúng có thể được mở rộng cho dự báo dài hạn (100 ngày) khi được kết hợp với các kỹ thuật như Rolling Window, Post-processing, và Smooth Bias Correction. Điều này mở ra khả năng ứng dụng các mô hình tuyến tính trong các bài toán dự báo dài hạn với dữ liệu tài chính.

Kết luận, các mô hình tuyến tính (Linear, NLinear, DLinear) đã chứng minh giá trị của chúng trong dự báo time series dài hạn với dữ liệu tài chính thực tế. Mặc dù không đạt kết quả tốt nhất tuyệt đối so với PatchTST, chúng cung cấp một giải pháp cân bằng tốt giữa hiệu quả, tính đơn giản, tốc độ và khả năng giải thích, phù hợp với mục tiêu của cuộc thi là khám phá sức mạnh của các mô hình tuyến tính thay vì dựa vào các mô hình deep learning phức tạp.

#### 8.1.4. Quá trình phát triển

Từ project gốc sử dụng LTSF-Linear models (dự đoán 5 ngày), một hệ thống hoàn chỉnh đã được phát triển. Quá trình phát triển gồm sáu bước chính. Bước đầu tiên là Pipeline Direct 100d, xây dựng pipeline để dự đoán 100 ngày. Bước thứ hai là Elastic Net Methods, thử nghiệm OLS và Rolling Elastic Net. Bước thứ ba là Bias Correction, phát triển các phương pháp điều chỉnh bias. Bước thứ tư là SOTA Models, so sánh và chọn PatchTST. Bước thứ năm là Optuna Optimization, tối ưu hyperparameters. Bước cuối cùng là Smooth Correction, kết hợp smooth transition và post-processing. Mỗi bước đóng góp quan trọng vào kết quả cuối cùng.

#### 8.1.5. Bài học quan trọng

Qua quá trình nghiên cứu từ Pipeline Direct 100d, Elastic Net Methods, SOTA Models Comparison, đến PatchTST với Bias Correction, bảy bài học quan trọng được rút ra.

**Bài học thứ nhất: Feature Engineering và Feature Selection quan trọng.** Từ Pipeline Direct 100d, việc xây dựng hệ thống đặc trưng phong phú (returns, volatility, moving averages, technical indicators, lag features) và feature selection đa chiều (Mutual Information, correlation, Stability Selection, Predictability Score) đóng vai trò quan trọng. Ensemble (Stack Ridge Ensemble) đạt MSE 299.79, tốt hơn các model đơn lẻ, chứng minh việc kết hợp nhiều mô hình có thể cải thiện kết quả.

**Bài học thứ hai: Rolling Window phù hợp với Time Series có nhiều biến động.** Từ so sánh OLS Elastic Net (MSE 5406.96) và Rolling Elastic Net (MSE 52.93), Rolling Elastic Net cho kết quả tốt hơn đáng kể vì mô phỏng đúng cách mô hình hoạt động trong thực tế - train lại tại mỗi thời điểm và tự động thích ứng với dữ liệu mới. Tuy nhiên, mô hình có xu hướng học từ pattern gần nhất, có thể dẫn đến bias khi thị trường thay đổi hướng.

**Bài học thứ ba: Mô hình đơn giản có thể tốt hơn mô hình phức tạp.** Từ so sánh SOTA Models, NLinear (mô hình Linear đơn giản) với Post-processing đạt MSE 98.52, tốt hơn các mô hình phức tạp như TimesNet (422.96), NBEATS (363.90), NHITS (221.80). Điều này cho thấy không phải lúc nào mô hình phức tạp cũng tốt hơn, và việc kết hợp mô hình đơn giản với post-processing có thể đạt kết quả tốt.

**Bài học thứ tư: So sánh toàn diện các phương pháp giúp tìm ra giải pháp tốt nhất.** Việc so sánh tất cả các mô hình SOTA cùng với các phương pháp bias correction khác nhau (Baseline, Bias Correction, Post-processing Linear/Ridge/Lasso) cho phép đánh giá khách quan và tìm ra phương pháp tối ưu. Post-processing Regression (Linear/Ridge/Lasso) hiệu quả hơn Bias Correction đơn giản (trừ bias trung bình), chứng minh việc học pattern bias phức tạp quan trọng hơn điều chỉnh một giá trị cố định.

**Bài học thứ năm: TimeSeriesSplit giúp học được pattern bias từ validation folds.** Thay vì chỉ sử dụng một tập validation cố định, TimeSeriesSplit tạo nhiều folds theo thời gian, cho phép thu thập nhiều cặp (prediction, ground_truth) từ các phần dữ liệu khác nhau. Mô hình post-processing học được pattern của bias từ các folds này, sau đó áp dụng cho predictions cuối cùng. Cách tiếp cận này hiệu quả hơn việc chỉ điều chỉnh một giá trị bias cố định.

**Bài học thứ sáu: Smooth Transition giữ tính liên tục và tăng độ tin cậy.** Từ phát hiện rằng Post-processing trực tiếp thay đổi giá trị đầu tiên (giảm độ tin cậy), phương pháp Smooth Bias Correction được phát triển. Phần đầu (20%) sử dụng smooth transition (giữ nguyên giá trị đầu), phần cuối (80%) sử dụng post-processing trực tiếp. Phương pháp này vừa giải quyết được bias lạc quan, vừa giữ được tính liên tục của predictions, đạt MSE 15.26 tốt hơn Post-processing trực tiếp (MSE 48.62).

**Bài học thứ bảy: Tối ưu hóa và Reproducibility quan trọng.** Optuna tự động tìm được hyperparameters tối ưu (giảm MSE từ ~800 xuống ~192 trên validation), tiết kiệm thời gian so với manual tuning. Sau khi có best parameters, sử dụng Fixed Parameters đảm bảo reproducibility và tiết kiệm thời gian khi không cần chạy Optuna lại. Validation Analysis kỹ lưỡng (3-fold cross-validation) giúp hiểu rõ vấn đề (bias lạc quan, heteroscedasticity) và từ đó phát triển giải pháp phù hợp.

Các bài học này sẽ hữu ích cho các dự án nghiên cứu tương lai, đặc biệt trong lĩnh vực time series forecasting với dữ liệu tài chính có nhiều biến động.

### 8.2. Disclaimer

#### 8.2.1. Mục đích nghiên cứu

Các phương pháp được trình bày trong bài viết này được phát triển **chỉ để thực nghiệm và mô phỏng lại giá cổ phiếu trong khuôn khổ [cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge)**. 

**Không được sử dụng với mục đích tài chính thực tế.**

Cần nhấn mạnh rằng các phương pháp này chỉ phù hợp cho mục đích học tập và nghiên cứu.

#### 8.2.2. Hạn chế của phương pháp

Các hạn chế sau được nhận thấy. Hạn chế đầu tiên là chỉ sử dụng dữ liệu được cung cấp. Trong khuôn khổ chương trình, chỉ sử dụng dữ liệu được chương trình cung cấp. Không thể áp dụng các dữ liệu hoặc số liệu phản ánh các yếu tố thực tế như kinh tế vĩ mô (GDP, lạm phát, lãi suất), tài chính (báo cáo tài chính, chỉ số P/E, P/B), địa chính trị (chính sách, quan hệ quốc tế), thị trường (tâm lý nhà đầu tư, dòng tiền), và các cổ phiếu cùng ngành (correlation, sector analysis).

Hạn chế thứ hai là chưa ổn định. Kết quả có thể thay đổi tùy thuộc vào dữ liệu và không đảm bảo sẽ cho kết quả tốt trong mọi trường hợp.

Hạn chế thứ ba là có thể không chính xác trong nhiều trường hợp khác. Phương pháp được tối ưu cho dữ liệu FPT trong khoảng thời gian cụ thể và có thể không phù hợp với các cổ phiếu khác, các khoảng thời gian khác, hoặc các điều kiện thị trường khác. Khuyến nghị được đưa ra là cần cẩn trọng khi áp dụng các phương pháp này.

#### 8.2.3. Về sự kiện "thiên nga đen"

Dữ liệu của [**cuộc thi AIO-2025: LTSF-Linear Forecasting Challenge**](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge) này diễn ra trong bối cảnh có sự kiện "thiên nga đen" - khi Tổng thống Donald Trump công bố sắc thuế và áp thuế rất nặng đối với hàng hóa xuất khẩu của Việt Nam. Đây là sự kiện chưa có tiền lệ, không có trong dữ liệu lịch sử, rất khó dự đoán vì không thể dự đoán được từ dữ liệu training, và có ảnh hưởng lớn, tác động mạnh đến giá cổ phiếu. Các phương pháp đã được phát triển để xử lý các biến động lớn, nhưng không đảm bảo sẽ xử lý được mọi sự kiện "thiên nga đen" trong tương lai.

#### 8.2.4. Khuyến nghị

Khuyến nghị đầu tiên là không sử dụng cho mục đích đầu tư thực tế. Các phương pháp này chỉ phù hợp cho mục đích học tập và nghiên cứu. Không nên sử dụng để đưa ra quyết định đầu tư thực tế.

Khuyến nghị thứ hai là cần bổ sung thêm dữ liệu. Để có kết quả tốt hơn, cần bổ sung các yếu tố như kinh tế vĩ mô, tài chính, địa chính trị, thị trường, và các cổ phiếu cùng ngành.

Khuyến nghị thứ ba là cần kiểm tra kỹ lưỡng. Trước khi áp dụng, cần kiểm tra kỹ lưỡng trên nhiều dữ liệu khác nhau và đảm bảo phương pháp phù hợp với điều kiện cụ thể.

Khuyến nghị thứ tư là cần cập nhật thường xuyên. Thị trường chứng khoán thay đổi liên tục và cần cập nhật model và phương pháp thường xuyên.

### 8.3. Lời kêu gọi hành động

Nếu quan tâm đến dự án này, độc giả được khuyến khích tìm hiểu thêm bằng cách đọc các notebook và code trong repository, thử nghiệm với các phương pháp khác nhau, và phát triển các cải tiến mới. Độc giả cũng được khuyến khích chia sẻ ý kiến, đóng góp ý kiến và phản hồi, và chia sẻ kinh nghiệm và bài học. Độc giả được khuyến khích nghiên cứu sâu hơn bằng cách tìm hiểu về các mô hình time series forecasting, nghiên cứu về bias correction methods, và phát triển các phương pháp mới. Cuối cùng, độc giả được khuyến khích ứng dụng có trách nhiệm bằng cách chỉ sử dụng cho mục đích học tập và nghiên cứu, không sử dụng cho mục đích đầu tư thực tế, và tuân thủ các quy định và đạo đức nghiên cứu.

---

## Tài liệu tham khảo

1. **AIO-2025: LTSF-Linear Forecasting Challenge:** [Kaggle Competition](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge)
2. **PatchTST Paper:** A Time Series is Worth 64 Words: Long-term Forecasting with Transformers (2022)
3. **Optuna Documentation:** Hyperparameter Optimization Framework
4. **NeuralForecast Library:** Time Series Forecasting Library
5. **Scikit-learn Documentation:** TimeSeriesSplit, Linear Regression
6. **LTSF-Linear Paper:** Are Transformers Effective for Time Series Forecasting? (2022)

---

## Repository GitHub

Mã nguồn và các notebook của dự án được công khai trên các repository sau:

- **[LTSF_Linear](https://github.com/HoangVo-Prog/LTSF_Linear)**: Repository chứa các implementation của các mô hình LTSF-Linear, Rolling Elastic Net, OLS Elastic Net, và PatchTST.
- **[AIO2025_Project6.1_StockForcasting](https://github.com/sonvt8/AIO2025_Project6.1_StockForcasting.git)**: Repository chứa hệ thống deployment với FastAPI và Streamlit, cùng các model artifacts.
- **[AIO Project 6](https://github.com/D9Dre4mer/AIO/tree/D9Dre4mer-AIO-Project/251201%20Project%206)**: Repository chứa phần thực nghiệm mô hình, bao gồm các notebook liên quan đến PatchTST được sử dụng trong bài viết này.
