# Cross-GPU Reproducibility: Khi Cùng Mô Hình, Cùng Seed Vẫn Cho Kết Quả Khác Nhau

*Tóm tắt: Bài viết phân tích vấn đề reproducibility khi train cùng mô hình trên các GPU khác nhau. Dựa trên 65 thực nghiệm trên 6 GPU/TPU, ta sẽ cùng tìm hiểu nguyên nhân gốc rễ và đưa ra giải pháp thực tế để đạt được reproducibility tốt nhất có thể.*

---

## Giới thiệu - Nối tiếp Series Hardware

Trong bài viết trước ["DIY AI Hardware Architecture — In a Nutshell"](https://aioconquer.aivietnam.edu.vn/posts/diy-ai-hardware-architecture-in-a-nutshell), ta đã cùng nhau tìm hiểu về các loại phần cứng phù hợp cho hệ thống AI - từ GPU, CPU đến các giải pháp chuyên dụng như TPU và FPGA. Ta cũng đã phân tích cách lựa chọn phần cứng phù hợp với nhu cầu cụ thể, từ giải pháp miễn phí như Google Colab đến các hệ thống enterprise như NVIDIA DGX A100.

Việc chọn đúng phần cứng là cực kỳ quan trọng, nếu không chúng ta sẽ dễ dàng gặp phải vấn đề: **dù cùng mô hình, cùng random seed nhưng kết quả lại khác nhau khi train trên các GPU khác nhau**. Hiện tượng này xảy ra không chỉ ở các hệ thống lớn mà còn xuất hiện ngay trên các GPU phổ biến cho cá nhân và phòng lab nhỏ như T4, L4, A100, RTX 3060, RTX 5090, thậm chí cả TPU của Google.

Bài viết này sẽ tiếp nối series Hardware bằng cách đi sâu vào vấn đề **cross-GPU reproducibility** - tính tái lập kết quả giữa các GPU khác nhau. Dựa trên 65 thực nghiệm chi tiết, ta sẽ cùng phân tích nguyên nhân gốc rễ, từ khác biệt số học (floating-point precision) đến các tối ưu phần cứng mặc định, và đưa ra các giải pháp thực tế để đạt được reproducibility tốt nhất có thể.

Tất cả code và notebook thực nghiệm được trình bày trong bài viết này đều có sẵn trong [GitHub repository](https://github.com/D9Dre4mer/AIO2025_Blogs/tree/main/251226%20Blog%20Hardware%202) ([`AIO2025_Blogs/251226 Blog Hardware 2`](https://github.com/D9Dre4mer/AIO2025_Blogs/tree/main/251226%20Blog%20Hardware%202)). Người đọc có thể truy cập repository để xem chi tiết các notebook training trên từng GPU/TPU, code phân tích kết quả, và tất cả các file JSON chứa kết quả thực nghiệm. Điều này cho phép người đọc reproduce và verify các findings được trình bày trong bài viết.

---

## ACT 1: Phát hiện Vấn đề (The Discovery)

### 1.1 Câu chuyện thực tế

Hãy tưởng tượng bạn đã chọn được GPU phù hợp, thiết lập môi trường training, và bắt đầu huấn luyện mô hình. Bạn đặt random seed = 42 để đảm bảo reproducibility, chạy training trên GPU của mình, và đạt được test accuracy 77.92%. Sau đó, bạn chia sẻ code với đồng nghiệp, họ chạy trên GPU khác với cùng seed, nhưng kết quả lại là 78.45% - chênh lệch 0.53%.

Câu hỏi đặt ra: **Tại sao lại như vậy?** Cùng mô hình, cùng seed, cùng dataset, nhưng kết quả khác nhau. Đây không phải là lỗi code hay cấu hình sai, mà là một hiện tượng phổ biến trong deep learning.

<div align="center">

![Reproducibility Challenge](https://i.ibb.co/wrWK46dh/Reproducibility-Challenge.png)

**Hình 1:** Minh họa thách thức reproducibility — cùng một mô hình deep learning chạy trên các GPU platform khác nhau (A, B, C) cho kết quả hơi khác nhau, tạo ra "Reproducibility Challenge" (hình AI minh hoạ).

</div>

Quan sát Hình 1, ta có thể thấy rằng mặc dù cùng một mô hình deep learning, nhưng khi chạy trên các GPU platform khác nhau, kết quả có sự khác biệt nhỏ (như các giá trị 0.9874, 0.9872, 0.9875). Đây chính là vấn đề mà ta sẽ phân tích chi tiết trong bài viết này.

### 1.2 Mô tả Vấn đề với Số liệu Thực nghiệm

Để hiểu rõ vấn đề, ta đã thực hiện **65 thực nghiệm** trên **6 GPU/TPU khác nhau**: Tesla T4, NVIDIA L4, NVIDIA A100-SXM4-40GB, NVIDIA GeForce RTX 3060, NVIDIA GeForce RTX 5090, và TPU v6e-1. Mỗi thực nghiệm sử dụng cùng một mô hình SimpleCNN trên dataset CIFAR-10, cùng random seed (42), và cùng hyperparameters.

**Phát hiện quan trọng**: Với cùng cấu hình baseline (TF32=False, AMP=False, num_workers=0), kết quả accuracy khác nhau giữa các GPU:

| GPU | Mean Accuracy | Std | Min | Max |
|-----|--------------|-----|-----|-----|
| TPU | 78.45% | 0.005% | 78.45% | 78.46% |
| NVIDIA GeForce RTX 3060 | 77.92% | 0.24% | 77.63% | 78.28% |
| NVIDIA L4 | 77.85% | 0.10% | 77.75% | 77.95% |
| NVIDIA GeForce RTX 5090 | 77.81% | 0.07% | 77.74% | 77.91% |
| NVIDIA A100-SXM4-40GB | 77.71% | 0.10% | 77.61% | 77.81% |
| Tesla T4 | 77.64% | 0.05% | 77.59% | 77.68% |

**Max difference**: 0.87% (giữa TPU 78.45% và T4 77.64%)

⚠️ **Lưu ý về Std của TPU**: Std của TPU rất nhỏ (0.005%) vì chỉ có 2 thực nghiệm với config baseline (`TF32=False, AMP=False, num_workers=0`). Hai thực nghiệm này sử dụng precision khác nhau:
- **FP32** (bf16_enabled=False): 78.46%
- **BFloat16** (bf16_enabled=True): 78.45%

TPU cho thấy dấu hiệu reproducibility cao trong phạm vi thực nghiệm này - ngay cả khi thay đổi precision (FP32 vs BFloat16), kết quả vẫn gần như giống hệt nhau (chênh lệch chỉ 0.01%). Tuy nhiên, số lượng runs còn hạn chế (chỉ 2 thực nghiệm với config baseline) nên chưa thể kết luận tổng quát. Với GPU thì baseline chỉ dùng FP32, nên việc so sánh cần thận trọng. Cần thêm nhiều thực nghiệm để xác nhận liệu TPU có thực sự có tính deterministic cao hơn GPU hay không.

Quan sát training curves, ta thấy rằng mặc dù giá trị accuracy cuối cùng khác nhau, nhưng **xu hướng học tập (training trajectory) lại rất tương đồng** giữa các GPU. Correlation giữa các trajectories đạt 0.978-0.999, cho thấy mô hình học theo cùng một cách, chỉ khác nhau về giá trị tuyệt đối.

<div align="center">

![Training Trajectory Comparison](https://i.ibb.co/xqY0drDX/Training-Trajectory-Comparison.png)

**Hình 2:** Training curves so sánh giữa các GPU - xu hướng tương đồng nhưng giá trị khác nhau.

</div>

Quan sát Hình 2, ta có thể thấy rằng tất cả các đường cong accuracy đều có hình dạng tương tự nhau. Chúng đều tăng dần từ epoch đầu tiên, đạt peak ở khoảng epoch 7-8, và sau đó ổn định. Điều này cho thấy mô hình học theo cùng một cách trên tất cả các GPU, chỉ khác nhau về giá trị tuyệt đối.

### 1.3 Phương pháp và Thiết lập Thực nghiệm

**Mục tiêu thực nghiệm**: Chứng minh và phân tích vấn đề cross-GPU reproducibility - cùng mô hình, cùng seed nhưng kết quả khác nhau trên GPU khác nhau. Ta muốn tìm hiểu nguyên nhân gốc rễ và đánh giá mức độ ảnh hưởng của các yếu tố như cấu hình mặc định, precision, và optimizer.

**Nơi thực hiện**: Các thực nghiệm được thực hiện trên hai môi trường chính:

- **Google Colab**: Sử dụng GPU T4, L4, A100 và TPU v6e-1 cho các thực nghiệm baseline và một số test cấu hình. Đây là môi trường miễn phí, dễ tiếp cận, phù hợp cho các thực nghiệm ban đầu và kiểm chứng ý tưởng.

- **Phòng lab**: Sử dụng GPU RTX 3060 và RTX 5090 cho các thực nghiệm chi tiết về cấu hình và optimizer sensitivity. Đây là môi trường có nhiều quyền kiểm soát hơn, cho phép ta test các cấu hình phức tạp và chạy nhiều thực nghiệm song song.

Việc sử dụng cả hai môi trường giúp ta có được dữ liệu đa dạng từ các GPU khác nhau, từ GPU miễn phí trên cloud đến GPU cao cấp trong phòng lab, đảm bảo tính đại diện và khả năng áp dụng rộng rãi của kết quả.

**Phương pháp thực nghiệm**: Ta đã thiết kế 3 nhóm thực nghiệm chính:

1. **Baseline experiments** (16 thực nghiệm): Train mô hình trên các GPU khác nhau với cấu hình mặc định để chứng minh vấn đề reproducibility. Mục tiêu là xác nhận rằng vấn đề thực sự tồn tại và đo lường mức độ ảnh hưởng.

2. **Deterministic config tests** (31 thực nghiệm): Test các cấu hình khác nhau (TF32, AMP, num_workers, cuDNN benchmark, deterministic algorithms) để tìm nguyên nhân và giải pháp. Mỗi cấu hình được test trên nhiều GPU để đảm bảo tính khách quan.

3. **Optimizer sensitivity tests** (18 thực nghiệm): So sánh các optimizer khác nhau (Adam, RMSProp, SGD) để hiểu độ nhạy cảm với floating-point differences. Mục tiêu là xác định optimizer nào dễ bị ảnh hưởng bởi khác biệt số học.

Tổng cộng **65 thực nghiệm** được thực hiện, mỗi thực nghiệm được lưu kết quả chi tiết (loss, accuracy theo epoch) để phân tích sau.

**Thiết lập chi tiết**:

Để đảm bảo tính khách quan, ta đã thiết lập thực nghiệm như sau:

- **Model**: SimpleCNN với BatchNorm, Dropout. Đây là mô hình đơn giản nhưng đủ để chứng minh vấn đề reproducibility mà không cần quá nhiều tài nguyên tính toán.

- **Dataset**: CIFAR-10 (50,000 training, 10,000 test). Dataset này phổ biến, dễ tải về, và đủ phức tạp để thể hiện các khác biệt reproducibility.

- **GPUs/TPUs**: T4, L4, A100, RTX 3060, RTX 5090, TPU v6e-1. Đây là các GPU/TPU phổ biến, đại diện cho nhiều thế hệ và kiến trúc khác nhau.

- **Cấu hình test**: TF32 (on/off), AMP (on/off), num_workers (0/4), cuDNN benchmark (on/off), deterministic algorithms (on/off). Mỗi cấu hình được test độc lập để xác định ảnh hưởng riêng lẻ.

- **Optimizers**: Adam, RMSProp, SGD. Ba optimizer phổ biến nhất trong deep learning, đại diện cho các loại khác nhau (adaptive vs non-adaptive).

- **Metrics**: Train/Test Loss, Accuracy theo epoch. Các metrics này giúp ta đánh giá cả quá trình training và kết quả cuối cùng.

Tất cả thực nghiệm đều sử dụng cùng random seed (42) và cùng hyperparameters để đảm bảo tính công bằng trong so sánh. Điều này có nghĩa là mọi khác biệt trong kết quả đều xuất phát từ khác biệt phần cứng hoặc cấu hình, không phải từ randomness hay hyperparameters.

 Vậy nguyên nhân là gì? Tại sao ngay cả với cùng cấu hình, kết quả vẫn khác nhau? Hãy cùng điều tra từng giả thuyết một cách có hệ thống.

---

## ACT 2: Điều tra Nguyên nhân (The Investigation)

### 2.1 Giả thuyết 1: Cấu hình Mặc định Khác nhau

Giả thuyết đầu tiên: Các GPU có cấu hình mặc định khác nhau, đặc biệt là về precision (độ chính xác số học). Ta đã test giả thuyết này bằng cách so sánh kết quả với TF32 bật và tắt.

📖 **TF32 (TensorFloat-32)**: Định dạng 19-bit do NVIDIA phát triển cho Tensor Cores. Sử dụng cùng exponent với FP32 (8 bit) nhưng chỉ 10 bit mantissa. Tự động kích hoạt trên GPU Ampere+ (A100, RTX 3060, RTX 5090). Tăng tốc đáng kể với độ chính xác gần FP32.

💡 **Fun fact**: TF32 được NVIDIA giới thiệu vào năm 2020 cùng với kiến trúc Ampere, đánh dấu một bước tiến quan trọng trong việc tối ưu hóa precision cho deep learning. Điều thú vị là TF32 không phải là một định dạng floating-point chuẩn IEEE 754, mà là một định dạng "tùy chỉnh" được thiết kế đặc biệt cho Tensor Cores - các đơn vị xử lý chuyên biệt được NVIDIA giới thiệu từ kiến trúc Volta (2017) để tăng tốc các phép toán ma trận, nền tảng của deep learning.

**Kết quả**: TF32 ảnh hưởng khác nhau trên GPU khác nhau:

- **GPU Ampere+** (A100, RTX 3060, RTX 5090): TF32 mặc định bật, có thể ảnh hưởng đến precision. Khi tắt TF32, accuracy có thể thay đổi nhẹ, cho thấy TF32 đóng vai trò trong việc tạo ra khác biệt.

- **GPU Turing** (T4): Không hỗ trợ TF32. Điều này có nghĩa là T4 luôn sử dụng FP32 hoặc FP16, không có tùy chọn TF32.

- **GPU Ada** (L4): Hỗ trợ TF32 nhưng ảnh hưởng khác so với Ampere+. Có thể do kiến trúc khác nhau hoặc implementation khác nhau.

- **TPU**: Không hỗ trợ TF32 (đây là tính năng NVIDIA GPU). TPU sử dụng BF16 hoặc FP32, không có TF32.

💡 **Fun fact**: TPU (Tensor Processing Unit) được Google phát triển từ năm 2015 (starting from the original TPU architecture [ISCA'17]) và đã được sử dụng trong trận đấu cờ vây nổi tiếng giữa AlphaGo và kỳ thủ Lee Sedol vào năm 2016. Trong bài viết này, ta sử dụng TPU v6e-1, một thế hệ hiện đại hơn nhiều so với TPU v1 được mô tả trong paper ISCA 2017. Từ thế hệ đầu tiên với hiệu suất 23 TFLOPs, đến năm 2024, thế hệ TPU Trillium đã đạt hiệu suất cao hơn 4,7 lần so với thế hệ trước và tiết kiệm năng lượng hơn 67%. Điều thú vị là TPU được thiết kế chuyên biệt cho AI, nên có thể tiết kiệm năng lượng gấp 80 lần so với GPU trong một số trường hợp.

<div align="center">

![Config Impact Comparison](https://i.ibb.co/ZpZTCYMx/Config-Impact-Comparison.png)

**Hình 3:** Ảnh hưởng của các cấu hình (TF32, AMP, num_workers, deterministic algorithms) đến accuracy.

</div>

Quan sát Hình 3, ta có thể thấy rằng TF32 có ảnh hưởng khác nhau trên các GPU khác nhau. Trên GPU Ampere+, việc bật/tắt TF32 có thể thay đổi accuracy một chút, trong khi trên GPU không hỗ trợ TF32, không có sự khác biệt. Điều này xác nhận rằng TF32 là một trong những nguyên nhân gây ra khác biệt giữa các GPU.

 **Kết luận**: TF32 mặc định bật trên GPU Ampere+ → khác biệt precision so với GPU không hỗ trợ TF32. Đây là một trong những nguyên nhân chính gây ra sự khác biệt giữa các GPU.

### 2.2 Giả thuyết 2: DataLoader Non-deterministic

Giả thuyết thứ hai: DataLoader với nhiều workers có thể gây ra non-deterministic data loading. Ta đã test bằng cách so sánh num_workers = 0 vs 4.

**Kết quả**: 

- **num_workers = 0**: Deterministic data loading, reproducibility tốt hơn. Với num_workers=0, DataLoader chỉ sử dụng main process để load data, không có multi-threading, nên thứ tự load data là hoàn toàn deterministic.

- **num_workers = 4**: Non-deterministic data loading do multi-threading, có thể gây khác biệt. Khi sử dụng nhiều workers, các thread có thể load data theo thứ tự khác nhau giữa các lần chạy, dẫn đến non-deterministic behavior.

**Kết luận**: Với baseline config (TF32=False, AMP=False, num_workers=0), max difference chỉ còn **0.87%** - một mức chấp nhận được. Điều này cho thấy num_workers=0 là một yếu tố quan trọng để đạt reproducibility tốt.

### 2.3 Giả thuyết 3: cuDNN Benchmark

Giả thuyết thứ ba: cuDNN benchmark tự động chọn kernel khác nhau giữa các lần chạy, gây ra non-deterministic behavior.

📖 **cuDNN Benchmark**: Tính năng tự động benchmark và chọn kernel tối ưu của cuDNN (CUDA Deep Neural Network library). Khi bật (`benchmark=True`), cuDNN sẽ test nhiều kernel khác nhau cho mỗi operation và chọn kernel nhanh nhất. Tuy nhiên, kernel được chọn có thể khác nhau giữa các lần chạy do timing variations, dẫn đến non-deterministic behavior. Khi tắt (`benchmark=False`), cuDNN sử dụng kernel mặc định cố định, đảm bảo tính deterministic nhưng có thể chậm hơn đáng kể (thường 5-20% tùy workload và GPU).

**Kết quả**: 

- **cuDNN benchmark = True**: Tự động chọn kernel tối ưu → non-deterministic. cuDNN sẽ benchmark các kernel khác nhau và chọn kernel nhanh nhất, nhưng kernel được chọn có thể khác nhau giữa các lần chạy.

- **cuDNN benchmark = False**: Sử dụng kernel cố định → deterministic hơn. Khi tắt benchmark, cuDNN sẽ sử dụng kernel mặc định, đảm bảo tính deterministic.

**Kết luận**: Tắt cuDNN benchmark giúp deterministic hơn, nhưng có thể làm giảm performance một chút (khoảng 5-10%). Đây là trade-off giữa reproducibility và performance.

### 2.4 Giả thuyết 4: AMP (Automatic Mixed Precision)

Giả thuyết thứ tư: AMP (Automatic Mixed Precision) có thể ảnh hưởng đến reproducibility do khác biệt số học giữa FP32 và FP16.

📖 **AMP (Automatic Mixed Precision)**: Kỹ thuật tự động chuyển đổi giữa FP32 và FP16. Sử dụng FP16 cho forward pass (nhanh hơn), FP32 cho backward pass (chính xác hơn). Giúp tăng tốc training 1.5-2x và giảm bộ nhớ. Có thể ảnh hưởng đến reproducibility do khác biệt số học.

💡 **Fun fact**: Trước năm 2012, việc huấn luyện mạng nơ-ron chủ yếu được thực hiện trên CPU, dẫn đến thời gian huấn luyện kéo dài hàng tuần hoặc hàng tháng. Sự ra đời của AlexNet vào năm 2012, được huấn luyện trên hai GPU GeForce GTX 580, đã đánh dấu bước ngoặt trong việc sử dụng GPU cho deep learning. Từ đó, GPU trở thành công cụ không thể thiếu trong AI, và NVIDIA đã trở thành công ty bán dẫn đầu tiên đạt giá trị thị trường trên 1,2 nghìn tỷ USD vào năm 2023.

**Kết quả**: 

- **AMP = True**: Có thể ảnh hưởng đến accuracy và reproducibility. Sử dụng FP16 trong forward pass có thể tạo ra khác biệt số học nhỏ, tích lũy qua các epochs.

- **AMP = False**: Reproducibility tốt hơn, nhưng correlation vẫn cao (0.985) ngay cả khi bật AMP. Điều này cho thấy AMP không phải là nguyên nhân chính, nhưng vẫn có ảnh hưởng.

**Kết luận**: AMP có thể ảnh hưởng nhưng không phải là nguyên nhân chính. Correlation giữa các GPU vẫn rất cao (0.985) ngay cả khi sử dụng AMP, cho thấy xu hướng học tập vẫn tương đồng.

### 2.5 Tổng hợp: Heatmap Ảnh hưởng

Để có cái nhìn tổng quan, ta đã tạo heatmap so sánh accuracy theo GPU và config:

<div align="center">

![Config Impact Heatmap](https://i.ibb.co/x8B3wSHy/Config-Impact-Heatmap.png)

**Hình 4:** Heatmap accuracy theo GPU và config - cấu hình baseline (TF32=False, AMP=False, num_workers=0) cho reproducibility tốt nhất.

</div>

Quan sát Hình 4, ta có thể thấy rõ sự khác biệt accuracy giữa các GPU và config. Màu xanh lá (accuracy cao) và màu đỏ (accuracy thấp) được phân bố khác nhau tùy theo config. Điều này cho thấy config có ảnh hưởng đáng kể đến kết quả.

**Key insight**: Config baseline (TF32=False, AMP=False, num_workers=0) → reproducibility tốt nhất với max diff chỉ 0.87% và correlation 0.985. Đây là cấu hình khuyến nghị khi cần reproducibility cao.

 Nhưng tại sao ngay cả với config giống nhau, vẫn có khác biệt? Hãy xem sâu hơn vào cơ chế số học và cách optimizer xử lý sai số.

---

## ACT 3: Hiểu sâu Vấn đề (The Deep Understanding)

### 3.1 Optimizer Sensitivity - Khuếch đại Sai số

Một phát hiện quan trọng: Các optimizer khác nhau có độ nhạy cảm khác nhau với floating-point differences. Ta đã test với 3 optimizer phổ biến: Adam, RMSProp, và SGD.

**Kết quả so sánh** (mean accuracy across all GPUs):

| Optimizer | Mean Accuracy | Độ nhạy cảm |
|-----------|---------------|-------------|
| Adam | 76.60% | Trung bình |
| RMSprop | 72.68% | Cao |
| SGD | 69.09% | Thấp |

So sánh này nhằm đánh giá độ nhạy với sai số số học, không nhằm khẳng định hiệu năng tối ưu của từng optimizer khi được tune đầy đủ.

<div align="center">

![Optimizer Sensitivity](https://i.ibb.co/vvQ6F0zd/Optimizer-Sensitivity.png)

**Hình 5:** So sánh accuracy theo optimizer cho mỗi GPU - Adam tốt nhất, RMSprop và SGD thấp hơn đáng kể.

</div>

Quan sát Hình 5, ta có thể thấy rằng Adam cho kết quả tốt nhất trên tất cả các GPU, với accuracy trung bình 76.60%. RMSprop và SGD thấp hơn đáng kể, nhưng pattern tương tự nhau trên các GPU khác nhau. Điều này cho thấy optimizer có ảnh hưởng đến accuracy, nhưng không làm thay đổi pattern reproducibility giữa các GPU.

**Cơ chế**: Adaptive optimizers (Adam, RMSProp) sử dụng momentum và adaptive learning rate, có thể **khuếch đại sai số nhỏ** từ floating-point precision thành khác biệt lớn hơn qua các epochs. Điều này giống như hiệu ứng "butterfly effect" - một sai số nhỏ ban đầu có thể dẫn đến khác biệt lớn sau nhiều epochs.

<div align="center">

![Adam Sensitivity Test - RTX 5090](https://i.ibb.co/B2hxxQPH/Adam-Sensitivity-Test-RTX-5090.png)

**Hình 6:** Training trajectory của Adam trên RTX 5090 - minh họa cách adaptive optimizer xử lý và tích lũy sai số qua các epochs.

</div>

Quan sát Hình 6, ta có thể thấy rằng mặc dù cùng sử dụng Adam optimizer và cùng seed, các lần chạy vẫn có thể cho kết quả hơi khác nhau do tích lũy sai số từ floating-point precision. Điều này minh họa rõ ràng cơ chế khuếch đại sai số của adaptive optimizers.

Cụ thể, khi có một sai số nhỏ trong gradient (do floating-point precision), adaptive optimizer sẽ:
1. Lưu trữ sai số này trong momentum buffer
2. Sử dụng sai số này để tính toán learning rate adaptive
3. Tích lũy sai số qua các epochs
4. Dẫn đến khác biệt lớn hơn ở cuối training

SGD ít nhạy cảm hơn vì không có adaptive learning rate, nhưng accuracy thấp hơn đáng kể trong thực nghiệm này. Điều này cho thấy trade-off giữa accuracy và reproducibility sensitivity.

### 3.2 Floating-Point Precision - Nguồn gốc Sai số

Để hiểu tại sao có khác biệt, ta cần hiểu về floating-point precision - nguồn gốc của mọi sai số.

**Khác biệt precision trên GPU khác nhau**:

- **GPU khác nhau hỗ trợ precision khác nhau**: T4 (**FP32**, **FP16**), A100 (**FP32**, **FP16**, **BF16**, TF32), RTX 5090 (**FP32**, **FP16**, **BF16**, TF32), TPU (**FP32**, **BF16**)

📖 **FP32 (Float32)**: Định dạng số thực 32-bit, độ chính xác chuẩn. Sử dụng 1 bit dấu + 8 bit exponent + 23 bit mantissa, phạm vi ~±3.4×10³⁸, độ chính xác ~7 chữ số thập phân. Được sử dụng mặc định trong hầu hết các phép tính deep learning.

💡 **Fun fact**: FP32 tuân theo chuẩn IEEE 754, được phát triển vào năm 1985 và trở thành tiêu chuẩn quốc tế cho floating-point arithmetic. Điều thú vị là chuẩn này được thiết kế để giải quyết vấn đề "floating-point crisis" - khi các máy tính khác nhau cho kết quả khác nhau cho cùng một phép tính. Tuy nhiên, ngay cả với chuẩn IEEE 754, vẫn có khác biệt do cách implementation trên phần cứng khác nhau - đây chính là nguồn gốc của vấn đề reproducibility mà ta đang nghiên cứu!

📖 **FP16 (Float16 / Half Precision)**: Định dạng số thực 16-bit, độ chính xác thấp hơn. Sử dụng 1 bit dấu + 5 bit exponent + 10 bit mantissa, phạm vi ~±65,504, độ chính xác ~3 chữ số thập phân. Giúp tăng tốc và giảm bộ nhớ, nhưng có thể mất độ chính xác.

📖 **BF16 (Brain Float16)**: Định dạng 16-bit do Google phát triển. Sử dụng 1 bit dấu + 8 bit exponent (giống FP32) + 7 bit mantissa. Phạm vi giống FP32 nhưng độ chính xác thấp hơn. Được sử dụng trên TPU và một số GPU hiện đại.

- **Thứ tự tính toán**: Parallel computation → non-deterministic order → khác biệt nhỏ trong kết quả. Khi GPU thực hiện nhiều phép tính song song, thứ tự hoàn thành có thể khác nhau giữa các lần chạy, dẫn đến khác biệt nhỏ trong kết quả.

- **Tích lũy**: Sai lệch nhỏ (ví dụ: 0.0001) → tích lũy qua epochs → khác biệt lớn (ví dụ: 0.87%). Mỗi epoch, sai lệch nhỏ này được tích lũy, và sau 10 epochs, nó có thể trở thành khác biệt đáng kể.

Đây là lý do tại sao ngay cả với cùng config, kết quả vẫn khác nhau - do bản chất của floating-point arithmetic và parallel computation. Đây không phải là bug, mà là đặc tính tự nhiên của tính toán số học trên phần cứng song song.

### 3.3 Statistical vs Bit-exact Reproducibility

Một khái niệm quan trọng cần phân biệt: **Statistical reproducibility** vs **Bit-exact reproducibility**.

**Định nghĩa**:

- **Bit-exact**: Kết quả hoàn toàn giống nhau (bit-by-bit) - **rất khó đạt** trên GPU khác nhau. Để đạt được bit-exact reproducibility, ta cần cùng một GPU, cùng một driver, cùng một version của cuDNN, và cùng một cấu hình. Ngay cả khi đó, vẫn có thể có khác biệt do non-deterministic operations.

- **Statistical**: Cùng xu hướng, cùng cấp độ performance - **mục tiêu thực tế**. Statistical reproducibility có nghĩa là mô hình học theo cùng một cách, chỉ khác nhau về giá trị tuyệt đối. Đây là mục tiêu thực tế mà ta có thể đạt được.

**Phân tích kết quả thực nghiệm**:

ℹ️ **Ghi chú về Notebooks**: Các thực nghiệm với baseline config (TF32=False, AMP=False, num_workers=0) được thực hiện trong các notebook sau:
- `02a_test_T4.ipynb` - Thực nghiệm trên Tesla T4
- `02b_test_L4.ipynb` - Thực nghiệm trên NVIDIA L4
- `02c_test_A100.ipynb` - Thực nghiệm trên NVIDIA A100-SXM4-40GB
- `02d_test_RTX3060.ipynb` - Thực nghiệm trên NVIDIA GeForce RTX 3060
- `02e_test_TPU_v6e1.ipynb` - Thực nghiệm trên TPU v6e-1
- `02f_test_RTX5090.ipynb` - Thực nghiệm trên NVIDIA GeForce RTX 5090



Với baseline config (TF32=False, AMP=False, num_workers=0):
- **Max diff**: 0.87% (giữa TPU 78.45% và T4 77.64%)
- **Mean correlation**: 0.985 (rất cao)
- **Category**: **Statistical (Chấp nhận)**

Quan sát các số liệu này, ta có thể thấy rằng mặc dù có khác biệt 0.87% về accuracy cuối cùng, nhưng correlation 0.985 cho thấy **xu hướng học tập gần như hoàn toàn giống nhau** giữa các GPU. Điều này có nghĩa là mô hình học theo cùng một cách, chỉ khác nhau về giá trị tuyệt đối do khác biệt số học - đây chính là **statistical reproducibility** mà ta mong muốn đạt được.

**Training trajectory analysis**:
- **Correlation giữa các GPU**: 0.978 - 0.999 (rất cao). Correlation được tính bằng Pearson correlation trên test accuracy theo epoch. Correlation này được tính bằng cách so sánh training curves giữa các GPU. Giá trị gần 1.0 cho thấy các curves gần như hoàn toàn giống nhau về hình dạng.

- **Convergence speed**: Tất cả GPU đạt 70% accuracy ở epoch 3-4 (tương đồng). Điều này cho thấy tốc độ học tập tương tự nhau trên tất cả các GPU.

- **Stability**: Std của accuracy qua epochs khoảng 5.1-5.9% (tương đồng). Độ ổn định của training cũng tương tự nhau, cho thấy không có GPU nào có vấn đề về stability.

Điều này cho thấy mặc dù giá trị tuyệt đối khác nhau, nhưng **xu hướng học tập hoàn toàn tương đồng** giữa các GPU. Đây chính là điều ta mong muốn - mô hình học theo cùng một cách, bất chấp khác biệt phần cứng.

### 3.4 Khi nào Chấp nhận được vs Vấn đề Nghiêm trọng

Dựa trên kết quả thực nghiệm, ta có thể đưa ra các tiêu chí cụ thể:

**Chấp nhận được** (Statistical Reproducibility):

- ✅ **Test accuracy lệch < 0.5%**: TỐT - như trong trường hợp RTX 3060 vs RTX 5090 (0.11%). Đây là mức reproducibility rất tốt, gần như không có khác biệt đáng kể.

- ⚠️ **Test accuracy lệch 0.5-2%**: CHẤP NHẬN ĐƯỢC (nếu correlation > 0.95) - như trong trường hợp TPU vs T4 (0.87%). Đây là mức reproducibility chấp nhận được, miễn là correlation cao.

- ✅ **Correlation > 0.95**: Xu hướng học tương đồng - tất cả GPU trong thực nghiệm đều đạt 0.978-0.999. Đây là dấu hiệu quan trọng nhất - nếu correlation cao, ta có thể yên tâm rằng mô hình học theo cùng một cách.

**Vấn đề nghiêm trọng**:

- ❌ **Loss lệch lớn ngay từ epoch đầu**: Dấu hiệu của khác biệt cấu hình hoặc bug. Nếu loss khác nhau ngay từ đầu, có thể có vấn đề về cấu hình hoặc implementation.

- ❌ **Quỹ đạo học khác hẳn** (overfit vs underfit): Mô hình học theo cách khác nhau, không phải chỉ khác giá trị. Nếu một GPU overfit trong khi GPU khác underfit, đây là dấu hiệu của vấn đề nghiêm trọng.

- ❌ **Correlation < 0.95**: Xu hướng học không tương đồng - cần kiểm tra lại cấu hình. Nếu correlation thấp, có nghĩa là mô hình học theo cách khác nhau, không chỉ khác giá trị.

Trong thực nghiệm của ta, tất cả GPU đều nằm trong vùng "chấp nhận được" với correlation 0.985, cho thấy reproducibility tốt mặc dù có khác biệt nhỏ về giá trị tuyệt đối.

 Vậy làm thế nào để đạt được reproducibility tốt nhất? Hãy cùng xem các giải pháp thực tế.

---

## ACT 4: Giải pháp Thực tế (The Solution)

### 4.1 Cấu hình Deterministic - Checklist Thực tế

Dựa trên kết quả thực nghiệm, đây là checklist cấu hình deterministic để đạt reproducibility tốt nhất:

**Code example** (PyTorch):

```python
import torch
import numpy as np

# 1. Set seed cho tất cả RNG
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 2. Tắt TF32
torch.backends.cudnn.allow_tf32 = False
if hasattr(torch.backends, 'cublas'):
    torch.backends.cublas.allow_tf32 = False

# 3. Tắt cuDNN benchmark
torch.backends.cudnn.benchmark = False

# 4. Deterministic algorithms
torch.backends.cudnn.deterministic = True
torch.use_deterministic_algorithms(True, warn_only=True)

# 5. DataLoader - num_workers = 0
train_loader = DataLoader(
    dataset, 
    batch_size=128, 
    shuffle=True, 
    num_workers=0  # Quan trọng!
)
```

Hãy giải thích từng bước một cách chi tiết:

**Bước 1**: Set seed cho tất cả RNG (Random Number Generator). Điều này đảm bảo rằng tất cả các số ngẫu nhiên (cho weight initialization, data shuffling, dropout) đều giống nhau giữa các lần chạy.

**Bước 2**: Tắt TF32. Điều này đảm bảo tất cả GPU sử dụng cùng precision (FP32), loại bỏ khác biệt do TF32.

**Bước 3**: Tắt cuDNN benchmark. Điều này đảm bảo cuDNN sử dụng cùng kernel, không tự động chọn kernel khác nhau.

**Bước 4**: Bật deterministic algorithms. Điều này buộc PyTorch sử dụng các thuật toán deterministic, mặc dù có thể chậm hơn một chút.

**Bước 5**: num_workers = 0. Điều này loại bỏ non-deterministic data loading do multi-threading.

**Kết quả**: Với cấu hình này, max diff chỉ còn **0.87%** và correlation đạt **0.985** - một mức reproducibility rất tốt cho cross-GPU training.

### 4.2 Chuẩn hóa Môi trường

Ngoài cấu hình deterministic, ta cần chuẩn hóa môi trường training:

- **Cùng batch size**: Đảm bảo tất cả GPU sử dụng cùng batch size. Batch size khác nhau có thể dẫn đến khác biệt trong gradient, ảnh hưởng đến training.

- **Cùng preprocessing**: Normalization, augmentation phải giống nhau. Nếu preprocessing khác nhau, dữ liệu đầu vào sẽ khác nhau, dẫn đến kết quả khác nhau.

- **Cùng precision**: FP32, FP16, hoặc BF16 - chọn một và giữ nguyên. Không nên mix các precision khác nhau giữa các GPU.

- **Cùng seed**: Sử dụng cùng random seed cho numpy, torch, và CUDA. Điều này đảm bảo tất cả randomness đều giống nhau.

### 4.3 Best Practices cho Nghiên cứu

Dựa trên findings từ thực nghiệm, đây là các best practices:

1. **So sánh xu hướng (trend), không so giá trị tuyệt đối**: 
   - Quan trọng hơn là xem training trajectory có tương đồng không
   - Correlation > 0.95 là dấu hiệu tốt
   - Không nên lo lắng về khác biệt 0.5-2% nếu correlation cao

2. **Ghi rõ GPU + precision trong báo cáo thực nghiệm**:
   - Ví dụ: "Trained on NVIDIA A100-SXM4-40GB, FP32, TF32=False"
   - Giúp người đọc hiểu context và có thể reproduce
   - Cho phép người đọc đánh giá kết quả trong context phù hợp

3. **Không debug chỉ dựa trên accuracy**:
   - Cần xem training trajectory, loss curves, và correlation
   - Accuracy cuối cùng có thể khác nhau nhưng xu hướng học vẫn đúng
   - Nếu correlation cao, ta có thể yên tâm rằng mô hình học đúng

4. **Sử dụng deterministic config khi cần reproducibility cao**:
   - Cho paper submission, benchmark, hoặc production
   - Chấp nhận trade-off về performance (có thể chậm hơn 10-20%)
   - Đảm bảo kết quả có thể reproduce và verify

### 4.4 Trade-offs và Khi nào Dùng gì

**Performance vs Reproducibility**:

- **Deterministic config** (TF32=False, num_workers=0, cuDNN benchmark=False):
  - ✅ Reproducibility tốt (max diff 0.87%, correlation 0.985)
  - ❌ Có thể chậm hơn 10-20%
  - 📌 Phù hợp: Research, paper submission, benchmark

- **Performance config** (TF32=True, num_workers>0, cuDNN benchmark=True):
  - ✅ Nhanh hơn 10-20%
  - ❌ Reproducibility kém hơn (max diff có thể > 2%)
  - 📌 Phù hợp: Development, experimentation, khi không cần reproducibility cao

- **Cân bằng** (TF32=False, num_workers=0, cuDNN benchmark=True):
  - ✅ Reproducibility tốt (gần deterministic)
  - ✅ Performance chấp nhận được (có thể chậm hơn đáng kể, thường 5-20% tùy workload và GPU)
  - 📌 Phù hợp: Hầu hết trường hợp nghiên cứu

**Khi nào cần bit-exact**: Rất hiếm, chỉ khi debug nghiêm trọng hoặc cần verify implementation. Trong hầu hết trường hợp, statistical reproducibility là đủ.

**Khi nào statistical đủ**: Hầu hết trường hợp nghiên cứu và production - mục tiêu là đảm bảo mô hình học theo cùng một cách, không phải cùng một con số. Reproducibility trong deep learning hiện đại nên được đánh giá ở mức learning dynamics, thay vì yêu cầu bit-exact numerical equality.

---

## KẾT LUẬN: Key Takeaways

### Tóm tắt

Qua 65 thực nghiệm trên 6 GPU/TPU khác nhau, ta có thể rút ra những kết luận quan trọng:

1. **Cross-GPU reproducibility là statistical, không phải bit-exact**: Mục tiêu thực tế là đảm bảo cùng xu hướng học tập (correlation > 0.95), không phải cùng giá trị tuyệt đối.

2. **So sánh xu hướng (trend) quan trọng hơn giá trị tuyệt đối**: Với baseline config, max diff 0.87% là chấp nhận được khi correlation đạt 0.985.

3. **Deterministic config giúp reproducibility tốt hơn đáng kể**: Tắt TF32, num_workers=0, và cuDNN benchmark=False giúp giảm max diff và tăng correlation.

4. **Optimizer sensitivity**: Adaptive optimizers (Adam, RMSProp) khuếch đại sai số nhỏ, nhưng vẫn cho kết quả tốt với correlation cao.

5. **Xác định cấu hình hardware trước khi bắt đầu**: Đây là bước quan trọng nhất và thường bị bỏ qua. Trước khi tiến hành bất kỳ công việc nào yêu cầu reproducibility cao (nghiên cứu, paper submission, production deployment), ta **phải xác định rõ ràng**:
   - Loại GPU/TPU cụ thể (T4, A100, RTX 5090, TPU v6e-1, ...)
   - Cấu hình precision (FP32, FP16, BF16, TF32 on/off)
   - Các thiết lập deterministic (num_workers, cuDNN benchmark, deterministic algorithms)
   - Version của PyTorch, CUDA, cuDNN
   
   Việc xác định cấu hình hardware từ đầu giúp:
   - Tránh mất thời gian debug khi kết quả khác nhau giữa các môi trường
   - Đảm bảo reproducibility ngay từ đầu thay vì phải điều chỉnh sau
   - Cho phép lựa chọn cấu hình phù hợp với yêu cầu (reproducibility vs performance)
   - Tạo baseline rõ ràng để so sánh và verify kết quả

### Hướng dẫn Nhanh (Quick Reference)

**Để reproducibility tốt nhất**:
- Tắt TF32: `torch.backends.cudnn.allow_tf32 = False`
- Tắt cuDNN benchmark: `torch.backends.cudnn.benchmark = False`
- num_workers = 0 trong DataLoader
- Deterministic algorithms: `torch.use_deterministic_algorithms(True, warn_only=True)`
- **Kết quả**: Max diff 0.87%, correlation 0.985

**Để performance tốt nhất**:
- Bật TF32 (nếu GPU hỗ trợ)
- Bật cuDNN benchmark
- num_workers > 0
- **Trade-off**: Reproducibility kém hơn

**Cân bằng** (khuyến nghị cho hầu hết trường hợp):
- Tắt TF32
- num_workers = 0
- Bật cuDNN benchmark (nếu cần performance)
- **Kết quả**: Reproducibility tốt, performance chấp nhận được

### Hướng mở rộng

Bài viết này tập trung vào mô hình nhỏ (SimpleCNN) trên dataset CIFAR-10. Các hướng mở rộng có thể bao gồm:

- **Mô hình lớn hơn**: ResNet, Transformer - liệu reproducibility có thay đổi?
- **Dataset khác**: ImageNet, NLP datasets - các findings có áp dụng được không?
- **Phân tích sâu hơn**: Impact của batch size, learning rate schedule, và các hyperparameters khác

---

## Tài liệu Tham khảo

1. PyTorch Team. (2025). *PyTorch Documentation - Reproducibility*. PyTorch Documentation. https://pytorch.org/docs/stable/notes/randomness.html

2. NVIDIA Corporation. (2025). *NVIDIA GPU Documentation - TF32, Mixed Precision Training*. NVIDIA Developer Documentation. https://blogs.nvidia.com/blog/tensorfloat-32-precision-format/

3. PyTorch Team. (2025). *PyTorch Documentation - TF32 on Ampere and Later Devices*. PyTorch Documentation. https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere

4. Google Research. (2017). *In-Datacenter Performance Analysis of a Tensor Processing Unit*. ISCA 2017. https://arxiv.org/abs/1704.04760

5. PyTorch Team. (2025). *PyTorch Documentation - CUDA Determinism*. PyTorch Documentation. https://pytorch.org/docs/stable/notes/cuda.html#reproducibility

6. NVIDIA Corporation. (2025). *cuDNN Documentation - Deterministic Algorithms*. NVIDIA Developer Documentation. https://docs.nvidia.com/deeplearning/cudnn/

---

ℹ️ **Lưu ý**: Tất cả số liệu và visualizations trong bài đều dựa trên 65 thực nghiệm thực tế. Code và data có sẵn trong repository để người đọc có thể reproduce và verify.

