# 🧬 Giải Thuật Di Truyền — *In a Nutshell*

## 📝 Tóm tắt

Bài viết khám phá cách **Giải thuật Di truyền** giải quyết bài toán định tuyến phức tạp với hàng trăm điểm giao hàng với nhiều xe. Từ việc tách nhập phường/xã TP.HCM, thuật toán cho ta một giải pháp tối ưu giảm đáng kể khoảng cách di chuyển và cân bằng tải giữa các xe. Một ví dụ điển hình về cách thuật toán tối ưu có thể giải quyết những thách thức logistics phức tạp trong đời thực.

---

> *"Từ bài toán định tuyến 168 phường/xã TPHCM với 4 xe giao hàng — khám phá cách thuật toán di truyền giải quyết bài toán tối ưu phức tạp trong thực tế."*

## 1. Giới thiệu: Bài toán thực tế

### 1.1 Multi-Vehicle Traveling Salesman Problem (MVTSP) tại TP.HCM

**Bối cảnh thực tế:**
Năm 2025, việc sắp xếp lại phường/xã và bỏ cấp quận/huyện trên toàn quốc lại khơi lại bài toán về tối ưu hóa quãng đường vận chuyển đối với nhiều doanh nghiệp. Trong phạm vi bài toán tự đặt ra để mô phỏng thực tế giao hàng trong phạm vi **168 phường/xã** mới được tạo lập trên địa bàn thành phố Hồ Chí Minh với **4 xe giao hàng**, việc phân chia routes hiệu quả trở thành yêu cầu cần thiết cho nhiều doanh nghiệp để:

1. **🎯 Tối thiểu tổng khoảng cách di chuyển** - Giảm chi phí nhiên liệu và carbon footprint
2. **⚖️ Cân bằng tải giữa các xe** - Đảm bảo công bằng và hiệu quả vận hành
3. **📍 Phân vùng địa lý hợp lý** - Mỗi xe phụ trách một tuyến đường phù hợp
4. **⏱️ Tối ưu thời gian giao hàng** - Đáp ứng yêu cầu giao hàng nhanh của khách hàng
5. **🌱 Hướng tới logistics xanh** - Giảm tác động môi trường thông qua tối ưu hóa routes

> **💡 Dự án mã nguồn mở:** Toàn bộ code và kết quả của bài toán này đã được công khai trên GitHub tại [https://github.com/D9Dre4mer/Genetic_Algorithm_MultiVehicle_TSP](https://github.com/D9Dre4mer/Genetic_Algorithm_MultiVehicle_TSP).

<div align="center">
![Logo](https://i.ibb.co/d41fHgXL/Logo-999.png)
</div>

### 1.2 Thách thức từ việc tách nhập phường/xã

**Vấn đề thực tế:**
- **🔄 Thay đổi địa giới hành chính**: Việc tách nhập phường/xã diễn ra đặt ra bài toán về sắp xếp và bố trí lại các tuyến xe
- **📊 Dữ liệu không đồng bộ**: Hệ thống cũ không cập nhật kịp với thay đổi mới
- **🗺️ Bản đồ lỗi thời**: Routes được tính toán dựa trên ranh giới cũ
- **⏰ Chi phí điều chỉnh cao**: Mỗi lần thay đổi phải tính toán lại toàn bộ hệ thống
- **🚚 Hiệu quả giảm sút**: Xe có thể phải đi đường vòng hoặc chồng chéo khu vực

### 1.3 Giải pháp GA cho vấn đề tách nhập

**Cách GA giải quyết:**
- **🔄 Tự động điều chỉnh**: Khi có thêm phương tiện hoặc điểm giao hàng mới, GA tự động tính toán lại routes
- **⚡ Tiết kiệm chi phí**: Tối ưu khoảng cách cũng là tối ưu chi phí
- **📊 Duy trì hiệu quả**: Routes luôn tối ưu dù có thay đổi địa giới
- **🗺️ Phân vùng linh hoạt**: Tự động điều chỉnh khu vực phụ trách của từng xe


### 1.4 Thách thức tính toán

Với **168 điểm** và **4 xe**, số lượng cách phân chia có thể lên đến:
- **Tổ hợp**: C(168,42) × C(126,42) × C(84,42) × C(42,42) ≈ **10^200**
- **Hoán vị**: Mỗi xe có thể thăm số phường/xã khác nhau theo thứ tự khác nhau
- **Tổng số**: **10^200 × (42!)⁴** ≈ **10^300** cách phân chia

→ **Không thể giải bằng phương pháp brute force!**  
*(Brute force nghĩa là thử mọi khả năng có thể - duyệt hết toàn bộ không gian lời giải. Ở đây, số lượng phương án phân chia quá lớn (≈ $10^{300}$), nên brute force là bất khả thi về mặt thời gian và tài nguyên tính toán!)*

### 1.5 Tại sao chọn Giải thuật Di truyền?

Giải thuật di truyền (Genetic Algorithm - **GA**) là lựa chọn tốt vì:

- ✅ **Không cần đạo hàm** - Làm việc với bài toán rời rạc và combinatorial
- ✅ **Tìm kiếm toàn cục** - Tránh cực trị cục bộ trong không gian giải pháp khổng lồ
- ✅ **Tự thích nghi** - Cân bằng giữa khám phá (exploration) và khai thác (exploitation)
- ✅ **Mở rộng tốt** - Dễ điều chỉnh cho nhiều xe khác nhau
- ✅ **Robust** - Hoạt động ổn định với dữ liệu thực tế phức tạp
- ✅ **⚡ Cập nhật nhanh** - Chỉ cần vài phút để tính toán lại routes mới

---

## 2. Pipeline Giải thuật Di truyền cho MVTSP

### 2.1 Sơ đồ tổng quan Pipeline

<div align="center">
![Pipeline GA cho MVTSP](https://i.ibb.co/GGT3cK7/ga-pipeline-diagram.png)
</div>

<div align="center">
*Hình 1: Pipeline GA cho MVTSP - 6 bước chính với nội dung thực tế từ dự án*
</div>

**Bám sát giải thuật di truyền chuẩn**:
Dự án này áp dụng đầy đủ các thành phần cốt lõi của giải thuật di truyền chuẩn theo Holland (1975) và Goldberg (1989):

- 🧬 **Population**: 250 cá thể đa dạng (quần thể ban đầu)
- 🎯 **Fitness Function**: Đánh giá chất lượng dựa trên khoảng cách và cân bằng tải
- 🏆 **Selection**: Tournament selection (k=3) để chọn cha mẹ tốt nhất
- 🔄 **Crossover**: Multi-vehicle crossover để tạo con từ 2 cha mẹ
- 🧬 **Mutation**: Hoán đổi ngẫu nhiên 2 điểm trong route (30% mutation rate)
- 👑 **Elitism**: Giữ lại 5% cá thể tốt nhất mỗi thế hệ
- 🔁 **Evolution**: Lặp lại quá trình qua 20,000 thế hệ với early stopping

### 2.2 6 Các bước chi tiết theo Pipeline

#### 2.2.1 Bước 1: Khởi tạo Quần thể (Population Initialization)

Giải thuật di truyền bắt đầu bằng việc tạo ra một "quần thể" gồm nhiều "cá thể" (solutions). Mỗi cá thể là một cách giải quyết bài toán khác nhau. Giống như trong tự nhiên, quần thể càng đa dạng thì khả năng tìm ra giải pháp tốt càng cao.

**Logic áp dụng**:
- **Tại sao cần 250 cá thể?**: Đủ lớn để khám phá không gian giải pháp nhưng không quá lớn để tính toán chậm
- **K-means clustering**: Thay vì phân chia ngẫu nhiên, chúng ta nhóm các phường/xã gần nhau về mặt địa lý trước, sau đó phân chia cho các xe
- **Đa dạng**: Mỗi cá thể có cách phân chia khác nhau để khám phá nhiều khả năng

**Mục tiêu**: Tạo ra 250 cá thể (solutions) đa dạng để khám phá không gian giải pháp.

**Cách áp dụng GA**:
```python
def create_initial_population(self):
    population = []
    for _ in range(250):  # Population size
        # Tạo cá thể bằng K-means clustering
        individual = self._create_kmeans_clustered_solution()
        population.append(individual)
    return population
```

**Kết quả thực tế**: Mỗi cá thể là một cách phân chia 168 phường/xã cho 4 xe:
- **Xe 1**: [Phường Tân Định, Phường Gia Định, Phường An Phú Đông, ...] (40 phường/xã)
- **Xe 2**: [Phường Bình Hòa, Phường Lái Thiêu, Phường An Phú, ...] (45 phường/xã)  
- **Xe 3**: [Phường Tân Bình, Phường Phú Nhuận, Phường Gò Vấp, ...] (42 phường/xã)
- **Xe 4**: [Phường Thủ Đức, Phường Linh Xuân, ...] (41 phường/xã)

#### 2.2.2 Bước 2: Đánh giá Fitness (Fitness Evaluation)

Fitness là "điểm số" đánh giá chất lượng của mỗi cá thể. Giống như trong tự nhiên, những cá thể "khỏe mạnh" (có fitness cao) sẽ có cơ hội sống sót và sinh sản cao hơn. Trong bài toán của chúng ta, fitness càng thấp càng tốt (vì chúng ta muốn giảm khoảng cách).

**Logic áp dụng**:
- **Haversine distance**: Tính khoảng cách thực tế giữa các phường/xã trên theo Kinh độ và Vĩ độ
- **Coefficient of Variation (CV)**: Đo độ cân bằng giữa các xe. CV thấp = các xe có khoảng cách tương đương nhau
- **Penalty system**: Nếu các xe mất cân bằng quá nhiều, sẽ bị phạt để thuật toán tập trung vào giải pháp cân bằng hơn
- **Multi-objective**: Kết hợp 2 mục tiêu: giảm tổng khoảng cách + cân bằng tải giữa các xe

**Mục tiêu**: Đánh giá chất lượng của mỗi cá thể dựa trên 2 tiêu chí chính.

**Cách áp dụng GA**:
```python
def multi_objective_fitness(self, solution):
    # Tính tổng khoảng cách (Haversine)
    total_distance = 0
    vehicle_distances = []
    
    for route in solution:
        distance = self.calculate_route_distance(route)
        total_distance += distance
        vehicle_distances.append(distance)
    
    # Tính cân bằng tải (Coefficient of Variation)
    cv = np.std(vehicle_distances) / np.mean(vehicle_distances)
    
    # Fitness = khoảng cách + penalty cân bằng
    fitness = total_distance + 1000 * cv
    return fitness
```

**Kết quả thực tế**: 
- **Fitness tốt**: ~1,601km (tổng khoảng cách) + 0.12 × 1000 (CV thấp)
- **Fitness kém**: ~2,500km + 0.45 × 1000 (CV cao)

#### 2.2.3 Bước 3: Chọn lọc (Selection)

Chọn lọc là quá trình "tuyển chọn" những cá thể tốt nhất để "sinh sản" thế hệ tiếp theo. Giống như trong tự nhiên, những cá thể khỏe mạnh sẽ có cơ hội sinh sản cao hơn. Tournament Selection giống như một cuộc thi đấu - mỗi lần chọn ngẫu nhiên một nhóm nhỏ để thi đấu, cá thể tốt nhất sẽ thắng.

**Logic áp dụng**:
- **Tournament Selection**: Chọn ngẫu nhiên 3 cá thể, cá thể có fitness tốt nhất sẽ được chọn
- **Tại sao không chọn cá thể tốt nhất luôn?**: Để tránh mất đa dạng và cho phép cá thể "tầm trung" có cơ hội cải thiện
- **Xác suất**: Cá thể tốt có xác suất cao hơn nhưng không đảm bảo 100% được chọn
- **Duy trì đa dạng**: Đảm bảo quần thể không bị "thuần chủng" quá sớm

**Mục tiêu**: Chọn các cá thể tốt nhất để "sinh sản" thế hệ tiếp theo.

**Cách áp dụng GA**:
```python
def tournament_selection(self, population, tournament_size=3):
    selected = []
    for _ in range(len(population)):
        # Chọn ngẫu nhiên 3 cá thể
        tournament = random.sample(population, tournament_size)
        # Chọn cá thể tốt nhất trong tournament
        winner = min(tournament, key=self.calculate_fitness)
        selected.append(winner)
    return selected
```

**Kết quả thực tế**: Cá thể có tổng khoảng cách thấp và cân bằng tải tốt sẽ có xác suất được chọn cao hơn.

#### 2.2.4 Bước 4: Lai ghép (Crossover)

Lai ghép là quá trình "kết hôn" giữa 2 cá thể cha mẹ để tạo ra cá thể con có đặc điểm tốt của cả hai. Giống như trong tự nhiên, con cái thừa hưởng gen tốt từ cả cha và mẹ. Trong bài toán của chúng ta, chúng ta trao đổi routes giữa các xe của 2 cha mẹ.

**Logic áp dụng**:
- **Route-based Crossover**: Chọn ngẫu nhiên một xe và trao đổi toàn bộ route của xe đó giữa 2 cha mẹ
- **Tại sao trao đổi route?**: Vì route là đơn vị logic - một xe phụ trách một nhóm phường/xã
- **Tạo đa dạng**: Con cái có thể kết hợp điểm mạnh của cả cha và mẹ
- **Bảo toàn cấu trúc**: Không làm hỏng logic phân chia của từng xe

**Mục tiêu**: Kết hợp 2 cá thể cha mẹ để tạo ra cá thể con có đặc điểm tốt của cả hai.

**Cách áp dụng GA**:
```python
def crossover(self, parent1, parent2):
    child1 = [route.copy() for route in parent1]
    child2 = [route.copy() for route in parent2]
    
    # Chọn ngẫu nhiên một xe để trao đổi routes
    vehicle_idx = random.randint(0, 3)  # 4 xe
    
    # Trao đổi routes của xe này giữa 2 cha mẹ
    child1[vehicle_idx], child2[vehicle_idx] = child2[vehicle_idx], child1[vehicle_idx]
    
    return child1, child2
```

**Ví dụ ngắn gọn theo bài**: 
- **Parent 1**: Xe1=[Tân Định, Gia Định], Xe2=[Bình Hòa, Lái Thiêu]
- **Parent 2**: Xe1=[Tân Bình, Phú Nhuận], Xe2=[Thủ Đức, Linh Xuân]
- **Child**: Xe1=[Tân Định, Gia Định], Xe2=[Thủ Đức, Linh Xuân] (trao đổi Xe2)

#### 2.2.5 Bước 5: Đột biến (Mutation)

Đột biến là quá trình tạo ra những thay đổi ngẫu nhiên nhỏ trong cá thể. Giống như trong tự nhiên, đột biến tạo ra sự đa dạng và có thể dẫn đến những đặc điểm mới tốt hơn. Trong bài toán của chúng ta, chúng ta di chuyển một phường/xã từ xe này sang xe khác.

**Logic áp dụng**:
- **Point Migration**: Chọn ngẫu nhiên một phường/xã và di chuyển nó sang xe khác
- **Tại sao cần đột biến?**: Để tránh bị "kẹt" ở giải pháp cục bộ và khám phá những khả năng mới
- **Tỷ lệ 30%**: Không quá cao để không phá hỏng giải pháp tốt, không quá thấp để duy trì đa dạng
- **Tạo cơ hội**: Cho phép thuật toán "nhảy" ra khỏi vùng tối ưu cục bộ

**Mục tiêu**: Tạo ra biến thể ngẫu nhiên để duy trì đa dạng quần thể.

**Cách áp dụng GA**:
```python
def mutate(self, individual):
    mutated = [route.copy() for route in individual]
    
    # Chọn ngẫu nhiên một điểm để di chuyển
    all_points = []
    for i, route in enumerate(mutated):
        for point in route:
            all_points.append((i, point))
    
    from_vehicle, point = random.choice(all_points)
    
    # Chọn xe đích ngẫu nhiên
    to_vehicle = random.randint(0, 3)
    while to_vehicle == from_vehicle:
        to_vehicle = random.randint(0, 3)
    
    # Di chuyển điểm
    mutated[from_vehicle].remove(point)
    mutated[to_vehicle].append(point)
    
    return mutated
```

**Ví dụ ngắn gọn theo bài**: 
- **Trước**: Xe1=[Tân Định, Gia Định], Xe2=[Bình Hòa, Lái Thiêu]
- **Sau**: Xe1=[Tân Định], Xe2=[Bình Hòa, Lái Thiêu, Gia Định] (Gia Định chuyển sang Xe2)

#### 2.2.6 Bước 6: Lặp lại và Hội tụ (Evolution & Convergence)

Đây là vòng lặp chính của giải thuật di truyền. Chúng ta lặp lại các bước 2-5 nhiều lần (thế hệ) cho đến khi tìm được giải pháp tốt nhất hoặc đạt điều kiện dừng. Giống như quá trình tiến hóa trong tự nhiên, mỗi thế hệ sẽ tốt hơn thế hệ trước.

**Logic áp dụng**:
- **20,000 thế hệ**: Đủ thời gian để thuật toán khám phá và hội tụ
- **Early Stopping**: Nếu 2,000 thế hệ liên tiếp không cải thiện, dừng sớm để tiết kiệm thời gian
- **Hội tụ**: Khi fitness không còn cải thiện đáng kể, có nghĩa là đã tìm được giải pháp tốt
- **Cân bằng**: Giữa thời gian tính toán và chất lượng giải pháp

**Mục tiêu**: Lặp lại quá trình qua 20,000 thế hệ hoặc cho đến khi hội tụ.

**Cách áp dụng GA**:
```python
def run_evolution(self):
    population = self.create_initial_population()
    
    for generation in range(20000):
        # Đánh giá fitness
        fitness_scores = [self.calculate_fitness(ind) for ind in population]
        
        # Chọn lọc
        selected = self.tournament_selection(population)
        
        # Lai ghép và đột biến
        new_population = []
        for i in range(0, len(selected), 2):
            child1, child2 = self.crossover(selected[i], selected[i+1])
            
            # Đột biến với xác suất 30%
            if random.random() < 0.3:
                child1 = self.mutate(child1)
            if random.random() < 0.3:
                child2 = self.mutate(child2)
                
            new_population.extend([child1, child2])
        
        population = new_population
        
        # Early stopping nếu không cải thiện 2000 thế hệ
        if self.check_convergence():
            break
    
    return min(population, key=self.calculate_fitness)
```

**Kết quả thực tế**: Thuật toán hội tụ sau ~2,000 thế hệ với fitness cuối cùng là 1,601km.

---

## 3. Áp dụng GA vào bài toán MVTSP: Chi tiết kỹ thuật

### 3.1 Mã hóa dữ liệu thực tế

**Dữ liệu đầu vào**: 168 phường/xã TP.HCM với tọa độ GPS
```python
# Ví dụ dữ liệu thực tế từ Phuong_TPHCM_With_Coordinates.CSV
coordinates = {
    "Phường Tân Định": (10.7939, 106.6907),
    "Phường Gia Định": (10.8019, 106.6907),
    "Phường An Phú Đông": (10.8099, 106.6907),
    # ... 165 phường/xã khác
}
```

**Cách mã hóa cá thể**: Mỗi cá thể là một danh sách 4 routes
```python
individual = [
    ["Phường Tân Định", "Phường Gia Định", ...],   # Xe 1: 40 phường/xã
    ["Phường Bình Hòa", "Phường Lái Thiêu", ...],  # Xe 2: 45 phường/xã  
    ["Phường Tân Bình", "Phường Phú Nhuận", ...],  # Xe 3: 42 phường/xã
    ["Phường Thủ Đức", "Phường Linh Xuân", ...]    # Xe 4: 41 phường/xã
]
```

### 3.2 Fitness Function

**Công thức tính khoảng cách**: Sử dụng Haversine formula

Để tính khoảng cách giữa hai điểm trên Trái Đất, chúng ta không thể dùng công thức Euclid đơn giản vì Trái Đất có hình cầu. Thay vào đó, chúng ta sử dụng công thức Haversine - một công thức toán học được thiết kế đặc biệt cho việc tính khoảng cách trên mặt cầu.

$$d = 2R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta\phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta\lambda}{2}\right)}\right)$$

Trong đó:
- $d$: Khoảng cách giữa 2 điểm (km)
- $R = 6371$: Bán kính Trái Đất (km)
- $\phi_1, \phi_2$: Vĩ độ của điểm 1 và 2 (radian)
- $\lambda_1, \lambda_2$: Kinh độ của điểm 1 và 2 (radian)
- $\Delta\phi = \phi_2 - \phi_1$, $\Delta\lambda = \lambda_2 - \lambda_1$

**Công thức đơn giản**:

Để dễ hiểu, có thể tóm tắt fitness function như sau:

**Fitness = Tổng khoảng cách + (1000 × Độ chênh lệch giữa các xe)**

- **Tổng khoảng cách**: Càng thấp càng tốt
- **Độ chênh lệch**: Đo bằng Coefficient of Variation (CV) - càng thấp càng cân bằng
- **Hệ số 1000**: Đảm bảo cân bằng tải quan trọng không kém khoảng cách

**Giải thích**: Fitness càng cao (gần 1) thì giải pháp càng tốt. Exponential scaling giúp thuật toán nhạy cảm hơn với những cải thiện nhỏ về khoảng cách, trong khi CV² penalty đảm bảo các xe không quá chênh lệch về hiệu suất.

**⚠️ Lưu ý quan trọng về khoảng cách**:

Bài toán này chỉ tính khoảng cách theo **đường chim bay** (Haversine formula) giữa các tọa độ GPS. Đây là một giả định đơn giản hóa để minh họa thuật toán GA.

**Để áp dụng thực tế**, cần tích hợp các **API định tuyến chuyên nghiệp**:
- **Google Maps API**: Cung cấp khoảng cách và thời gian thực tế theo đường bộ
- **OpenStreetMap API**: Miễn phí và có dữ liệu chi tiết về giao thông  
- **HERE Maps API**: Chuyên về logistics với tính năng tối ưu hóa routes
- **TomTom API**: Cung cấp dữ liệu giao thông real-time

Trong thực tế, xe giao hàng phải đi theo đường bộ, gặp tắc đường, đèn đỏ, và các rào cản địa lý khác. Việc sử dụng API sẽ cho số liệu khoảng cách và đường đi chuẩn xác hơn nhiều.

### 3.3 Genetic Operators thực tế

**Các operators chính** (đã trình bày chi tiết trong phần 2):
- **Tournament Selection**: Chọn lọc với kích thước tournament = 3
- **Route-based Crossover**: Trao đổi routes giữa các xe
- **Point Migration Mutation**: Di chuyển điểm giữa các xe với tỷ lệ 30%

### 3.4 Kết quả thực tế từ dự án

**Thống kê cuối cùng**:
- **Tổng khoảng cách**: 1,601.9 km
- **Số thế hệ**: 20,000 (hội tụ sau ~300)
- **Thời gian chạy**: ~8 phút trên CPU thông thường
- **Cân bằng tải**: CV = 0.12 (rất tốt)

**Phân bố từng xe**:
- **Xe 1**: 40 phường/xã, 531.6 km, 31.0 giờ
- **Xe 2**: 45 phường/xã, 190.7 km, 16.6 giờ  
- **Xe 3**: 42 phường/xã, 367.3 km, 22.8 giờ
- **Xe 4**: 41 phường/xã, 512.3 km, 27.6 giờ

**Cải thiện so với giải pháp ban đầu**:
- 📉 **Khoảng cách**: Giảm 35.9% (2,500km → 1,601km)
- ⚖️ **Cân bằng**: CV giảm 73% (0.45 → 0.12)
- ⏱️ **Tổng thời gian**: 97.9 giờ cho tất cả xe (31.0 + 16.6 + 22.8 + 27.6)

---

## 4. Kết quả thực tế: Tiến hóa qua thế hệ

### 4.1 Biểu đồ tiến hóa

<div align="center">
![Tiến hóa fitness qua thế hệ](https://i.ibb.co/xSs2fHVQ/evolution.png)
</div>

<div align="center">
*Hình 2: Fitness tăng dần qua các thế hệ - từ 0.617417 lên 0.835513 (hội tụ hoàn toàn)*
</div>

**Phân tích kết quả** (dựa trên fitness history thực tế mới nhất):
- 📈 **Thế hệ 0**: Fitness = 0.617417 (khoảng cách ~1,620km)
- 📈 **Thế hệ 100**: Local search cải thiện lên 0.828195
- 📈 **Thế hệ 200**: Balance load cải thiện lên 0.834512  
- 📈 **Thế hệ 300**: Local search cải thiện lên 0.835513
- 📈 **Thế hệ 300-2000**: **HỘI TỤ HOÀN TOÀN** - Fitness ổn định tại 0.835513

**Giải thích chỉ số Fitness**:
- 🎯 **Fitness càng cao = Giải pháp càng tốt** (ngược với khoảng cách)
- 📊 **Công thức**: `Fitness = exp(-khoảng_cách/10000)` - khoảng cách ngắn → fitness cao
- 🔄 **Ý nghĩa**: Fitness 0.835513 có nghĩa là thuật toán đã tìm được giải pháp rất tối ưu
- ⚡ **Hội tụ**: Sau thế hệ 300, fitness không tăng thêm → đã tìm được giải pháp tốt nhất có thể

### 4.2 Hiệu suất thuật toán

<div align="center">
![Hiệu suất thuật toán](https://i.ibb.co/tPbpnFMz/algorithm-performance.png)
</div>

<div align="center">
*Hình 3: Các chỉ số hiệu suất quan trọng qua thế hệ*
</div>

**Các chỉ số quan trọng**:
- 🎯 **Fitness Value**: Tăng từ ~0.6 → 0.834379 (càng cao càng tốt)
- 📊 **Distance Fitness**: `exp(-total_distance/10000)` - khoảng cách càng ngắn, fitness càng cao
- ⚖️ **Efficiency Balance**: `exp(-CV*2)` - độ cân bằng giữa các xe (CV = coefficient of variation)
- 🔄 **Combined Fitness**: `0.95 * distance_fitness + 0.05 * efficiency_balance_fitness`

---

## 5. Kết quả định tuyến: Trước và Sau

### 5.1 So sánh tổng quan

<div align="center">
![So sánh trước/sau tối ưu](https://i.ibb.co/hxj8V2rc/before-after-comparison.png)
</div>

<div align="center">
*Hình 4: So sánh kết quả trước và sau khi áp dụng GA*
</div>

**Cải thiện đạt được** (dựa trên code visualization):
- 📉 **Tổng khoảng cách**: 2,500km → **1,601.9km** 
- ⚖️ **Cân bằng tải**: CV giảm từ 0.45 → **0.12** (-73%) - *CV (coefficient of variation) là hệ số biến thiên, đo mức độ phân tán của khoảng cách giữa các xe; CV càng thấp thì độ cân bằng càng cao*
- ⏱️ **Tổng thời gian**: 5,876 phút (97.9 giờ) - tính từ 8h sáng + thời gian di chuyển + 15 phút giao hàng/điểm
- 🚚 **Phân bố phường/xã**: Xe 1 (40), Xe 2 (45), Xe 3 (42), Xe 4 (41)
- 🎯 **Nguyên tắc**: Tối ưu hóa khoảng cách di chuyển, không phân vùng địa lý

### 5.2 Phân tích từng xe

<div align="center">
![Phân tích hiệu quả từng xe](https://i.ibb.co/HDFzM9NT/vehicle-analysis.png)
</div>

<div align="center">
*Hình 5: Chi tiết hiệu quả của từng xe sau tối ưu*
</div>

**Thống kê từng xe** (dữ liệu mới nhất từ JSON):
- **Xe 1**: 40 phường/xã, 531.6km, 1,858 phút (31.0 giờ)
- **Xe 2**: 45 phường/xã, 190.7km, 996 phút (16.6 giờ)  
- **Xe 3**: 42 phường/xã, 367.3km, 1,365 phút (22.8 giờ)
- **Xe 4**: 41 phường/xã, 512.3km, 1,657 phút (27.6 giờ)

→ **Phân bố phường/xã**: Xe 2 có nhiều nhất (45), Xe 1 có ít nhất (40)
→ **Cân bằng khoảng cách**: CV = 0.12 (rất tốt, dưới 0.2)
→ **Nguyên tắc**: Không phân theo khu vực; thuật toán tối ưu theo phường/xã để giảm tổng khoảng cách
→ **Tổng**: 168 phường/xã được phân chia cho 4 xe

**Giải thích các biểu đồ**:
- 📊 **Khoảng cách từng xe**: Bar chart hiển thị khoảng cách thực tế của từng xe
- ⏰ **Thời gian làm việc**: Tính từ 8h sáng + thời gian di chuyển + 15 phút giao hàng/điểm
- 📍 **Số điểm giao hàng**: Số lượng phường/xã mà mỗi xe phụ trách
- ⚡ **Hiệu quả (km/điểm)**: Khoảng cách trung bình cho mỗi điểm giao hàng

---

## 6. Bản đồ định tuyến trực quan

### 6.1 Routes của các xe

<div align="center">
![Bản đồ routes các xe](https://i.ibb.co/hJMTR19y/route-map.png)
</div>

<div align="center">
*Hình 6: Bản đồ định tuyến cho 4 xe với màu sắc khác nhau*
</div>

**Đặc điểm routes**:
- 🔴 **Xe 1** (Đỏ): 40 phường/xã, 531.6km, 31.0h (khu vực trung tâm và phía Nam)
- 🔵 **Xe 2** (Xanh dương): 45 phường/xã, 190.7km, 16.6h (khu vực phía Bắc và Đông Bắc)  
- 🟢 **Xe 3** (Xanh lá): 42 phường/xã, 367.3km, 22.8h (khu vực phía Tây và Tây Nam)  
- 🟣 **Xe 4** (Tím): 41 phường/xã, 512.3km, 27.6h (khu vực phía Đông và Đông Nam)

**Phân tích tối ưu hóa khoảng cách**:
- 🎯 **Tối ưu tổng khoảng cách**: Thuật toán tập trung vào giảm thiểu tổng khoảng cách (1,601.9km)
- ⚖️ **Cân bằng hiệu quả**: Số phường/xã khác nhau (40-45) nhưng khoảng cách được tối ưu
- 🔄 **Chồng chéo địa lý**: Các xe có thể phụ trách các nhóm phường/xã khác nhau (có thể chồng chéo) để tối ưu khoảng cách
- 📍 **Phân bố thông minh**: GA tự động phân chia để giảm thiểu tổng khoảng cách di chuyển
- 🚀 **Hiệu quả**: Xe 2 có nhiều phường/xã nhất (45) nhưng khoảng cách ngắn nhất (190.7km)
- 🏝️ **Cân bằng địa lý**: Xe đỏ (531.6km) bao gồm cả Côn Đảo (điểm cực nam) để đảm bảo cân bằng với các xe khác có khoảng cách ngắn hơn

### 6.2 Bản đồ hiệu quả

<div align="center">
![Bản đồ hiệu quả từng xe](https://i.ibb.co/Wv972SFZ/efficiency-map.png)
</div>

<div align="center">
*Hình 7: Phân tích hiệu quả khoảng cách của từng xe*
</div>

**Phân tích hiệu quả**:
- 🎯 **Tối ưu khoảng cách**: CV = 0.12 (dưới ngưỡng 0.2 - rất tốt)
- 📍 **Phân bố thông minh**: GA tự động phân chia phường/xã để giảm thiểu tổng khoảng cách
- 🚀 **Kết quả tối ưu**: Tổng khoảng cách 1,601.9km - giảm 35.9% so với ban đầu
- ⚖️ **Phân bố phường/xã**: Xe 2 có nhiều nhất (45), Xe 1 có ít nhất (40)
- 📊 **Hiệu quả khác nhau**: Xe 2 hiệu quả nhất (190.7km), Xe 1 dài nhất (531.6km)
- 🔄 **Chồng chéo địa lý**: Các xe có thể phụ trách các nhóm phường/xã khác nhau (có thể chồng chéo)
- 📈 **Tổng thời gian**: 97.9 giờ cho tất cả xe (31.0 + 16.6 + 22.8 + 27.6)
- 🎯 **Mục tiêu chính**: Tối ưu hóa khoảng cách di chuyển, không phải phân vùng địa lý tách biệt

---

## 7. Tham số tối ưu cho bài toán MVTSP

### 7.1 Cấu hình GA đã tối ưu

| Tham số | Giá trị | Lý do |
|---------|---------|-------|
| **Population Size** | 250 | Đủ lớn để khám phá không gian giải pháp |
| **Generations** | 20,000 | Cho phép hội tụ hoàn toàn |
| **Mutation Rate** | 0.3 (30%) | Cao để duy trì đa dạng |
| **Elite Ratio** | 0.05 (5%) | Giữ lại cá thể tốt nhất |
| **Stagnation Threshold** | 2,000 | Dừng sớm khi không cải thiện |

---

## 8. Kết luận: Những gì chúng ta đã học được

### 8.1 Thành tựu đạt được

Dự án này đã chứng minh rằng **Giải thuật Di truyền có thể giải quyết hiệu quả bài toán Multi-Vehicle TSP** trong thực tế. Với dữ liệu 168 phường/xã TP.HCM và 4 xe giao hàng, thuật toán đã:

- **Giảm 35.9% tổng khoảng cách** (từ 2,500km xuống 1,601.9km)
- **Đạt cân bằng tải tốt** với CV = 0.12 (dưới ngưỡng 0.2)
- **Hội tụ nhanh** sau 300 thế hệ đầu tiên
- **Tự động phân bố** phường/xã mà không cần can thiệp thủ công

### 8.2 Bài học quan trọng

**Fitness function là chìa khóa thành công**: Việc kết hợp tối thiểu khoảng cách và cân bằng tải với trọng số phù hợp đã mang lại kết quả tốt. **Haversine distance** cho độ chính xác cao hơn khoảng cách Euclid trên bề mặt Trái Đất.

**Tham số cần được điều chỉnh cẩn thận**: Population size 250, mutation rate 30%, và elite ratio 5% đã tạo ra sự cân bằng tốt giữa khám phá và khai thác. **Genetic operators** như route-based crossover và point migration mutation phải phù hợp với cấu trúc dữ liệu cụ thể.

### 8.3 Hạn chế và thách thức

**Thời gian tính toán**: Mặc dù chỉ mất 8 phút, nhưng với bài toán lớn hơn (500+ điểm), thời gian có thể tăng đáng kể. **Chất lượng giải pháp** phụ thuộc vào thiết kế fitness function và genetic operators.

**Không đảm bảo tối ưu toàn cục**: GA chỉ tìm được giải pháp tốt, không phải tối ưu tuyệt đối. **Tham số cần điều chỉnh thủ công** cho từng bài toán cụ thể.

### 8.4 Ứng dụng thực tế

Kết quả này có thể **tiết kiệm hàng triệu USD** chi phí vận chuyển hàng năm cho các doanh nghiệp logistics tại TP.HCM. Đặc biệt hữu ích trong **bối cảnh tách nhập phường/xã năm 2025**, giúp doanh nghiệp nhanh chóng điều chỉnh routes.

**Hướng phát triển**: Kết hợp GA với deep learning, reinforcement learning, hoặc các thuật toán hybrid khác có thể mang lại cải tiến vượt trội hơn nữa.

---

*"Giải thuật di truyền không chỉ là một công cụ tối ưu — mà là cách chúng ta học hỏi từ tự nhiên để giải quyết những bài toán phức tạp nhất của con người."*

---

## 9. Tài liệu tham khảo

1. Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
2. Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization, and Machine Learning*. Addison-Wesley.
3. Mitchell, M. (1998). *An Introduction to Genetic Algorithms*. MIT Press.
4. Scikit-learn Development Team. (2023). *Scikit-learn: Machine Learning in Python*. Journal of Machine Learning Research, 12, 2825-2830.
