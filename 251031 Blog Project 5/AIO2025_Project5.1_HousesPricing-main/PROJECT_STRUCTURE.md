# 📁 Cấu trúc Dự án: House Price Prediction

## 📋 Tổng quan

Dự án **House Price Prediction** là hệ thống dự đoán giá nhà sử dụng Machine Learning, được xây dựng với các công nghệ hiện đại:

- **ML Model**: XGBoost Regressor
- **API Framework**: FastAPI
- **Frontend**: Streamlit
- **Experiment Tracking**: MLflow
- **Containerization**: Docker & Docker Compose

**Dataset**: Ames Housing Dataset (1460 mẫu, 80 features)

---

## 🗂️ Cấu trúc Thư mục

```
AIO2025_Project5.1_HousesPricing-main/
│
├── 📂 data/                           # Dữ liệu
│   └── raw/                           # Dữ liệu gốc (không thay đổi)
│       └── train-house-prices-advanced-regression-techniques.csv
│
├── 📂 src/                            # Source code chính
│   │
│   ├── 📂 api/                        # FastAPI Application
│   │   ├── __init__.py
│   │   ├── main.py                    # FastAPI app, endpoints định nghĩa
│   │   ├── models.py                  # Pydantic models cho request/response
│   │   ├── inference.py               # Inference logic & CLI tool
│   │   ├── run_api.py                 # Script khởi chạy API server
│   │   ├── test_api.py                # Test script cho API endpoints
│   │   ├── Dockerfile                 # Docker image cho API service
│   │   └── README.md                  # Tài liệu API
│   │
│   ├── 📂 processing/                 # Xử lý dữ liệu
│   │   ├── __init__.py
│   │   ├── transformers.py            # Custom transformers (OrdinalMapper, MissingnessIndicator, etc.)
│   │   └── data_processing.py         # Preprocessing pipeline
│   │
│   ├── 📂 e_featuring/                # Feature Engineering
│   │   ├── __init__.py
│   │   └── data_featuring.py          # Domain-specific features (18 features)
│   │
│   ├── 📂 training/                   # Training Model
│   │   ├── __init__.py
│   │   ├── pipeline.py                # Training pipeline builder
│   │   └── train_model.py             # Main training script với MLflow
│   │
│   ├── 📂 configs/                     # Cấu hình
│   │   └── best_model_config.json     # Cấu hình model tốt nhất (hyperparameters, performance metrics)
│   │
│   ├── 📂 frontend/                    # Streamlit UI
│   │   ├── app.py                     # Streamlit application
│   │   ├── Dockerfile                 # Docker image cho frontend
│   │   ├── FRONTEND_DESIGN.md         # Thiết kế UI
│   │   └── README.md                  # Tài liệu frontend
│   │
│   ├── 📂 models/                      # Models đã train (auto-generated, không commit)
│   │   ├── best_pipeline.joblib      # Pipeline hoàn chỉnh (features + model)
│   │   └── feature_pipeline.joblib   # Feature engineering pipeline
│   │
│   └── __init__.py
│
├── 📂 deployments/                     # Deployment configurations
│   │
│   ├── 📂 api/                         # API deployment
│   │   ├── docker-compose.yaml        # Docker Compose cho API + Frontend + MLflow
│   │   └── Dockerfile                 # (tùy chọn, có thể dùng từ src/api/)
│   │
│   └── 📂 mlflow/                      # MLflow tracking server
│       └── docker-compose.yaml        # MLflow standalone server
│
├── 📂 notebooks/                       # Jupyter Notebooks
│   ├── house_price_analysis.ipynb      # Exploratory data analysis
│   └── house_price_analysis_mlflow.ipynb  # Experiments với MLflow tracking
│
├── 📄 train.py                         # Script training chính (entry point)
├── 📄 requirements.txt                 # Python dependencies
├── 📄 README.md                        # Tài liệu chính của dự án
├── 📄 .gitignore                       # Git ignore rules
├── 📄 .gitattributes                   # Git attributes
└── 📄 .pre-commit-config.yaml         # Pre-commit hooks
```

---

## 🔍 Chi tiết các Module

### 1. 📂 `data/raw/` - Dữ liệu

**Mục đích**: Lưu trữ dữ liệu gốc, không thay đổi.

**File chính**:
- `train-house-prices-advanced-regression-techniques.csv`: Dataset Ames Housing với 1460 mẫu và 80 features

**Lưu ý**: 
- Dữ liệu gốc giữ nguyên, không thay đổi
- Không lưu intermediate data, chỉ dùng pipeline để transform

---

### 2. 📂 `src/api/` - FastAPI Application

**Mục đích**: Cung cấp REST API để dự đoán giá nhà.

#### `main.py`
- **Chức năng**: FastAPI application với các endpoints
- **Endpoints**:
  - `GET /` - Root endpoint
  - `GET /health` - Health check
  - `GET /model/info` - Thông tin model
  - `POST /predict` - Dự đoán giá cho một căn nhà
  - `POST /predict/batch` - Dự đoán batch nhiều căn nhà
- **Features**:
  - Auto-load model khi start
  - CORS middleware
  - Exception handling
  - Model lifecycle management

#### `models.py`
- **Chức năng**: Pydantic models định nghĩa request/response schemas
- **Models**:
  - `HouseFeatures`: Input features cho prediction
  - `PredictionResponse`: Response với predicted price
  - `BatchPredictionRequest/Response`: Batch prediction
  - `HealthResponse`: Health check response
  - `ModelInfoResponse`: Model information response

#### `inference.py`
- **Chức năng**: CLI tool để chạy inference từ command line
- **Usage**: `python src/api/inference.py <input_csv> --output <output_csv>`

#### `run_api.py`
- **Chức năng**: Script khởi chạy API server với uvicorn

#### `test_api.py`
- **Chức năng**: Test script để kiểm tra các API endpoints

---

### 3. 📂 `src/processing/` - Xử lý Dữ liệu

**Mục đích**: Preprocessing pipeline và custom transformers.

#### `transformers.py`
**Custom Transformers**:
- `OrdinalMapper`: Map categorical variables sang numeric theo thứ tự
- `MissingnessIndicator`: Tạo indicators cho missing values
- `RarePooler`: Gộp rare categories thành 'Other'
- `TargetEncoderTransformer`: Target encoding với smoothing
- `FiniteCleaner`: Convert infinite values thành NaN
- `DropAllNaNColumns`: Loại bỏ columns toàn NaN

#### `data_processing.py`
**Preprocessing Pipeline**:
- Ordinal encoding cho 20 ordinal features
- Missing value imputation (categorical: most_frequent, numerical: median)
- One-hot encoding cho categorical features
- Quantile transformation cho continuous features

---

### 4. 📂 `src/e_featuring/` - Feature Engineering

**Mục đích**: Tạo domain-specific features.

#### `data_featuring.py`
**18 Domain Features**:
- **Area features**: `TotalSF`, `TotalPorchSF`, `LotArea_clip`
- **Age features**: `HouseAge`, `RemodAge`, `GarageAge`
- **Bathroom features**: `TotalBath`, `BathPerBedroom`
- **Binary features**: `IsRemodeled`, `Has2ndFlr`
- **Ratio features**: `RoomsPerArea`, `LotAreaRatio`
- **Seasonal features**: `MoSold_sin`, `MoSold_cos`
- **Interaction features**: `Neighborhood_BldgType`, `IQ_OQ_GrLiv`, `IQ_OQ_TotalSF`
- **Log transformation**: `Ln_TotalSF`

---

### 5. 📂 `src/training/` - Training Model

**Mục đích**: Training pipeline và model training.

#### `pipeline.py`
- **Chức năng**: Build complete training pipeline
- **Functions**:
  - `build_model_pipeline()`: Kết hợp feature pipeline + model
  - `evaluate_model()`: Cross-validation và test evaluation

#### `train_model.py`
- **Chức năng**: Main training script với MLflow tracking
- **Quy trình**:
  1. Load data và split train/test (80/20)
  2. Build feature pipeline với domain features
  3. Create XGBoost model với hyperparameters từ config
  4. Cross-validation (5-fold)
  5. Test set evaluation
  6. Log metrics và artifacts vào MLflow
  7. Save pipeline (`best_pipeline.joblib` và `feature_pipeline.joblib`)

---

### 6. 📂 `src/configs/` - Cấu hình

**Mục đích**: Lưu trữ cấu hình model tốt nhất.

#### `best_model_config.json`
- **Nội dung**:
  - `hyperparameters`: Best hyperparameters cho XGBoost
  - `feature_engineering`: Config cho feature engineering (target encoder columns, alpha, rare pooler)
  - `performance`: Expected performance metrics (CV RMSE, Test RMSE, R²)

---

### 7. 📂 `src/frontend/` - Streamlit UI

**Mục đích**: Giao diện web để người dùng tương tác với model.

#### `app.py`
- **Chức năng**: Streamlit application
- **Features**:
  - Form nhập thông tin căn nhà
  - Kết nối với FastAPI backend
  - Hiển thị kết quả prediction
  - Error handling và validation

---

### 8. 📂 `src/models/` - Trained Models

**Mục đích**: Lưu trữ models đã train (auto-generated, không commit lên Git).

**Files**:
- `best_pipeline.joblib`: Pipeline hoàn chỉnh (features + model)
- `feature_pipeline.joblib`: Feature engineering pipeline (riêng biệt)

**Lưu ý**: 
- Models được tạo sau khi chạy `train.py`
- Không commit vào Git (trong `.gitignore`)

---

### 9. 📂 `deployments/` - Deployment

**Mục đích**: Cấu hình deployment cho production.

#### `deployments/api/docker-compose.yaml`
**Services**:
- **api**: FastAPI service (port 8000)
- **frontend**: Streamlit app (port 8501)
- **mlflow**: MLflow tracking server (port 5555)

**Features**:
- Docker network isolation
- Health checks
- Volume mounts cho models và data
- Auto-restart policies

#### `deployments/mlflow/docker-compose.yaml`
- **Chức năng**: Standalone MLflow server
- **Port**: 5555

---

### 10. 📂 `notebooks/` - Jupyter Notebooks

**Mục đích**: Exploratory data analysis và experiments.

**Files**:
- `house_price_analysis.ipynb`: EDA và baseline models
- `house_price_analysis_mlflow.ipynb`: Experiments với MLflow tracking

---

## 🔄 Quy trình Workflow

### Training Flow

```
1. Raw Data (data/raw/)
   ↓
2. Preprocessing (src/processing/)
   - Custom transformers
   - Missing value imputation
   - Encoding
   ↓
3. Feature Engineering (src/e_featuring/)
   - Domain features
   - Interaction features
   - Transformations
   ↓
4. Model Training (src/training/)
   - XGBoost training
   - Cross-validation
   - Test evaluation
   ↓
5. MLflow Tracking
   - Log hyperparameters
   - Log metrics
   - Save artifacts
   ↓
6. Save Pipeline (src/models/)
   - best_pipeline.joblib
   - feature_pipeline.joblib
```

### Inference Flow

```
1. Request (API/Frontend)
   ↓
2. Feature Extraction (convert to DataFrame)
   ↓
3. Load Pipeline (best_pipeline.joblib)
   ↓
4. Transform & Predict
   ↓
5. Return Result
```

---

## 📊 Model Performance

**Best Model Configuration** (từ `best_model_config.json`):
- **CV RMSE**: 25259.42 ± 3479.64
- **Test RMSE**: 24608.89
- **Test R²**: 0.9210

**Dataset**:
- **Samples**: 1460
- **Features**: 80 (sau feature engineering: 98+)
- **Target**: SalePrice (continuous)

---

## 🛠️ Tech Stack

### Core Libraries
- **pandas** (1.5.3): Data manipulation
- **numpy** (1.24.3): Numerical operations
- **scikit-learn** (1.2.2): ML models & preprocessing
- **xgboost**: Gradient boosting model

### API & Web
- **FastAPI** (0.95.2): REST API framework
- **uvicorn** (0.22.0): ASGI server
- **streamlit** (≥1.24.0): Frontend UI
- **requests** (≥2.28.0): HTTP client

### ML Operations
- **MLflow** (2.3.1): Experiment tracking & model management

### Visualization
- **matplotlib** (3.7.1): Plotting
- **seaborn** (0.12.2): Statistical graphics

### Utilities
- **joblib** (1.3.1): Model serialization
- **pytest** (7.3.1): Testing
- **pyyaml** (6.0): YAML parsing

---

## 🚀 Quick Start

### 1. Training Model

```bash
# Install dependencies
pip install -r requirements.txt

# Train model
python train.py
```

### 2. Run API

```bash
# Method 1: Python script
python src/api/run_api.py

# Method 2: Docker Compose (recommended)
cd deployments/api
docker compose up -d --build
```

### 3. Access Services

- **API**: http://localhost:8000 (Docs: http://localhost:8000/docs)
- **Frontend**: http://localhost:8501
- **MLflow**: http://localhost:5555

---

## 📝 Notes

### File Organization
- **Raw data**: Giữ nguyên trong `data/raw/`
- **Trained models**: Lưu trong `src/models/` (không commit lên Git)
- **MLflow data**: Lưu trong `deployments/mlflow/` (không commit lên Git)
- **Intermediate data**: **Không lưu** - chỉ dùng pipeline để transform

### Git Ignore
- `src/models/*.joblib`: Trained models
- `mlruns/`: MLflow tracking data
- `__pycache__/`: Python cache
- `.env`: Environment variables

### Best Practices
- Mọi thay đổi dữ liệu đều qua pipeline (không thay đổi raw data)
- Models được version với MLflow
- API có health checks và error handling
- Docker images được optimize cho production

---

## 🔗 Liên kết

- **API Documentation**: Xem `src/api/README.md`
- **Frontend Design**: Xem `src/frontend/FRONTEND_DESIGN.md`
- **Main README**: Xem `README.md`

---

**Tác giả**: AIO2025 Project 5.1 Team  
**Cập nhật**: 2025

