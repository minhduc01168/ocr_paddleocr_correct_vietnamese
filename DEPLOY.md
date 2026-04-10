# 🚀 Hướng dẫn Deploy — OCR SageMaker Service

> **Phiên bản**: 2.0 | **Cập nhật**: 2026-04-10
>
> Service gồm 2 thành phần:
> - **`ocr-engine/`** — AI Engine nặng: PaddleOCR + HuggingFace Correction Model
> - **`ocr/`** — Orchestrator nhẹ: nhận request và điều phối về Engine

---

## 📋 Mục lục

1. [Đề xuất phần cứng](#-đề-xuất-phần-cứng)
2. [Biến môi trường](#-biến-môi-trường)
3. [Chạy local (Docker Compose)](#-chạy-local-docker-compose)
4. [Chạy local (không Docker)](#-chạy-local-không-docker)
5. [Deploy SageMaker](#-deploy-sagemaker)
6. [Checklist deploy](#-checklist-deploy)
7. [Troubleshooting](#-troubleshooting)

---

## 🖥️ Đề xuất phần cứng

### Môi trường chạy

| Tình huống | CPU | RAM | GPU | Disk | Ghi chú |
|---|---|---|---|---|---|
| **Dev local (CPU)** | 4+ cores | 16 GB | Không cần | 20 GB | `OCR_IMAGE_SIZE=1024`, `OCR_DPI=150` để tăng tốc |
| **Dev local (GPU)** | 4+ cores | 16 GB | 6 GB VRAM | 20 GB | NVIDIA RTX 3060 trở lên |
| **Staging server** | 8 cores | 32 GB | 8 GB VRAM | 50 GB | Phù hợp tải vừa (<100 tài liệu/ngày) |
| **Production (khuyến nghị)** | 16+ cores | 64 GB | 16 GB VRAM | 100 GB | Xử lý song song nhiều tài liệu |

### AWS SageMaker Instance Types

| Instance | vCPU | RAM | GPU/VRAM | Giá (~USD/h) | Phù hợp |
|---|---|---|---|---|---|
| `ml.c5.2xlarge` | 8 | 16 GB | CPU only | ~$0.34 | Dev/Test, khối lượng thấp |
| `ml.c5.4xlarge` | 16 | 32 GB | CPU only | ~$0.68 | Staging, tài liệu nhỏ |
| **`ml.g4dn.xlarge`** ⭐ | 4 | 16 GB | T4/16 GB | ~$0.74 | **Khuyến nghị** — GPU/CPU ratio tốt nhất |
| `ml.g4dn.2xlarge` | 8 | 32 GB | T4/16 GB | ~$1.18 | Production, xử lý song song |
| `ml.g5.xlarge` | 4 | 16 GB | A10G/24 GB | ~$1.41 | PDF phức tạp, chữ nhỏ |
| `ml.g5.2xlarge` | 8 | 32 GB | A10G/24 GB | ~$1.82 | High-throughput production |

> **💡 Khuyến nghị**: Bắt đầu với `ml.g4dn.xlarge`. Chỉ upgrade lên `g5` khi cần OCR tài liệu nhiều ảnh phức tạp hoặc cần accuracy cao hơn.

### Tham số môi trường theo phần cứng

```
# CPU (ml.c5.*)
OCR_IMAGE_SIZE=1024
OCR_DPI=150
MAX_WORKERS=4       # = vCPU count / 2

# GPU T4 16GB (ml.g4dn.*)  ← KHUYẾN NGHỊ
OCR_IMAGE_SIZE=1600
OCR_DPI=200
MAX_WORKERS=2       # Giới hạn 2 để tránh OOM VRAM

# GPU A10G 24GB (ml.g5.*)
OCR_IMAGE_SIZE=2048
OCR_DPI=300
MAX_WORKERS=2
```

---

## 🔧 Biến môi trường

### `ocr-engine/` — AI Engine

| Biến | Mặc định | Mô tả |
|---|---|---|
| `API_PORT` | `8002` | Port API (SageMaker BẮT BUỘC `8080`) |
| `LOG_LEVEL` | `INFO` | `DEBUG` \| `INFO` \| `WARNING` \| `ERROR` |
| `CORRECT_MODEL_ID` | `protonx-models/protonx-legal-tc` | HuggingFace model ID hoặc đường dẫn local |
| `MODEL_CACHE_DIR` | `~/.cache/huggingface` | Thư mục cache HuggingFace |
| `OCR_IMAGE_SIZE` | `1600` | Kích thước ảnh tối đa trước OCR (px) |
| `OCR_DPI` | `200` | DPI render PDF → ảnh |
| `MAX_PDF_PAGES` | `100` | Số trang PDF tối đa |
| `MAX_FILE_SIZE_MB` | `50` | Dung lượng file tối đa (MB) |
| `MAX_WORKERS` | auto | Số thread song song (auto: GPU=2, CPU=cpu_count) |

### `ocr/` — Orchestrator

| Biến | Mặc định | Mô tả |
|---|---|---|
| `API_PORT` | `8001` | Port API (SageMaker BẮT BUỘC `8080`) |
| `LOG_LEVEL` | `INFO` | Log level |
| `WORKERS` | `1` | Số Uvicorn workers |
| `OCR_ENGINE_URL` | `http://localhost:8002` | URL nội bộ tới Engine service |
| `GATEWAY_URL` | `http://localhost:4000` | URL LiteLLM Gateway (AI Hub) |
| `GATEWAY_KEY` | `sk-aihub-gateway-master` | Auth key cho Gateway |
| `MAX_PDF_PAGES` | `100` | Số trang PDF tối đa |
| `MAX_FILE_SIZE_MB` | `50` | Dung lượng file tối đa (MB) |

---

## 🐳 Chạy local (Docker Compose)

### Yêu cầu

- Docker Desktop ≥ 24.0
- RAM khả dụng: **≥ 12 GB** (Engine cần ~6-8 GB khi load model)
- Disk trống: **≥ 20 GB** (image + model cache)

### Bước 1: Chuẩn bị file cấu hình

```bash
# Tạo file .env ở root (tùy chọn — có thể bỏ qua, docker-compose có default)
cat > .env << 'EOF'
# Gateway URL — đổi nếu AI Hub chạy ở port khác
GATEWAY_URL=http://host.docker.internal:4000
GATEWAY_KEY=sk-aihub-gateway-master

# Model HuggingFace (mặc định tải từ Hub, cần internet lần đầu)
CORRECT_MODEL_ID=protonx-models/protonx-legal-tc

# Tuỳ chỉnh OCR (CPU local: giảm xuống để chạy nhanh hơn)
OCR_IMAGE_SIZE=1024
OCR_DPI=150
EOF
```

### Bước 2: Build và chạy

```bash
# Build cả 2 image (lần đầu: ~15-20 phút)
docker compose build

# Chạy nền
docker compose up -d

# Xem log real-time
docker compose logs -f ocr-engine

# Kiểm tra health status
docker compose ps
```

> ⚠️ **Lưu ý**: Engine cần **1-3 phút** để load model sau khi container start. `ocr` service sẽ tự đợi engine ready (healthcheck).

### Bước 3: Kiểm tra

```bash
# Kiểm tra Engine
curl http://localhost:8002/ping
# → {"status":"ok","engine":"AI Inference Node"}

# Kiểm tra Orchestrator
curl http://localhost:8001/
# → {"status":"ok","service":"OCR Orchestrator Service"}

# Test OCR với file thực
curl -X POST http://localhost:8001/api/v1/ocr/process \
  -F "file=@/path/to/document.pdf"
```

### Bước 4: Dừng service

```bash
docker compose down          # Dừng, giữ volumes (model cache)
docker compose down -v       # Dừng + xoá volumes (sẽ tải lại model)
```

### Bật GPU (NVIDIA)

Bỏ comment phần `deploy.resources` trong `docker-compose.yml`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Sau đó thêm vào `.env`:
```
OCR_IMAGE_SIZE=1600
OCR_DPI=200
MAX_WORKERS=2
```

---

## 💻 Chạy local (không Docker)

### Engine

```bash
cd ocr-engine
cp .env.example .env
# Sửa .env theo môi trường của bạn

pip install -r requirements.txt
python main.py
# Engine chạy tại: http://localhost:8002
```

### Orchestrator

```bash
cd ocr
cp .env.example .env
# Đặt OCR_ENGINE_URL=http://localhost:8002

pip install -r requirements.txt
python main.py
# Orchestrator chạy tại: http://localhost:8001
```

---

## ☁️ Deploy SageMaker

> **Lưu ý**: Chỉ deploy `ocr-engine/` lên SageMaker. `ocr/` (Orchestrator) không cần thiết trên SageMaker.

### Bước 1: Upload model weights lên S3 (VPC không có internet)

Chạy trên **SageMaker Notebook** (có internet qua NAT Gateway):

```python
# prepare_model.py
import os, boto3, tarfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# ── Cấu hình ─────────────────────────────────────────────────────────────
S3_BUCKET  = "your-enterprise-bucket"        # ← đổi theo bucket của bạn
S3_PREFIX  = "ocr-service/model-artifacts"
MODEL_ID   = "protonx-models/protonx-legal-tc"
LOCAL_DIR  = "/tmp/ocr_model_weights"
# ─────────────────────────────────────────────────────────────────────────

print(f"[1/4] Downloading: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model     = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

print(f"[2/4] Saving to: {LOCAL_DIR}")
os.makedirs(LOCAL_DIR, exist_ok=True)
tokenizer.save_pretrained(LOCAL_DIR)
model.save_pretrained(LOCAL_DIR)

print("[3/4] Packaging to tar.gz...")
tar_path = "/tmp/model.tar.gz"
with tarfile.open(tar_path, "w:gz") as tar:
    tar.add(LOCAL_DIR, arcname="model_weights")

print(f"[4/4] Uploading to s3://{S3_BUCKET}/{S3_PREFIX}/model.tar.gz")
boto3.client("s3").upload_file(tar_path, S3_BUCKET, f"{S3_PREFIX}/model.tar.gz")
print("✅ Model artifact uploaded!")
```

### Bước 2: Tạo Dockerfile SageMaker

```dockerfile
# ocr-engine/Dockerfile.sagemaker
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 \
    libsm6 libxext6 libxrender1 curl \
    tesseract-ocr tesseract-ocr-vie poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

# SageMaker BẮT BUỘC port 8080
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    FLAGS_call_stack_level=0 \
    API_PORT=8080 \
    # SageMaker extract model.tar.gz vào đây tự động
    CORRECT_MODEL_ID=/opt/ml/model/model_weights

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8080/ping || exit 1

CMD ["python", "main.py"]
```

### Bước 3: Build & Push lên ECR

```python
# build_and_push.py
import boto3, subprocess

AWS_REGION    = "ap-southeast-1"    # ← đổi theo region của bạn
ECR_REPO      = "aihub-ocr-engine"
IMAGE_TAG     = "v2.0.0"
SOURCE_DIR    = "./ocr-engine"      # chạy từ root project

account_id = boto3.client("sts").get_caller_identity()["Account"]
ecr_uri    = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO}:{IMAGE_TAG}"
ecr_base   = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com"

# Tạo ECR repository
ecr = boto3.client("ecr", region_name=AWS_REGION)
try:
    ecr.create_repository(repositoryName=ECR_REPO)
    print("✅ ECR repo created")
except ecr.exceptions.RepositoryAlreadyExistsException:
    print("ℹ️  ECR repo exists")

# Login
subprocess.run(
    f"aws ecr get-login-password --region {AWS_REGION} | docker login --username AWS --password-stdin {ecr_base}",
    shell=True, check=True
)

# Build và Push
subprocess.run(
    ["docker", "build", "-f", f"{SOURCE_DIR}/Dockerfile.sagemaker", "-t", ecr_uri, SOURCE_DIR],
    check=True
)
subprocess.run(["docker", "push", ecr_uri], check=True)
print(f"✅ Pushed: {ecr_uri}")
```

### Bước 4: Tạo SageMaker Endpoint

```python
# deploy_endpoint.py
import boto3
from datetime import datetime

# ── Cấu hình ─────────────────────────────────────────────────────────────
AWS_REGION     = "ap-southeast-1"
S3_BUCKET      = "your-enterprise-bucket"
S3_PREFIX      = "ocr-service/model-artifacts"
ECR_IMAGE_URI  = "123456789.dkr.ecr.ap-southeast-1.amazonaws.com/aihub-ocr-engine:v2.0.0"
SAGEMAKER_ROLE = "arn:aws:iam::123456789:role/SageMakerExecutionRole"

# Chọn instance type theo đề xuất phần cứng ở trên
INSTANCE_TYPE  = "ml.g4dn.xlarge"   # GPU T4 — khuyến nghị

VPC_SUBNETS    = ["subnet-xxxxxxxx", "subnet-yyyyyyyy"]  # Private subnets
VPC_SGS        = ["sg-xxxxxxxx"]    # Security group
ENDPOINT_NAME  = "aihub-ocr-endpoint"
# ─────────────────────────────────────────────────────────────────────────

ts          = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name  = f"ocr-engine-{ts}"
config_name = f"ocr-config-{ts}"
sm          = boto3.client("sagemaker", region_name=AWS_REGION)

# 1. Tạo Model
print(f"[1/3] Creating model: {model_name}")
sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=SAGEMAKER_ROLE,
    PrimaryContainer={
        "Image":        ECR_IMAGE_URI,
        "ModelDataUrl": f"s3://{S3_BUCKET}/{S3_PREFIX}/model.tar.gz",
        "Environment": {
            "API_PORT":                              "8080",
            "CORRECT_MODEL_ID":                      "/opt/ml/model/model_weights",
            "OCR_IMAGE_SIZE":                        "1600",   # Điều chỉnh theo instance
            "OCR_DPI":                               "200",
            "MAX_PDF_PAGES":                         "100",
            "MAX_FILE_SIZE_MB":                      "50",
            "MAX_WORKERS":                           "2",      # GPU: 2, CPU: cpu_count
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONUNBUFFERED":                      "1",
        }
    },
    VpcConfig={
        "Subnets":          VPC_SUBNETS,
        "SecurityGroupIds": VPC_SGS,
    },
)

# 2. Tạo Endpoint Config
print(f"[2/3] Creating config: {config_name}")
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName":          "primary",
        "ModelName":            model_name,
        "InstanceType":         INSTANCE_TYPE,
        "InitialInstanceCount": 1,
        "InitialVariantWeight": 1.0,
    }],
    # Tăng timeout cho tài liệu nặng (PDF nhiều trang)
    # "ModelClientConfig": {"InvocationTimeoutInSeconds": 300}
)

# 3. Tạo hoặc Update Endpoint
print(f"[3/3] Deploying endpoint: {ENDPOINT_NAME}")
try:
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
except sm.exceptions.ResourceInUse:
    print("  ℹ️  Endpoint exists — updating...")
    sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

print("⏳ Waiting for InService... (~10-15 mins)")
sm.get_waiter("endpoint_in_service").wait(
    EndpointName=ENDPOINT_NAME,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40}
)
print(f"✅ Endpoint ready: {ENDPOINT_NAME}")
```

### Bước 5: Gọi API từ ứng dụng

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-1")

def ocr_document(file_path: str) -> str:
    """Gửi file PDF/image trực tiếp."""
    with open(file_path, "rb") as f:
        body = f.read()

    ext = file_path.lower().split(".")[-1]
    ct  = {"pdf": "application/pdf", "png": "image/png",
           "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")

    resp = runtime.invoke_endpoint(
        EndpointName="aihub-ocr-endpoint",
        ContentType=ct,
        Body=body,
    )
    return json.loads(resp["Body"].read())["markdown"]
```

---

## ✅ Checklist deploy

| # | Việc cần làm | Người thực hiện | Thời gian |
|---|---|---|---|
| 1 | Tạo `.env` từ `.env.example` co cả 2 service | Developer | 5 phút |
| 2 | `docker compose build && docker compose up -d` | Developer | 15-20 phút |
| 3 | Test endpoint local bằng `curl` hoặc `test_endpoints.py` | Developer | 10 phút |
| 4 | *(SageMaker)* Xin quyền IAM: ECR + SageMaker | IT Admin | 1-2 ngày |
| 5 | *(SageMaker)* Tạo VPC Endpoints cho SageMaker | Network Team | 0.5 ngày |
| 6 | *(SageMaker)* Chạy `prepare_model.py` → upload S3 | Developer | 30-60 phút |
| 7 | *(SageMaker)* Build `Dockerfile.sagemaker` → push ECR | Developer | 20-40 phút |
| 8 | *(SageMaker)* Chạy `deploy_endpoint.py` | Developer | 15 phút + 10-15 phút chờ |
| 9 | *(SageMaker)* Test API từ Notebook | Developer | 15 phút |

---

## 🔍 Troubleshooting

### Engine không start được

```bash
# Xem log detail
docker compose logs ocr-engine --tail=100

# Thường gặp:
# 1. OOM — tăng RAM Docker Desktop (≥ 12GB)
# 2. Model không tải được — kiểm tra internet hoặc MODEL_CACHE_DIR
# 3. Port conflict — đổi OCR_ENGINE_PORT trong .env
```

### SageMaker: Model không tải được (VPC)

> [!CAUTION]
> **KHÔNG** để `CORRECT_MODEL_ID=protonx-models/protonx-legal-tc` trong môi trường VPC. Container sẽ fail khi không có internet. Phải trỏ vào `/opt/ml/model/model_weights`.

### SageMaker: Request timeout

Thêm `ModelClientConfig` vào `create_endpoint_config`:

```python
"ModelClientConfig": {
    "InvocationTimeoutInSeconds": 300,   # 5 phút
    "InvocationMaxRetries": 1
}
```

### Cold start chậm (~3-8 phút)

Bình thường. Engine cần load PaddleOCR + HuggingFace model. Request đầu tiên sau deploy/restart sẽ chậm. Sau đó sẽ nhanh bình thường.

> [!TIP]
> **Tiết kiệm chi phí**: Dùng **SageMaker Batch Transform** thay Real-time Endpoint nếu OCR không cần 24/7 — chỉ tốn tiền khi đang xử lý.
