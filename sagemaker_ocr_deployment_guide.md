# 🚀 Hướng dẫn Deploy OCR Service lên AWS SageMaker (VPC Enterprise)

> **Môi trường**: SageMaker Notebook + S3 + VPC (không có internet trực tiếp)
> **Service**: `services/ocr-engine` — Chứa PaddleOCR + HuggingFace Correction Model
> **Target**: SageMaker Real-time Endpoint, có thể call API từ nội bộ VPC

---

## 📋 Tổng quan luồng deploy

```
[Local Machine]           [SageMaker Notebook]           [AWS]
services/ocr-engine/  →  Build Docker Image         →  ECR (image registry)
                      →  Download & Upload weights   →  S3 (model artifact)
                                                     →  SageMaker Endpoint
                                                     →  Call API từ VPC
```

> [!IMPORTANT]
> **Tại sao chỉ dùng `ocr-engine` mà không dùng `ocr` (orchestrator)?**
> `services/ocr-engine` đã có đầy đủ `/ping` + `/invocations` theo chuẩn SageMaker và chứa toàn bộ logic ML. `services/ocr` (orchestrator) chỉ là proxy wrapper phục vụ local Docker Compose — **không cần thiết trên SageMaker**.

---

## 🔴 ĐIỀU KIỆN TIÊN QUYẾT

Yêu cầu quyền IAM tối thiểu trong môi trường enterprise:

| Dịch vụ | Quyền cần có |
|--------|-------------|
| **ECR** | `ecr:GetAuthorizationToken`, `ecr:CreateRepository`, `ecr:BatchCheckLayerAvailability`, `ecr:PutImage`, `ecr:InitiateLayerUpload`, `ecr:UploadLayerPart`, `ecr:CompleteLayerUpload` |
| **S3** | `s3:PutObject`, `s3:GetObject`, `s3:ListBucket` |
| **SageMaker** | `sagemaker:CreateModel`, `sagemaker:CreateEndpointConfig`, `sagemaker:CreateEndpoint`, `sagemaker:InvokeEndpoint` |
| **IAM Role** | SageMaker Execution Role có `AmazonSageMakerFullAccess` + `AmazonS3ReadOnlyAccess` |

---

## BƯỚC 1: Chuẩn bị Model Artifact → Upload S3

> [!WARNING]
> **Vấn đề VPC**: Trong môi trường VPC không có internet, SageMaker instance sẽ **không thể download** model `protonx-models/protonx-legal-tc` từ HuggingFace Hub khi khởi động. Bạn phải bake model weights vào S3 trước.

### 1.1 — Tải model weights về và upload S3

Chạy trên **SageMaker Notebook** (Notebook thường có internet qua NAT Gateway):

```python
# prepare_model.py — Chạy trên SageMaker Notebook
import os, boto3, tarfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# =============================================
# CẤU HÌNH — Thay đổi theo môi trường của bạn
S3_BUCKET  = "your-enterprise-bucket"
S3_PREFIX  = "ocr-service/model-artifacts"
MODEL_ID   = "protonx-models/protonx-legal-tc"
LOCAL_DIR  = "/tmp/ocr_model_weights"
# =============================================

print(f"[1/4] Downloading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_ID)

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

---

## BƯỚC 2: Sửa code cho SageMaker-ready

### 2.1 — Sửa `services/ocr-engine/app/config.py`

```python
# app/config.py — Phiên bản SageMaker-ready
import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", case_sensitive=False)

    app_name:  str = "Enterprise OCR Engine"
    api_port:  int = 8080  # SageMaker BẮT BUỘC port 8080

    max_pdf_pages:    int = 100
    max_file_size_mb: int = 50

    # Trỏ đến /opt/ml/model — SageMaker tự extract model.tar.gz vào đây
    correct_model_id: str = "/opt/ml/model/model_weights"

    max_workers: Optional[int] = None

settings = Settings()
```

### 2.2 — Tạo `Dockerfile.sagemaker`

```dockerfile
# services/ocr-engine/Dockerfile.sagemaker
FROM python:3.11-slim-bookworm

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 libgomp1 \
    libsm6 libxext6 libxrender1 \
    tesseract-ocr tesseract-ocr-vie poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/program

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY main.py .

# BẮT BUỘC cho SageMaker
ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
ENV API_PORT=8080
ENV CORRECT_MODEL_ID=/opt/ml/model/model_weights

EXPOSE 8080

CMD ["python", "main.py"]
```

---

## BƯỚC 3: Build & Push Docker Image lên ECR

```python
# build_and_push.py — Chạy trên SageMaker Notebook terminal
import boto3, subprocess

AWS_REGION     = "ap-southeast-1"
ECR_REPO_NAME  = "enterprise-ocr-engine"
IMAGE_TAG      = "v1.0.0"
SOURCE_DIR     = "/home/ec2-user/SageMaker/services/ocr-engine"

account_id = boto3.client("sts").get_caller_identity()["Account"]
ecr_uri    = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPO_NAME}:{IMAGE_TAG}"

# Tạo ECR repo
ecr = boto3.client("ecr", region_name=AWS_REGION)
try:
    ecr.create_repository(repositoryName=ECR_REPO_NAME)
    print("✅ ECR repo created")
except ecr.exceptions.RepositoryAlreadyExistsException:
    print("ℹ️  ECR repo exists")

# Login Docker vào ECR
login_cmd = f"aws ecr get-login-password --region {AWS_REGION} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com"
subprocess.run(login_cmd, shell=True, check=True)

# Build image
subprocess.run(["docker", "build", "-f", f"{SOURCE_DIR}/Dockerfile.sagemaker", "-t", ecr_uri, SOURCE_DIR], check=True)

# Push lên ECR
subprocess.run(["docker", "push", ecr_uri], check=True)
print(f"✅ Image pushed: {ecr_uri}")
```

---

## BƯỚC 4: Tạo SageMaker Endpoint

```python
# deploy_endpoint.py — Chạy trên SageMaker Notebook
import boto3
from datetime import datetime

# =============================================
# CẤU HÌNH
AWS_REGION           = "ap-southeast-1"
S3_BUCKET            = "your-enterprise-bucket"
S3_MODEL_PREFIX      = "ocr-service/model-artifacts"
ECR_IMAGE_URI        = "123456789.dkr.ecr.ap-southeast-1.amazonaws.com/enterprise-ocr-engine:v1.0.0"
SAGEMAKER_ROLE       = "arn:aws:iam::123456789:role/SageMakerExecutionRole"
INSTANCE_TYPE        = "ml.g4dn.xlarge"   # GPU (khuyến nghị) | hoặc ml.c5.2xlarge CPU
VPC_SUBNET_IDS       = ["subnet-xxxxxxxxx", "subnet-yyyyyyyyy"]  # Private subnets
VPC_SECURITY_GROUPS  = ["sg-xxxxxxxxx"]   # Security group của SageMaker
ENDPOINT_NAME        = "enterprise-ocr-endpoint"
# =============================================

ts = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name  = f"ocr-engine-{ts}"
config_name = f"ocr-config-{ts}"

sm = boto3.client("sagemaker", region_name=AWS_REGION)

# 1. Tạo Model
print(f"[1/3] Creating model: {model_name}")
sm.create_model(
    ModelName=model_name,
    ExecutionRoleArn=SAGEMAKER_ROLE,
    PrimaryContainer={
        "Image":        ECR_IMAGE_URI,
        "ModelDataUrl": f"s3://{S3_BUCKET}/{S3_MODEL_PREFIX}/model.tar.gz",
        "Environment": {
            "API_PORT":                              "8080",
            "CORRECT_MODEL_ID":                      "/opt/ml/model/model_weights",
            "MAX_PDF_PAGES":                         "100",
            "MAX_FILE_SIZE_MB":                      "50",
            "MAX_WORKERS":                           "2",
            "PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK": "True",
            "PYTHONUNBUFFERED":                      "1",
        }
    },
    VpcConfig={  # BẮT BUỘC trong enterprise VPC
        "Subnets":          VPC_SUBNET_IDS,
        "SecurityGroupIds": VPC_SECURITY_GROUPS,
    },
)

# 2. Tạo Endpoint Config
print(f"[2/3] Creating endpoint config: {config_name}")
sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[{
        "VariantName":          "primary",
        "ModelName":            model_name,
        "InstanceType":         INSTANCE_TYPE,
        "InitialInstanceCount": 1,
        "InitialVariantWeight": 1.0,
    }],
)

# 3. Tạo hoặc Update Endpoint
print(f"[3/3] Creating endpoint: {ENDPOINT_NAME}")
try:
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)
except sm.exceptions.ResourceInUse:
    print("  ℹ️  Endpoint exists, updating...")
    sm.update_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=config_name)

# Chờ sẵn sàng (~10-15 phút)
print("⏳ Waiting for InService... (~10-15 mins)")
sm.get_waiter("endpoint_in_service").wait(
    EndpointName=ENDPOINT_NAME,
    WaiterConfig={"Delay": 30, "MaxAttempts": 40}
)
print(f"✅ Endpoint ready: {ENDPOINT_NAME}")
```

---

## BƯỚC 5: Cấu hình VPC (Yêu cầu Network/Admin Team)

### Security Group Rules cần thiết

```
Security Group của SageMaker Endpoint:
  Inbound:
    - Port 8080 TCP  ← từ SG của AI Hub service (để invoke endpoint)
  
  Outbound:
    - Port 443 HTTPS → S3 VPC Endpoint (download model.tar.gz)
    - Port 443 HTTPS → ECR VPC Endpoint (pull Docker image)
```

### VPC Endpoints cần tạo (nếu chưa có)

```bash
# Kiểm tra VPC Endpoints hiện tại
aws ec2 describe-vpc-endpoints \
  --filters "Name=service-name,Values=*sagemaker*" \
  --query "VpcEndpoints[*].{Service:ServiceName,State:State}" \
  --output table

# Cần có 2 endpoints:
# - com.amazonaws.REGION.sagemaker.runtime  (để InvokeEndpoint)
# - com.amazonaws.REGION.sagemaker.api      (để quản lý)
```

---

## BƯỚC 6: Gọi API từ ứng dụng

### Cách 1 — Gọi qua boto3 (Khuyến nghị, từ trong VPC)

```python
import boto3, json

runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-1")

# Gửi file PDF/image binary trực tiếp
def ocr_document(file_path: str) -> str:
    with open(file_path, "rb") as f:
        body = f.read()
    
    ext = file_path.lower().split(".")[-1]
    ct  = {"pdf": "application/pdf", "png": "image/png",
           "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
    
    resp = runtime.invoke_endpoint(
        EndpointName="enterprise-ocr-endpoint",
        ContentType=ct,
        Body=body,
    )
    return json.loads(resp["Body"].read())["markdown"]

# Gửi base64 JSON
def ocr_base64(b64_str: str, filename: str = "doc.pdf") -> str:
    resp = runtime.invoke_endpoint(
        EndpointName="enterprise-ocr-endpoint",
        ContentType="application/json",
        Body=json.dumps({"data": b64_str, "filename": filename}).encode(),
    )
    return json.loads(resp["Body"].read())["markdown"]
```

### Cách 2 — Tích hợp vào AI Hub `ocr_routes.py`

```python
# Thay OCR_SERVICE_URL bằng boto3 SageMaker Runtime call
import boto3, json
from fastapi import APIRouter, UploadFile, File, HTTPException

router   = APIRouter()
_runtime = boto3.client("sagemaker-runtime", region_name="ap-southeast-1")
ENDPOINT = "enterprise-ocr-endpoint"

@router.post("/api/v1/ocr/process")
async def process_document(file: UploadFile = File(...)):
    try:
        body = await file.read()
        ext  = file.filename.lower().rsplit(".", 1)[-1] if "." in file.filename else ""
        ct   = {"pdf": "application/pdf", "png": "image/png",
                "jpg": "image/jpeg", "jpeg": "image/jpeg"}.get(ext, "application/octet-stream")
        
        resp   = _runtime.invoke_endpoint(EndpointName=ENDPOINT, ContentType=ct, Body=body)
        result = json.loads(resp["Body"].read())
        return {"filename": file.filename, "status": "success", "markdown": result["markdown"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

---

## 📊 Checklist Deploy Tổng hợp

| # | Việc cần làm | Ai làm | Thời gian |
|---|---|---|---|
| 1 | Xin quyền IAM: ECR + SageMaker | IT Admin | 1-2 ngày |
| 2 | Tạo VPC Endpoints cho SageMaker Runtime & API | Network Team | 0.5 ngày |
| 3 | Upload code `ocr-engine` lên SageMaker Notebook | Developer | 15 phút |
| 4 | Sửa `config.py` + tạo `Dockerfile.sagemaker` | Developer | 15 phút |
| 5 | Chạy `prepare_model.py` → Upload weights lên S3 | Developer | 30-60 phút |
| 6 | Chạy `build_and_push.py` → Build & Push ECR | Developer | 20-40 phút |
| 7 | Chạy `deploy_endpoint.py` → Tạo SageMaker Endpoint | Developer | 15 phút chờ 10-15 phút |
| 8 | Test API call từ Notebook | Developer | 15 phút |
| 9 | Tích hợp vào AI Hub `ocr_routes.py` | Developer | 30 phút |

---

## ⚠️ Lưu ý quan trọng

> [!CAUTION]
> **Không để `CORRECT_MODEL_ID=protonx-models/protonx-legal-tc`** trong môi trường VPC — container sẽ fail khi không có internet. Phải trỏ vào `/opt/ml/model/model_weights`.

> [!WARNING]
> **Cold Start ~3-8 phút**: Lần đầu hoặc sau khi update endpoint, container cần giải nén `model.tar.gz` và load toàn bộ PaddleOCR + HuggingFace model. Request đầu tiên sẽ timeout nếu không tăng thời gian chờ.

> [!TIP]
> **Tăng Invocation Timeout**: Thêm vào `create_endpoint_config`:
> ```python
> "ModelClientConfig": {"InvocationTimeoutInSeconds": 300, "InvocationMaxRetries": 1}
> ```

> [!NOTE]
> **Tiết kiệm chi phí**: Nếu OCR không cần 24/7, dùng **SageMaker Batch Transform** thay vì Real-time Endpoint — chỉ tốn tiền khi đang xử lý, không tốn tiền idle.
