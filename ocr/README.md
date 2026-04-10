# ── OCR Service ──────────────────────────────────────────────────────────
Dịch vụ OCR đa năng, được thiết kế chạy dưới dạng **Microservice** độc lập, hỗ trợ tốt 2 môi trường: **Local Docker** và **AWS SageMaker Endpoint (BYOC)**.

## Kiến trúc Thư mục mới
Sau khi tái cấu trúc vào `services/ocr`, kiến trúc hiện tại như sau:
```text
services/ocr/
├── main.py                 # Entrypoint khởi động Uvicorn
├── app/                    # <--- Core logic (Renamed from src)
│   ├── config.py           # Quản lý cấu hình pydantic-settings
│   ├── utils.py            # Logger & tiện ích tải file
│   ├── model.py            # OCR Processor (Gọi Gateway cho Inference)
│   └── api.py              # FastAPI Endpoints (Local & SageMaker)
├── Dockerfile              # Container Image cho SageMaker
├── Makefile                # Công cụ build/run/deploy nhanh
└── requirements.txt
```

## Các lệnh nhanh (Makefile)
Chúng tôi đã cung cấp `Makefile` để đơn giản hóa quá trình vận hành:

| Lệnh | Mô tả |
|------|-------|
| `make build` | Build Docker image nội bộ |
| `make run` | Chạy container tại port **8001** (Khớp với Project) |
| `make stop` | Dừng và xóa container |
| `make logs` | Xem log từ container đang chạy |
| `make push` | Đẩy image lên Amazon ECR (Cần cấu hình AWS CLI) |

## Deploy lên AWS SageMaker (BYOC) - Chi tiết

Để triển khai dịch vụ này lên AWS SageMaker dưới dạng "Bring Your Own Container", hãy làm theo các bước sau:

### 1. Đẩy code nào lên AWS?
Khi triển khai qua Docker, bạn **KHÔNG** cần đẩy toàn bộ Project. Docker sẽ chỉ đóng gói các file nằm trong thư mục `services/ocr/`. Các file quan trọng nhất gồm:
- `app/`: Toàn bộ logic xử lý.
- `main.py`: Entrypoint của server.
- `Dockerfile`: Chỉ dẫn cho AWS cách chạy container.
- `requirements.txt`: Các thư viện cần cài đặt.

### 2. Quy trình đẩy Image lên ECR
1.  Tạo Repository trên Amazon ECR (VD: `aihub-ocr-service`).
2.  Mở [Makefile](file:///Users/minhduc168/APW/develop/enterprise-ai-platform/services/ocr/Makefile), cập nhật `ECR_REPO_URL` với link ECR của bạn.
3.  Chạy lệnh: `make build` rồi `make push`.

### 3. Cấu hình trên SageMaker Console
Khi tạo **SageMaker Model** và **Endpoint**:
- **Image Artifact**: Chọn link image ECR bạn vừa push.
- **Port**: Thiết lập **Container Port** là `8080` (AWS mặc định).
- **Environment Variables**:
  - `API_PORT=8080` (Bắt buộc để SageMaker nhận diện).
  - `GATEWAY_URL`: URL tới LiteLLM Gateway của bạn.
  - `LITELLM_MASTER_KEY`: Key xác thực Gateway.

## Tại sao sử dụng Port 8001 và 8080?
- **8001 (Local):** Để đồng nhất với các cấu hình cũ của Project (Gateway, Flow Nodes, Tools).
- **8080 (Production/AWS):** Đây là cổng mặc định mà AWS SageMaker luôn kỳ vọng để thực hiện Health Check (`/ping`) và Inference (`/invocations`). Hệ thống đã được cấu hình để tự động chuyển đổi giữa hai cổng này tùy môi trường.

## Kiểm tra tự động
```bash
# Test health check local (8001)
python test_endpoints.py --host http://localhost:8001

# Test đầy đủ với file PDF
python test_endpoints.py --host http://localhost:8001 --pdf path/to/test.pdf
```
