# ─── OCR AI Engine (Heavy ML Service) ────────────────────────────────────────
# Image nặng (~5-8GB): PaddleOCR + HuggingFace Transformers + OpenCV + Tesseract
FROM python:3.11-slim-bookworm

# Thiết lập môi trường để tránh tương tác khi cài đặt
ENV DEBIAN_FRONTEND=noninteractive

# Cài đặt các thư viện hệ thống cần thiết cho OpenCV, Docling và Paddle
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    tesseract-ocr \
    tesseract-ocr-vie \
    poppler-utils \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Copy requirements và cài đặt ─────────────────────────────────────────
# Lần đầu build: lâu (~10-20 phút) vì tải Paddle + Transformers + torch
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY . .

# ── Port: 8002 local, SageMaker yêu cầu 8080 (ghi đè qua ENV) ────────────
EXPOSE 8002

# ── Biến môi trường tối ưu cho ML production ─────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    # Tắt kiểm tra source để tăng tốc start-up
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    # Giảm log noise của Paddle
    FLAGS_call_stack_level=0 \
    # HuggingFace cache — mount volume vào đây để tránh tải lại model khi restart
    HF_HOME=/app/hf_cache \
    TRANSFORMERS_CACHE=/app/hf_cache

# ── Health Check: kiểm tra /ping sau khi model đã load xong ──────────────
# start-period=120s vì load PaddleOCR + HF model mất 1-3 phút
HEALTHCHECK --interval=30s --timeout=15s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:${API_PORT:-8002}/ping || exit 1

CMD ["python", "main.py"]
