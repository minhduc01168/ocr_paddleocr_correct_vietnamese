# Lớp Engine - Chứa toàn bộ logic ML nặng (PaddleOCR, Docling, Transformers)
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
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements và cài đặt
# Lưu ý: Lần đầu build sẽ lâu vì tải Paddle và Transformers
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy mã nguồn
COPY . .

# Port mặc định cho AI Engine
EXPOSE 8002

# Biến môi trường để tối ưu luồng xử lý
ENV PYTHONUNBUFFERED=1
ENV PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

CMD ["python", "main.py"]
