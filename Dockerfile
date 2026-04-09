# Lớp Orchestrator - Siêu nhẹ (Chỉ là Client gọi Gateway)
FROM python:3.11-slim

WORKDIR /app

# Chỉ cài đặt các thư viện HTTP và API cơ bản
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Port ổn định cho Orchestrator
EXPOSE 8001

ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py"]
