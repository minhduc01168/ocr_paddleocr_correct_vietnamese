# ─── Orchestrator (Proxy nhẹ) ─────────────────────────────────────────────
# Chỉ cần Python stdlib + HTTP client. Image rất nhẹ (~200MB sau build).
FROM python:3.11-slim

WORKDIR /app

# Chỉ cài đặt các thư viện HTTP và API cơ bản
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# ── Port Orchestrator (8001 local, đổi API_PORT=8080 khi SageMaker) ──────
EXPOSE 8001

# ── Biến môi trường tối ưu cho production ────────────────────────────────
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# ── Health Check: Docker Swarm/ECS/K8s biết khi nào service sẵn sàng ────
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${API_PORT:-8001}/')" || exit 1

CMD ["python", "main.py"]
