import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """
    Cấu hình ứng dụng được quản lý bởi pydantic-settings.

    Thứ tự ưu tiên đọc giá trị (từ cao -> thấp):
    1. Biến môi trường hệ thống (Environment Variables) - Dùng cho SageMaker/Docker
    2. File .env tại thư mục gốc dự án - Dùng khi phát triển local
    3. Giá trị mặc định được khai báo dưới đây
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        # Không lỗi nếu thiếu file .env (SageMaker không có file này)
        extra="ignore",
        case_sensitive=False,
    )

    # ── App & Server ───────────────────────────────────────────────────────
    app_name: str = "OCR Orchestrator Service"
    # Mặc định local là 8001; SageMaker yêu cầu 8080 (ghi đè qua ENV trong Docker)
    api_port: int = 8001
    # Số Uvicorn workers (None = 1 worker; tăng lên 2-4 cho production)
    workers: Optional[int] = None
    # Log level: DEBUG | INFO | WARNING | ERROR
    log_level: str = "INFO"

    # ── Process Limits ─────────────────────────────────────────────────────
    max_pdf_pages: int = 100
    max_file_size_mb: int = 50

    # ── Gateway / LiteLLM Configuration ───────────────────────────────────
    # Local dev: Gateway chạy trong docker compose của AI Hub
    # Docker: đổi sang host.docker.internal:4000
    # Production SageMaker: đổi sang URL public/internal của AI Hub Gateway
    gateway_url: str = "http://localhost:4000"
    gateway_key: str = "sk-aihub-gateway-master"

    # ── OCR Engine Internal URL ────────────────────────────────────────────
    # Orchestrator gọi Engine này để xử lý OCR nặng (PaddleOCR + HuggingFace)
    # Local dev : http://localhost:8002
    # Docker Compose: http://ocr-engine:8002  (dùng tên service Docker)
    ocr_engine_url: str = "http://localhost:8002"

    # ── Concurrent Control ─────────────────────────────────────────────────
    max_workers: Optional[int] = None

    def get_gateway_url(self) -> str:
        """Tự động chuyển đổi host.docker.internal thành localhost trên macOS chạy trực tiếp."""
        import sys
        url = self.gateway_url
        if "host.docker.internal" in url and sys.platform == "darwin":
            return url.replace("host.docker.internal", "localhost")
        return url

    def get_engine_url(self) -> str:
        """Tự động chuyển đổi tên service Docker thành localhost khi chạy trực tiếp."""
        import sys
        url = self.ocr_engine_url
        if "ocr-engine" in url and sys.platform == "darwin":
            return url.replace("ocr-engine", "localhost")
        return url


settings = Settings()
