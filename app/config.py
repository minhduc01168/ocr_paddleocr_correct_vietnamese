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

    # API & SageMaker Settings
    app_name: str = "Enhanced Local OCR & SageMaker Service"
    # Mặc định local là 8001; SageMaker yêu cầu 8080 (sẽ ghi đè qua ENV trong Docker)
    api_port: int = 8001

    # Process Limits
    max_pdf_pages: int = 100
    max_file_size_mb: int = 50

    # ML Model
    correct_model_id: str = "protonx-models/protonx-legal-tc"

    # Gateway / LiteLLM Configuration
    # Mặc định local là localhost:4000; Trong Docker sẽ ghi đè thành host.docker.internal:4000
    gateway_url: str = "http://localhost:4000"
    gateway_key: str = "sk-aihub-gateway-master"

    # Concurrent Control
    # Để trống (None) -> hệ thống tự detect:
    #   - GPU (CUDA): tối đa 2 luồng (tránh OOM VRAM)
    #   - CPU / MPS: số lượng CPU core vật lý
    max_workers: Optional[int] = None

    def get_gateway_url(self) -> str:
        """Tự động chuyển đổi host.docker.internal thành localhost trên macOS chạy trực tiếp."""
        import sys
        url = self.gateway_url
        if "host.docker.internal" in url and sys.platform == "darwin":
            return url.replace("host.docker.internal", "localhost")
        return url


settings = Settings()

