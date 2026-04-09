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
    # SageMaker yêu cầu port 8080; chạy local có thể đổi sang 8001 qua .env
    api_port: int = 8080

    # Process Limits
    max_pdf_pages: int = 100
    max_file_size_mb: int = 50

    # ML Model
    correct_model_id: str = "protonx-models/protonx-legal-tc"

    # Concurrent Control
    # Để trống (None) -> hệ thống tự detect:
    #   - GPU (CUDA): tối đa 2 luồng (tránh OOM VRAM)
    #   - CPU / MPS: số lượng CPU core vật lý
    max_workers: Optional[int] = None


settings = Settings()

