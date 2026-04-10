import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator
from typing import Optional, Any


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

    # ── App & Server ────────────────────────────────────────────────────────
    app_name: str = "OCR AI Engine"
    # Local Docker: 8002 | SageMaker BẮT BUỘC: 8080 (ghi đè qua ENV)
    api_port: int = 8002
    # Số Uvicorn workers (None = 1 worker mặc định)
    workers: Optional[int] = None
    # Log level: DEBUG | INFO | WARNING | ERROR
    log_level: str = "INFO"

    # ── Process Limits ──────────────────────────────────────────────────────
    max_pdf_pages: int = 100
    max_file_size_mb: int = 50

    # ── ML Model ────────────────────────────────────────────────────────────
    # Local/HuggingFace Hub ID hoặc đường dẫn tuyệt đối khi deploy SageMaker
    # SageMaker: /opt/ml/model/model_weights
    correct_model_id: str = "protonx-models/protonx-legal-tc"
    # Thư mục cache cho HuggingFace (mount qua Docker volume để tránh tải lại)
    model_cache_dir: Optional[str] = None

    # ── OCR Image Processing ────────────────────────────────────────────────
    # Kích thước ảnh tối đa (pixel) trước khi đưa vào OCR — tăng = chính xác hơn nhưng chậm hơn
    ocr_image_size: int = 1600
    # DPI render PDF thành ảnh — 200 là cân bằng chất lượng/tốc độ tốt
    ocr_dpi: int = 200

    # ── Concurrent Control ──────────────────────────────────────────────────
    # Để trống (None) -> hệ thống tự detect:
    #   - GPU (CUDA): tối đa 2 luồng (tránh OOM VRAM)
    #   - CPU / MPS: số lượng CPU core vật lý
    max_workers: Optional[int] = None

    @field_validator("max_workers", "workers", "api_port", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: Any) -> Any:
        """Chuyển chuỗi rỗng (từ Docker env `VAR=`) → None thay vì lỗi parse int."""
        if isinstance(v, str) and v.strip() == "":
            return None
        return v

    def configure_hf_cache(self) -> None:
        """Thiết lập HuggingFace cache dir nếu được cấu hình."""
        if self.model_cache_dir:
            os.environ["HF_HOME"] = self.model_cache_dir
            os.environ["TRANSFORMERS_CACHE"] = self.model_cache_dir


settings = Settings()
settings.configure_hf_cache()
