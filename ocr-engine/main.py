import uvicorn
from app.api import app
from app.config import settings
from app.utils import logger

if __name__ == "__main__":
    logger.info(f"Khởi động OCR Engine trên cổng {settings.api_port}...")
    # Thường chạy trên 8002 cho local, 8080 cho SageMaker
    uvicorn.run(app, host="0.0.0.0", port=settings.api_port)
