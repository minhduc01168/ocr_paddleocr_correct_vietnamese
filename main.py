import sys
import os

# Đảm bảo app/ module nằm trong system path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn
from app.config import settings
from app.api import app

if __name__ == "__main__":
    # Start server based on config settings
    # Nếu chạy trên Amazon SageMaker, hệ thống sẽ tự tìm thấy cổng API_PORT (Thường là 8080)
    uvicorn.run("app.api:app", host="0.0.0.0", port=settings.api_port, reload=True)
