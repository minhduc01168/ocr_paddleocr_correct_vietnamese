import logging
import sys
import os
import aiofiles
import httpx
from app.config import settings

# Cấu hình logging tĩnh (Singleton Logger)
def setup_logger(name="ocr_service"):
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # StreamHandler cho console / terminal (SageMaker CloudWatch capture được stdout, stderr)
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
    return logger

logger = setup_logger()

async def download_file_stream(url: str, dest_path: str) -> bool:
    """
    Stream download file để giới hạn RAM. Lỗi nếu kích thước vượt Max Size (VD 50MB).
    """
    max_bytes = settings.max_file_size_mb * 1024 * 1024
    downloaded = 0
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream("GET", url) as response:
                response.raise_for_status()
                # logger.info(f"Bắt đầu tải file: {url}")
                async with aiofiles.open(dest_path, "wb") as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        downloaded += len(chunk)
                        if downloaded > max_bytes:
                            raise Exception(f"File tải xuống vượt quá giới hạn an toàn {settings.max_file_size_mb}MB.")
                        await f.write(chunk)
                # logger.info(f"Tải thành công: {dest_path}")
                return True
    except Exception as e:
        logger.error(f"Lỗi khi Stream Download URL {url}: {e}")
        # Dọn dẹp file dở dang
        if os.path.exists(dest_path):
            os.remove(dest_path)
        raise e
