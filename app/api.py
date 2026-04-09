import os
import tempfile
import base64
from urllib.parse import urlparse

import aiofiles

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel

from app.config import settings
from app.utils import logger, download_file_stream
from app.model import DocumentsOCRProcessor

# ========================================== #
# FASTAPI APP Initialization 
# ========================================== #
app = FastAPI(title=settings.app_name)

ocr_processor = None

@app.on_event("startup")
def load_model():
    global ocr_processor
    try:
        ocr_processor = DocumentsOCRProcessor()
        # logger.info(f"Khởi động thành công Server. Lắng nghe Port: {settings.api_port}")
    except Exception as e:
        logger.error(f"[!] Lỗi khi load mô hình: {e}")

# ========================================== #
# AWS SAGEMAKER INVOCATION ENDPOINTS
# ========================================== #

@app.get("/ping")
async def ping():
    """Endpoint bắt buộc cho AWS SageMaker Health Check."""
    if ocr_processor:
        return {"status": "ok", "service": "SageMaker OCR Worker"}
    return JSONResponse(status_code=503, content={"status": "Chưa tải xong Model"})

@app.post("/invocations")
async def invocations(request: Request):
    """
    Điểm trỏ SageMaker Endpoint chính:
    Hỗ trợ 2 Content-Types:
    1. application/json (body: {"data": "base64_string_here", "filename": "doc.pdf"})
    2. application/pdf, image/png, image/jpeg, ... (Trực tiếp Binary)
    """
    if not ocr_processor:
        raise HTTPException(status_code=500, detail="Mô hình OCR chưa sẵn sàng.")
    
    content_type = request.headers.get("Content-Type", "")
    tmp_path = None
    filename = "sagemaker_doc"

    try:
        # Nếu truyền thông qua JSON Base64
        if "application/json" in content_type:
            body = await request.json()
            b64_str = body.get("data", "")
            if not b64_str:
                 raise HTTPException(status_code=400, detail="Không tìm thấy 'data' chứa base64")

            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            raw_data = base64.b64decode(b64_str)
            filename = body.get("filename", "document.png")

        # Còn lại là gửi Raw Binary / File gốc (PDF, JPEG, ...)
        else:
            raw_data = await request.body()
            # Dự đoán đuôi mở rộng từ Content-Type
            extension = ""
            if "pdf" in content_type: extension = ".pdf"
            elif "png" in content_type: extension = ".png"
            elif "jpeg" in content_type or "jpg" in content_type: extension = ".jpg"
            filename = f"document{extension}"

        # Ghi payload vào thư mục tạm an toàn
        suffix = f".{filename.split('.')[-1]}" if '.' in filename else ""
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        with open(tmp_path, "wb") as f:
            f.write(raw_data)

        # Chạy dự đoán mà không làm thắt cổ chai Event Loop
        result_markdown = await run_in_threadpool(ocr_processor.process_file, tmp_path)
            
        return JSONResponse(status_code=200, content={
            "filename": filename,
            "status": "success",
            "markdown": result_markdown
        })
        
    except Exception as e:
        logger.error(f"Lỗi Invocation: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


# ========================================== #
# LOCAL COMPATIBILITY ENDPOINTS (Cũ)
# ========================================== #

class Base64OCRRequest(BaseModel):
    base64_data: str
    filename: str = "document.png"
    engine: str = "paddle"  # "paddle" hoặc "docling"

@app.get("/")
def check_health():
    return {"status": "ok", "service": settings.app_name}

@app.post("/api/v1/ocr/process")
async def process_document(file: UploadFile = File(...), engine: str = "paddle"):
    if not ocr_processor:
        raise HTTPException(status_code=500, detail="Mô hình rỗng.")
        
    tmp_path = None
    try:
        suffix = f".{file.filename.split('.')[-1]}" if '.' in file.filename else ""
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        async with aiofiles.open(tmp_path, 'wb') as f:
            while True:
                chunk = await file.read(1024 * 1024)  # Đọc 1MB mỗi lần
                if not chunk:
                    break
                await f.write(chunk)

        # Non-blocking call
        m_down = await run_in_threadpool(ocr_processor.process_file, tmp_path, engine=engine)
        return JSONResponse(status_code=200, content={"filename": file.filename, "status": "success", "markdown": m_down})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.get("/api/v1/ocr/process")
async def process_document_url(url: str, engine: str = "paddle"):
    if not ocr_processor:
        raise HTTPException(status_code=500, detail="Mô hình rỗng")
        
    tmp_path = None
    try:
        parsed_url = urlparse(url)
        filename = os.path.basename(parsed_url.path) or "downloaded_image.jpg"
        suffix = f".{filename.split('.')[-1]}" if '.' in filename else ""
        
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)

        # Sử dụng Chunk Downloader stream an toàn từ utils
        await download_file_stream(url, tmp_path)

        m_down = await run_in_threadpool(ocr_processor.process_file, tmp_path)
        return {"filename": filename, "status": "success", "markdown": m_down}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/api/v1/ocr/process/base64")
async def process_document_base64(request: Base64OCRRequest):
    if not ocr_processor:
        raise HTTPException(status_code=500, detail="Mô hình rỗng")
        
    tmp_path = None
    try:
        b64_str = request.base64_data
        if "," in b64_str:
            b64_str = b64_str.split(",")[1]
            
        content = base64.b64decode(b64_str)
        filename = request.filename
        suffix = f".{filename.split('.')[-1]}" if '.' in filename else ""
        
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        with open(tmp_path, "wb") as f:
            f.write(content)

        m_down = await run_in_threadpool(ocr_processor.process_file, tmp_path, engine=request.engine)
        return {"filename": filename, "status": "success", "markdown": m_down}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
