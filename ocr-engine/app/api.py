import os
import base64
import tempfile
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Any

from app.engine import engine
from app.utils import logger

app = FastAPI(title="OCR AI Engine (Backend)")

class MessageContent(BaseModel):
    type: str
    text: Optional[str] = None
    image_url: Optional[dict] = None

class Message(BaseModel):
    role: str
    content: Any # Can be string or list of MessageContent

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.3

@app.get("/ping")
async def ping():
    return {"status": "ok", "engine": "AI Inference Node"}

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible endpoint to handle vision/OCR tasks.
    """
    model = request.model
    last_msg = request.messages[-1]
    
    # 1. Trích xuất nội dung (Text + Image)
    b64_data = None
    if isinstance(last_msg.content, list):
        for item in last_msg.content:
            if isinstance(item, dict) and item.get("type") == "image_url":
                url = item.get("image_url", {}).get("url", "")
                if url.startswith("data:"):
                    b64_data = url.split(",")[1]
            elif hasattr(item, "type") and item.type == "image_url":
                url = item.image_url.get("url", "") if isinstance(item.image_url, dict) else item.image_url.url
                if url.startswith("data:"):
                    b64_data = url.split(",")[1]
    
    if not b64_data:
        raise HTTPException(status_code=400, detail="Không tìm thấy dữ liệu hình ảnh/tài liệu trong request.")

    # 2. Lưu tạm file với nhận diện định dạng (Magic Bytes)
    tmp_path = None
    try:
        raw_bytes = base64.b64decode(b64_data)
        
        # Nhận diện định dạng thực tế của file thay vì gán cứng
        if raw_bytes.startswith(b'%PDF'):
            suffix = ".pdf"
        elif raw_bytes.startswith(b'\xff\xd8\xff'):
            suffix = ".jpg"
        elif raw_bytes.startswith(b'\x89PNG\r\n\x1a\n'):
            suffix = ".png"
        else:
            # Fallback theo model nếu không nhận diện được magic bytes
            suffix = ".pdf" if "docling" in model else ".jpg"
            
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        
        with open(tmp_path, "wb") as f:
            f.write(raw_bytes)

        # 3. Định tuyến luồng xử lý
        if model == "paddle-ocr-vl":
            result_markdown = engine.paddle_ocr_flow(tmp_path)
        elif model == "docling-parser":
            result_markdown = engine.docling_flow(tmp_path)
        else:
            raise HTTPException(status_code=404, detail=f"Model {model} không được hỗ trợ bởi Engine này.")

        # 4. Trả về format OpenAI chuẩn
        if not result_markdown:
            logger.warning(f"Không tìm thấy trang nào hoặc không đọc được file: {tmp_path}")
            return {"error": "Could not read document."}

        return {
            "id": f"ocr-{os.urandom(4).hex()}",
            "object": "chat.completion",
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": result_markdown
                    },
                    "finish_reason": "stop"
                }
            ]
        }

    except Exception as e:
        logger.error(f"Error in Engine execution: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
