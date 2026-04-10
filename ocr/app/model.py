import os
import base64
import httpx
import time
from typing import Optional
from app.config import settings
from app.utils import logger

class DocumentsOCRProcessor:
    def __init__(self):
        # Orchestrator gọi qua AI Hub Gateway để logging FinOps
        self.gateway_url = settings.get_gateway_url()
        self.engine_url = settings.get_engine_url()
        logger.info(f"[+] Orchestrator ready. Gateway: {self.gateway_url}")

    def _gateway_chat(self, model: str, messages: list):
        """Helper to call OCR Engine via AI Hub LiteLLM Gateway."""
        url = f"{self.gateway_url.rstrip('/')}/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.gateway_key}"
        }
        try:
            # Tăng timeout lên 30 phút (1800s) cho các mô hình quét nặng trên CPU
            with httpx.Client(timeout=1800.0) as client:
                resp = client.post(
                    url,
                    headers=headers,
                    json={
                        "model": model,
                        "messages": messages,
                        "temperature": 0.0
                    }
                )
                resp.raise_for_status()
                result = resp.json()
                return result["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Gateway error (OCR Engine): {e}")
            return None

    def process_file(self, file_path: str, engine: str = "paddle") -> str:
        """
        Gửi toàn bộ file qua Gateway để Engine xử lý.
        engine: "paddle" -> sử dụng luồng PaddleOCR cũ.
        engine: "docling" -> sử dụng luồng Docling thông minh.
        """
        start_time = time.time()
        try:
            # 1. Đọc file và chuyển sang Base64
            with open(file_path, "rb") as f:
                b64_data = base64.b64encode(f.read()).decode("utf-8")
            
            # 2. Xác định model tương ứng trong Gateway
            model_id = "paddle-ocr-vl" if engine == "paddle" else "docling-parser"
            
            logger.info(f"[*] Sending request to Gateway using model: {model_id} (Engine: {engine})")

            # 3. Gọi Gateway (Chuẩn Vision API để tính token)
            result_markdown = self._gateway_chat(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract content from this document and return as Markdown."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:application/pdf;base64,{b64_data}"} # PDF or Image
                            }
                        ]
                    }
                ]
            )


            if not result_markdown:
                return "Error: Gateway returned empty result or connection failed."

            logger.info(f"[+] Completed in {time.time() - start_time:.2f}s")
            return result_markdown

        except Exception as e:
            logger.error(f"Error in Orchestrator: {e}")
            return f"Error: {str(e)}"
