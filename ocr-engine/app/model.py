import os
import io
import time
import base64
import tempfile
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
from PIL import Image

import torch
from paddleocr import PaddleOCRVL
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from concurrent.futures import ThreadPoolExecutor, as_completed

from app.config import settings
from app.utils import logger

# Tắt các log không cần thiết của Paddle
os.environ['PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK'] = 'True'
os.environ['PADDLE_TF_CPP_MIN_LOG_LEVEL'] = '3'

class DocumentsOCRProcessor:
    def __init__(self):
        # 0. Thiết lập thiết bị ưu tiên theo thứ tự: CUDA > MPS > CPU
        if torch.cuda.is_available():
            self.device = "cuda"
            # GPU: max workers giới hạn tối đa 2 để tránh Out of Memory
            default_workers = 2
        elif torch.backends.mps.is_available():
            self.device = "mps"
            default_workers = os.cpu_count() or 4
        else:
            self.device = "cpu"
            default_workers = os.cpu_count() or 4

        self.size_input = settings.ocr_image_size
        self.max_page_pdf = settings.max_pdf_pages
        self.dpi = settings.ocr_dpi
        
        # Nhận diện workers từ file cấu hình .env, fallback vào Hardware
        self.max_workers = settings.max_workers if settings.max_workers else default_workers
        
        # 1. OCR Components
        # logger.info("[*] Đang khởi tạo PaddleOCRVL...")
        self.ocr_pipeline = PaddleOCRVL() 

        # 2. Correction Components
        # logger.info(f"[*] Đang tải Model sửa lỗi: {settings.correct_model_id}...")
        self.correct_tokenizer = AutoTokenizer.from_pretrained(settings.correct_model_id)
        self.correct_model = AutoModelForSeq2SeqLM.from_pretrained(settings.correct_model_id).to(self.device).eval()
        
        logger.info(f"[+] Hệ thống đã sẵn sàng trên thiết bị: {self.device.upper()} (Sử dụng tối đa {self.max_workers} thread(s) song song)!")

    # --- HÀM TIỆN ÍCH HÌNH ẢNH & PDF ---

    def resize_image(self, img_cv):
        """Hàm Resize ảnh để tối ưu chất lượng đầu vào OCR (Numpy input)"""
        h, w = img_cv.shape[:2]                
        if max(h, w) > self.size_input:            
            scale = self.size_input / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h))        
        return img_cv

    def read_pdf_fitz(self, pdf_file):
        """Đọc PDF bằng fitz với giới hạn trang và cơ chế fallback"""
        try:
            doc = fitz.open(pdf_file)
            list_all_page_cv = []
            for idx_page, page in enumerate(doc):
                if idx_page >= self.max_page_pdf:
                    logger.warning(f"Vượt quá giới hạn {self.max_page_pdf} trang, chỉ xử lý đến trang này.")
                    break
                
                # Render trang thành pixmap
                pix = page.get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB, annots=True)
                # Chuyển trực tiếp Pixmap sang Numpy Array (BGR cho OpenCV)
                img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
                img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                list_all_page_cv.append(self.resize_image(img_cv))
            doc.close()
            return list_all_page_cv
        except Exception as ex:
            logger.warning(f"[!] Lỗi Fitz: {ex}. Gặp lỗi định dạng PDF, thử đọc theo định dạng ảnh gốc...")
            img_fallback = cv2.imread(pdf_file)
            return [self.resize_image(img_fallback)] if img_fallback is not None else []

    def _to_base64(self, pil_img):
        if pil_img is None: return ""
        try:
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
        except Exception as e:
            return f"Error_Encoding: {str(e)}"

    # --- LOGIC SỬA LỖI VĂN BẢN ---

    def _should_correct(self, label):
        text_labels = ['title', 'paragraph_title', 'text', 'paragraph', 'header', 'footer', 'number']
        return any(tl in label.lower() for tl in text_labels)

    def _correct_text_cleanly(self, text):
        if not text or len(text.strip()) < 2: return text
        lines = text.split('\n')
        final_lines = []
        for line in lines:
            if not line.strip(): 
                final_lines.append(line); continue
            parts = re.split(r'(?=[•·])', line)
            corrected_parts = []
            for part in parts:
                if not part.strip(): 
                    if part: corrected_parts.append(part); continue
                pattern = r'^(\s*(?:[#\*\->\+•·\.]|\d+[\.\)]|[a-zA-Z][\.\)])+\s*)(.*)'
                match = re.match(pattern, part, re.DOTALL)
                prefix, content = (match.group(1), match.group(2)) if match else ("", part)
                if not content.strip(): 
                    corrected_parts.append(prefix); continue
                
                inputs = self.correct_tokenizer(content, return_tensors="pt", truncation=True, max_length=256).to(self.device)
                with torch.no_grad():
                    outputs = self.correct_model.generate(**inputs, max_new_tokens=256, num_beams=5)
                corrected_content = self.correct_tokenizer.decode(outputs[0], skip_special_tokens=True)
                corrected_parts.append(f"{prefix}{corrected_content}")
            final_lines.append("".join(corrected_parts))
        return "\n".join(final_lines)

    def _process_table_cells(self, table_content):
        if '<table>' in table_content:
            def replace_cell(match):
                cell_text = match.group(1)
                return f"<td>{self._correct_text_cleanly(cell_text)}</td>" if len(cell_text.strip()) > 2 else match.group(0)
            return re.sub(r'<td>(.*?)</td>', replace_cell, table_content, flags=re.DOTALL)
        
        lines = table_content.split('\n')
        processed_table = []
        for line in lines:
            if '|' in line and '---' not in line:
                cells = line.split('|')
                fixed_cells = [f" {self._correct_text_cleanly(c.strip())} " if len(c.strip()) > 2 else c for c in cells]
                processed_table.append("|".join(fixed_cells))
            else:
                processed_table.append(line)
        return "\n".join(processed_table)

    # --- BÓC TÁCH METADATA & FORMAT ---

    def _extract_page_metadata(self, page_data):
        data = page_data.get('res', page_data) if isinstance(page_data, dict) else page_data
        if isinstance(data, dict):
            settings = data.get('model_settings', {})
            parsing_list = data.get('parsing_res_list', [])
            imgs_in_doc = page_data.get('imgs_in_doc', [])
        else:
            settings = getattr(data, 'model_settings', {})
            parsing_list = getattr(data, 'parsing_res_list', [])
            imgs_in_doc = getattr(page_data, 'imgs_in_doc', [])
        ignore_labels = settings.get('markdown_ignore_labels', ['header', 'footer'])
        return parsing_list, imgs_in_doc, ignore_labels

    def _find_image_in_doc(self, bbox, imgs_in_doc):
        if not bbox or len(bbox) != 4: return None
        for img_info in imgs_in_doc:
            img_box = img_info.get('coordinate', []) if isinstance(img_info, dict) else getattr(img_info, 'coordinate', [])
            if img_box and len(img_box) == 4:
                if all(abs(float(bbox[j]) - float(img_box[j])) < 10 for j in range(4)):
                    return img_info.get('img') if isinstance(img_info, dict) else getattr(img_info, 'img')
        return None

    def _format_block_to_markdown(self, label, content, bbox, imgs_in_doc):
        if label in ['image', 'figure', 'seal', 'signature', 'header_image']:
            target_img = self._find_image_in_doc(bbox, imgs_in_doc)
            if target_img:
                return f"![{label}](data:image/jpeg;base64,{self._to_base64(target_img)})"
            return f"![{label}](Image_Not_Found)"
        elif label == 'table':
            return self._process_table_cells(content)
        elif label in ['formula', 'equation']:
            return content if content.startswith('$') else f"$$\n{content}\n$$"
        elif self._should_correct(label):
            fixed_text = self._correct_text_cleanly(content)
            if label == 'title': return f"# {fixed_text}"
            if 'paragraph_title' in label or (label != 'text' and 'title' in label): return f"## {fixed_text}"
            return fixed_text
        return content

    # --- LUỒNG XỬ LÝ CHÍNH TỔNG HỢP ---

    def _process_frame(self, img_cv, page_index: int):
        """Hàm xử lý OCR lõi cho 1 khối ảnh (Numpy). Tạo ra log và tự dọn dẹp biến tạm."""
        logger.info(f"Bắt đầu OCR Trang {page_index}...")
        
        # Đảm bảo file ảnh tạm được dọn sạch dẹp bằng try..finally
        fd, tmp_path = tempfile.mkstemp(suffix='.jpg')
        os.close(fd) # đóng handle, chỉ giữ file path
        cv2.imwrite(tmp_path, img_cv)

        try:
            ocr_output = self.ocr_pipeline.predict(tmp_path)
            markdown_blocks = []
            for page_data in ocr_output:
                parsing_list, imgs_in_doc, ignore_labels = self._extract_page_metadata(page_data)
                for block in parsing_list:
                    if isinstance(block, dict):
                        label = block.get('label', block.get('block_label', '')).lower()
                        content = block.get('block_content', block.get('content', '')).strip()
                        bbox = block.get('bbox', block.get('block_bbox', []))
                    else:
                        label = getattr(block, 'label', getattr(block, 'block_label', '')).lower()
                        content = getattr(block, 'content', getattr(block, 'block_content', '')).strip()
                        bbox = getattr(block, 'bbox', getattr(block, 'block_bbox', []))

                    visual_labels = ['image', 'figure', 'seal', 'signature', 'header_image']
                    is_visual = label in visual_labels
                    
                    if not label or label in ignore_labels: continue
                    if not is_visual and not content: continue
                    
                    markdown_blocks.append(self._format_block_to_markdown(label, content, bbox, imgs_in_doc))
            
            logger.info(f"Đã dự đoán xong Trang {page_index}.")
            return "\n\n".join(markdown_blocks)
            
        except Exception as e:
            logger.error(f"Lỗi khi OCR trang {page_index}: {e}")
            return f"Error OCR: {e}"
        finally:
            if os.path.exists(tmp_path): 
                os.remove(tmp_path)

    def process_file(self, file_path):
        """Hàm entry-point xử lý cả Ảnh và PDF với khả năng ĐA LUỒNG"""
        start_time = time.time()
        ext = file_path.lower().split('.')[-1]
        
        frames = []
        if ext == 'pdf':
            # logger.info(f"[*] Đang bóc tách PDF: {file_path}")
            frames = self.read_pdf_fitz(file_path)
        else:
            # logger.info(f"[*] Đang xử lý Ảnh: {file_path}")
            img_cv = cv2.imread(file_path)
            if img_cv is not None:
                frames = [self.resize_image(img_cv)]

        if not frames:
            return "Lỗi: Không thể đọc hay nội suy dữ liệu từ file gốc."

        results = [None] * len(frames)
        
        # Bật Multi-threading / Đa luồng xử lý
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {
                executor.submit(self._process_frame, frame, i+1): i 
                for i, frame in enumerate(frames)
            }
            
            for future in as_completed(future_to_index):
                i = future_to_index[future]
                try:
                    page_content = future.result()
                    # Ghép đánh số dường phân tách trực tiếp
                    page_header = f"## [TRANG {i+1}]"
                    results[i] = f"{page_header}\n\n{page_content}"
                except Exception as exc:
                    logger.error(f"Lỗi Critical khi run thread trang {i+1}: {exc}")
                    results[i] = f"## [TRANG {i+1}]\nLỗi bóc tách luồng xử lý."
            
        # logger.info(f"[+] Hoàn thành file toàn vẹn sau {time.time() - start_time:.2f} giây.")
        
        # Lọc những string còn sót nếu có, rồi gộp lại qua Markdown Divider
        final_markdown = "\n\n---\n\n".join(filter(None, results))
        return final_markdown
