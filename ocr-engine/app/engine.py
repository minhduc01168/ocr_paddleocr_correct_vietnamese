import os
import io
import time
import base64
import tempfile
import re
import cv2
import numpy as np
import fitz  # PyMuPDF
import torch
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Dict, Any

# ML Libraries
import paddle
from paddleocr import PaddleOCRVL
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from app.config import settings
from app.utils import logger

class OCREngine:
    def __init__(self):
        # 0. Detect and set device for Paddle and Torch
        if torch.cuda.is_available():
            self.device = "cuda"
            paddle.set_device('gpu')
            default_workers = 2
        elif torch.backends.mps.is_available():
            self.device = "mps"
            try:
                paddle.set_device('mps')
            except:
                paddle.set_device('cpu')
            default_workers = os.cpu_count() or 4
        else:
            self.device = "cpu"
            paddle.set_device('cpu')
            default_workers = os.cpu_count() or 4

        self.max_workers = settings.max_workers if settings.max_workers else default_workers
        self.size_input = settings.ocr_image_size
        self.dpi = settings.ocr_dpi

        logger.info("[*] Initializing Single PaddleOCRVL instance (Stability mode)...")
        self.ocr_pipeline = PaddleOCRVL() 
        self.ocr_lock = threading.Lock()

        logger.info(f"[*] Loading Correction Model: {settings.correct_model_id}...")
        self.correct_tokenizer = AutoTokenizer.from_pretrained(settings.correct_model_id)
        self.correct_model = AutoModelForSeq2SeqLM.from_pretrained(settings.correct_model_id).to(self.device).eval()
        
        self._docling_converter = None
        logger.info(f"[+] Stable OCR Engine ready on {self.device.upper()} ({self.max_workers} workers)")

    def paddle_ocr_flow(self, file_path: str) -> str:
        ext = file_path.lower().split('.')[-1]
        frames = []
        if ext == 'pdf':
            frames = self._read_pdf_fitz(file_path)
        else:
            img_cv = cv2.imread(file_path)
            if img_cv is not None:
                frames = [self._resize_image(img_cv)]

        if not frames:
            logger.error(f"Engine cannot extract pages from: {file_path}")
            return "Error: Could not read document."

        results = [None] * len(frames)
        logger.info(f"[*] Starting Stable Pipelined Execution for {len(frames)} pages")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_index = {}
            for i, frame in enumerate(frames):
                try:
                    logger.info(f"[*] Page {i+1}/{len(frames)}: Running OCR Inference ({self.device.upper()})...")
                    with self.ocr_lock:
                        ocr_output = self.ocr_pipeline.predict(frame)
                    logger.info(f"[+] Page {i+1}/{len(frames)}: Inference completed.")
                    
                    if ocr_output:
                        page_data = ocr_output[0]
                        future = executor.submit(self._post_process_page_data, page_data, i+1)
                        future_to_index[future] = i
                except Exception as e:
                    logger.error(f"Error starting pipeline for page {i+1}: {e}")
                    results[i] = f"## [TRANG {i+1}]\nError starting OCR."

            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    page_content = future.result()
                    results[idx] = f"## [TRANG {idx+1}]\n\n{page_content}"
                except Exception as exc:
                    logger.error(f"Post-processing error page {idx+1}: {exc}")
                    results[idx] = f"## [TRANG {idx+1}]\nError formatting page data."
            
        return "\n\n---\n\n".join(filter(None, results))

    def docling_flow(self, file_path: str) -> str:
        from docling.document_converter import DocumentConverter, PdfFormatOption
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        
        if not self._docling_converter:
            pipeline_opts = PdfPipelineOptions()
            pipeline_opts.do_ocr = True
            pipeline_opts.do_table_structure = True 
            self._docling_converter = DocumentConverter(
                format_options={"pdf": PdfFormatOption(pipeline_options=pipeline_opts)}
            )
            
        result = self._docling_converter.convert(file_path)
        doc = result.document
        full_md = doc.export_to_markdown()
        if "---" not in full_md:
            return f"## [TRANG 1]\n\n{full_md}"
        return full_md

    def _resize_image(self, img_cv):
        h, w = img_cv.shape[:2]                
        if max(h, w) > self.size_input:            
            scale = self.size_input / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img_cv = cv2.resize(img_cv, (new_w, new_h))        
        return img_cv

    def _read_pdf_fitz(self, pdf_file):
        doc = fitz.open(pdf_file)
        pages = []
        for idx, page in enumerate(doc):
            if idx >= settings.max_pdf_pages: break
            pix = page.get_pixmap(dpi=self.dpi, colorspace=fitz.csRGB)
            img_np = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3)
            img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            pages.append(self._resize_image(img_cv))
        doc.close()
        return pages

    def _to_base64(self, pil_img):
        if pil_img is None: return ""
        try:
            from PIL import Image
            if isinstance(pil_img, np.ndarray):
                pil_img = Image.fromarray(cv2.cvtColor(pil_img, cv2.COLOR_BGR2RGB))
            buffered = io.BytesIO()
            pil_img.save(buffered, format="JPEG", quality=85)
            return base64.b64encode(buffered.getvalue()).decode()
        except: return ""

    def _should_correct(self, label):
        text_labels = ['title', 'paragraph_title', 'text', 'paragraph', 'header', 'footer', 'number']
        return any(tl in label.lower() for tl in text_labels)

    def _correct_text_batch(self, texts: List[str]) -> List[str]:
        if not texts: return []
        valid_indices = [i for i, t in enumerate(texts) if t and len(t.strip()) > 2]
        if not valid_indices: return texts
        to_correct = [texts[i] for i in valid_indices]
        results = list(texts)
        try:
            inputs = self.correct_tokenizer(to_correct, return_tensors="pt", padding=True, truncation=True, max_length=256).to(self.device)
            with torch.no_grad():
                outputs = self.correct_model.generate(**inputs, max_new_tokens=256)
            corrected = self.correct_tokenizer.batch_decode(outputs, skip_special_tokens=True)
            for idx, corrected_text in zip(valid_indices, corrected):
                results[idx] = corrected_text
        except Exception as e:
            logger.error(f"Batch correction failed: {e}")
        return results

    def _correct_text_cleanly(self, text):
        if not text or len(text.strip()) < 2: return text
        inputs = self.correct_tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(self.device)
        with torch.no_grad():
            outputs = self.correct_model.generate(**inputs, max_new_tokens=256)
        return self.correct_tokenizer.decode(outputs[0], skip_special_tokens=True)

    def _process_table_cells(self, table_content):
        if '<table>' in table_content:
            table_content = re.sub(r'\n\s*\n', '\n', table_content)
            def replace_cell(match):
                cell_text = match.group(1)
                return f"<td>{self._correct_text_cleanly(cell_text)}</td>" if len(cell_text.strip()) > 2 else match.group(0)
            return re.sub(r'<td>(.*?)</td>', replace_cell, table_content, flags=re.DOTALL)
        return table_content

    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        if not box1 or not box2 or len(box1) != 4 or len(box2) != 4: return 0.0
        x1, y1, x2, y2 = max(box1[0], box2[0]), max(box1[1], box2[1]), min(box1[2], box2[2]), min(box1[3], box2[3])
        inter_area = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = area1 + area2 - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def _extract_page_metadata(self, data):
        if isinstance(data, dict):
            res = data.get('res', data)
            pl = res.get('parsing_res_list', [])
            imgs = res.get('imgs_in_doc', [])
        else:
            res = getattr(data, 'res', data)
            pl = getattr(res, 'parsing_res_list', [])
            imgs = getattr(res, 'imgs_in_doc', [])
        return pl, imgs, ['header', 'footer']

    def _find_image_in_doc(self, bbox, imgs_in_doc):
        if not bbox or len(bbox) != 4: return None
        best_iou, best_img = 0.5, None
        for img_info in imgs_in_doc:
            img_box = img_info.get('coordinate', []) if isinstance(img_info, dict) else getattr(img_info, 'coordinate', [])
            if img_box and len(img_box) == 4:
                iou = self._calculate_iou(bbox, img_box)
                if iou > best_iou:
                    best_iou, best_img = iou, (img_info.get('img') if isinstance(img_info, dict) else getattr(img_info, 'img'))
        return best_img

    def _get_block_info(self, block):
        if isinstance(block, dict):
            label = block.get('label', block.get('block_label', '')).lower()
            content = block.get('block_content', block.get('content', '')).strip()
            bbox = block.get('bbox', block.get('block_bbox', block.get('coordinate', [])))
        else:
            label = getattr(block, 'label', getattr(block, 'block_label', '')).lower()
            content = getattr(block, 'block_content', getattr(block, 'content', '')).strip()
            bbox = getattr(block, 'bbox', getattr(block, 'block_bbox', getattr(block, 'coordinate', [])))
        return label, content, bbox

    def _format_block_to_markdown(self, label, content, bbox, imgs_in_doc):
        if label in ['image', 'figure', 'seal', 'signature', 'header_image', 'picture']:
            target_img = self._find_image_in_doc(bbox, imgs_in_doc)
            if target_img is not None:
                b64 = self._to_base64(target_img)
                return f"![{label}](data:image/jpeg;base64,{b64})"
            return "" 
        elif label == 'table':
            return self._process_table_cells(content)
        elif self._should_correct(label):
            if label == 'title': return f"# {content}"
            if label == 'paragraph_title': return f"## {content}"
            return content
        return content

    def _post_process_page_data(self, page_data, page_index: int):
        try:
            raw_list, imgs_in_doc, ignore_labels = self._extract_page_metadata(page_data)
            if not raw_list: return ""
            containers, text_candidates = [], []
            for block in raw_list:
                label, content, bbox = self._get_block_info(block)
                if not label or label in ignore_labels: continue
                if label in ['table', 'image', 'figure', 'picture', 'seal', 'signature']:
                    containers.append({'label': label, 'content': content, 'bbox': bbox, 'block': block})
                else:
                    text_candidates.append({'label': label, 'content': content, 'bbox': bbox, 'block': block})
            filtered_list = list(containers)
            for txt in text_candidates:
                if not any(self._calculate_iou(txt['bbox'], cnt['bbox']) > 0.6 for cnt in containers):
                    filtered_list.append(txt)
            filtered_list.sort(key=lambda x: (x['bbox'][1] if len(x['bbox'])>1 else 0, x['bbox'][0] if x['bbox'] else 0))
            blocks_to_correct, indices_for_correction = [], []
            for i, item in enumerate(filtered_list):
                if self._should_correct(item['label']) and item['content']:
                    blocks_to_correct.append(item['content'])
                    indices_for_correction.append(i)
            if blocks_to_correct:
                logger.info(f"[*] Page {page_index}: Correcting {len(blocks_to_correct)} blocks...")
                corrected_values = self._correct_text_batch(blocks_to_correct)
                for idx, val in zip(indices_for_correction, corrected_values):
                    filtered_list[idx]['content'] = val
            markdown_blocks = []
            for item in filtered_list:
                formatted = self._format_block_to_markdown(item['label'], item['content'], item['bbox'], imgs_in_doc)
                if formatted: markdown_blocks.append(formatted)
            return "\n\n".join(markdown_blocks)
        except Exception as e:
            logger.error(f"Post-processing failed for page {page_index}: {e}")
            return "Error processing content."

engine = OCREngine()
