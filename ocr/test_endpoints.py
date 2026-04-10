#!/usr/bin/env python3
"""
Script kiểm tra sức khỏe các Endpoints của OCR Service.
Dùng để verify Server hoạt động đúng trước khi deploy lên SageMaker.

Cách dùng:
    python test_endpoints.py [--host http://localhost:8080] [--pdf path/to/test.pdf]
"""

import sys
import argparse
import base64
import json
import httpx


def test_health(host: str):
    """Test GET / và GET /ping"""
    print("\n[1/4] Kiểm tra Health Check Endpoints...")
    
    # Local health
    r = httpx.get(f"{host}/")
    assert r.status_code == 200, f"  FAIL /: {r.status_code}"
    print(f"  ✅ GET /       -> {r.json()}")

    # SageMaker ping
    r = httpx.get(f"{host}/ping")
    assert r.status_code == 200, f"  FAIL /ping: {r.status_code} - {r.text}"
    print(f"  ✅ GET /ping   -> {r.json()}")


def test_file_upload(host: str, file_path: str):
    """Test POST /api/v1/ocr/process với multipart file upload"""
    print(f"\n[2/4] Kiểm tra File Upload Endpoint với file: {file_path}")
    
    with open(file_path, "rb") as f:
        files = {"file": (file_path.split("/")[-1], f, "application/pdf")}
        r = httpx.post(f"{host}/api/v1/ocr/process", files=files, timeout=300)
    
    assert r.status_code == 200, f"  FAIL: {r.status_code} - {r.text}"
    result = r.json()
    markdown_preview = result.get("markdown", "")[:200]
    print(f"  ✅ POST /api/v1/ocr/process -> status={result.get('status')}")
    print(f"     Markdown preview: {markdown_preview}...")


def test_base64_upload(host: str, file_path: str):
    """Test POST /api/v1/ocr/process/base64"""
    print(f"\n[3/4] Kiểm tra Base64 Endpoint với file: {file_path}")

    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    payload = {"base64_data": b64, "filename": file_path.split("/")[-1]}
    r = httpx.post(f"{host}/api/v1/ocr/process/base64", json=payload, timeout=300)

    assert r.status_code == 200, f"  FAIL: {r.status_code} - {r.text}"
    result = r.json()
    print(f"  ✅ POST /api/v1/ocr/process/base64 -> status={result.get('status')}")


def test_sagemaker_invocation(host: str, file_path: str):
    """Test POST /invocations với cả 2 dạng binary và JSON base64"""
    print(f"\n[4/4] Kiểm tra SageMaker /invocations Endpoint với file: {file_path}")

    # 4a. Raw Binary
    with open(file_path, "rb") as f:
        raw = f.read()
    
    ct = "application/pdf" if file_path.endswith(".pdf") else "image/jpeg"
    r = httpx.post(f"{host}/invocations", content=raw,
                   headers={"Content-Type": ct}, timeout=300)
    assert r.status_code == 200, f"  FAIL (binary): {r.status_code} - {r.text}"
    print(f"  ✅ POST /invocations (raw binary) -> status={r.json().get('status')}")

    # 4b. JSON Base64
    with open(file_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    
    payload = {"data": b64, "filename": file_path.split("/")[-1]}
    r = httpx.post(f"{host}/invocations", json=payload,
                   headers={"Content-Type": "application/json"}, timeout=300)
    assert r.status_code == 200, f"  FAIL (json base64): {r.status_code} - {r.text}"
    print(f"  ✅ POST /invocations (JSON base64) -> status={r.json().get('status')}")


def main():
    parser = argparse.ArgumentParser(description="OCR Service Endpoint Tester")
    parser.add_argument("--host", default="http://localhost:8080", help="Base URL của service")
    parser.add_argument("--pdf", default=None, help="Đường dẫn đến file PDF hoặc ảnh để test")
    args = parser.parse_args()

    print(f"=== OCR Service Test Suite ===")
    print(f"Target Host: {args.host}")

    try:
        test_health(args.host)

        if args.pdf:
            test_file_upload(args.host, args.pdf)
            test_base64_upload(args.host, args.pdf)
            test_sagemaker_invocation(args.host, args.pdf)
        else:
            print("\n[!] Không có --pdf nên bỏ qua các bài test upload file.")
            print("    Chạy lại với: python test_endpoints.py --pdf /đường/dẫn/tới/test.pdf")

        print("\n✅ Tất cả các bài test đã PASS!")

    except AssertionError as e:
        print(f"\n❌ Test FAIL: {e}")
        sys.exit(1)
    except httpx.ConnectError:
        print(f"\n❌ Không kết nối được đến {args.host}. Server có đang chạy không?")
        sys.exit(1)


if __name__ == "__main__":
    main()
