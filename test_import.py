import sys
import os
from pathlib import Path

# Thêm root path
root = Path(__file__).parent.parent
sys.path.append(str(root))

print(f"Root: {root}")
print(f"Sys Path: {sys.path[:3]}")

try:
    from src.gateway.gateway_client import GatewayClient
    g = GatewayClient()
    print("Success: GatewayClient imported!")
except Exception as e:
    print(f"Failure: {e}")
    import traceback
    traceback.print_exc()
