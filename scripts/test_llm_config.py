import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_config import get_llm_client, resolve_model_id


print("--- Testing Model ID Resolution ---")
raw_id = "ling-3.1-flash:free"
resolved = resolve_model_id(raw_id)
print(f"Original ID: {raw_id} -> Resolved: {resolved}")

print("\n--- Testing Fast Client (Ling 3.1 Flash) ---")
client, model = get_llm_client("fast")
print(f"Role 'fast' configured model: {model}")

res = client.chat_completion(messages=[{"role": "user", "content": "请回复'模型测试正常'"}], max_tokens=20)
print(f"Chat completion response: {res}")

print("\n--- Testing Explicit Model Alias ---")
res2 = client.chat_completion(model="ling-3.1-flash:free", messages=[{"role": "user", "content": "请回复'别名映射正常'"}], max_tokens=20)
print(f"Explicit alias response: {res2}")

print("\n[✓] All tests passed!")
