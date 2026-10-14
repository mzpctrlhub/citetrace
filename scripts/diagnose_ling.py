import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm_config import get_llm_client
from free import EXTRACTION_PROMPT

client, model = get_llm_client("fast")
raw_client = client._get_client()

print("Testing EXTRACTION_PROMPT with max_tokens=3000...")
resp = raw_client.chat.completions.create(
    model="inclusionai/ling-3.1-flash:free",
    messages=[{"role": "user", "content": EXTRACTION_PROMPT}],
    max_tokens=3000
)

msg = resp.choices[0].message
print("finish_reason:", resp.choices[0].finish_reason)
print("content len:", len(msg.content or ""))
reasoning = getattr(msg, "reasoning_content", "") or ""
print("reasoning_content len:", len(reasoning))
print("content:", msg.content)
if not msg.content:
    print("Full reasoning snippet:", reasoning[-500:])
