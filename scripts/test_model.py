from openai import OpenAI
from pathlib import Path

env_file = Path(".env")
env_vars = {}
for line in env_file.read_text(encoding="utf-8").splitlines():
    if line.strip() and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env_vars[k.strip()] = v.strip()

api_key = env_vars.get("COMMANDCODE_API_KEY", "")
base_url = "https://api.commandcode.ai/provider/v1"
client = OpenAI(api_key=api_key, base_url=base_url)

prompt = """You are an academic citation analyst. Decompose the following paragraph into atomic claims and match each claim with the cited reference key(s).

Paragraph:
"Recurrent neural networks, long short-term memory [13] and gated recurrent [7] neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [35, 2, 5]. Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [38, 24, 15]."

Output purely a valid JSON array of objects with keys:
- "atomic_claim": string
- "target_keys": list of strings (e.g. ["[13]"])
"""

import time

for attempt in range(5):
    try:
        print(f"Attempt {attempt+1}...")
        response = client.chat.completions.create(
            model="inclusionai/ling-3.1-flash:free",
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=600
        )
        print("--- Atomic Claim Extraction Result ---")
        print("Choice message content:", response.choices[0].message.content)
        print("Usage:", response.usage)
        break
    except Exception as e:
        print(f"Attempt {attempt+1} failed:", e)
        time.sleep(3)




