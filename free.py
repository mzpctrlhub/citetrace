# free.py
import json
import re
from llm_config import get_llm_client

# 1. 替换为全新的待测试学术段落（来自 Attention Is All You Need 第 2 节 Background，citations/4.json）
# 本段包含 [16] Neural GPU, [18] ByteNet, [9] ConvS2S, [12] 梯度/长程依赖 四大文献
TEST_PARAGRAPH = """The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU [16], ByteNet [18] and ConvS2S [9], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions. In these models, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, linearly for ConvS2S and logarithmically for ByteNet. This makes it more difficult to learn dependencies between distant positions [12]."""

CITED_KEYS = ["[16]", "[18]", "[9]", "[12]"]

# 2. 以“文献为中心”的精准提取提示词（一个引用文献对应一条完整责任主张）
EXTRACTION_PROMPT = f"""You are an expert academic citation analyst.
Your task is to analyze the following academic paragraph and extract the EXACT attribution claim for EACH cited reference.

[Extraction Rules]:
1. Exactly ONE comprehensive claim per cited reference key.
2. For each reference, summarize all capabilities, architectural designs, mathematical complexities, or theoretical assertions specifically attributed to that reference.
3. If multiple references are cited for a shared property, clearly state that shared property in the claim for each of those references.
4. Each claim must be a complete, self-contained English sentence mentioning the model/concept name explicitly.
5. Output MUST be strictly valid JSON without conversational preamble or markdown code fences.

Input Paragraph:
"{TEST_PARAGRAPH}"

All citation keys present in text:
{CITED_KEYS}

Respond strictly in this JSON format:
[
  {{
    "target_key": "[X]",
    "paper_role": "Short phrase describing the cited entity (e.g., 'ByteNet architecture')",
    "attributed_claim": "The complete, standalone statement of what the text claims about or attributes to this reference.",
    "context_scope": "Key phrase from original text indicating this attribution"
  }}
]
"""

def extract_json(raw_text: str):
    """鲁棒提取 JSON（兼容带 ```json ``` 标记或纯文本输出）"""
    raw_text = raw_text.strip()
    match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw_text)
    if match:
        raw_text = match.group(1).strip()
    return json.loads(raw_text)

print("=" * 80)
print(">> 测试 Ling 3.1 Flash 在全新多引用段落上的零泄漏解耦能力")
print("=" * 80)
print(f"【测试输入段落】:\n{TEST_PARAGRAPH}\n")
print(f"【涉及引用清单】: {', '.join(CITED_KEYS)}\n")
print("提示词中无任何目标论文名称泄漏，正在调用 Ling 3.1 Flash 盲测解耦...\n")

client, model = get_llm_client("fast")

response_text = client.chat_completion(
    messages=[
        {"role": "user", "content": EXTRACTION_PROMPT}
    ],
    temperature=0.0,
)

print("-" * 80)
print(">> 模型原始输出:")
print(response_text)
print("-" * 80)

# 解析与胜任力审计
try:
    claims = extract_json(response_text)
    print(f"\n>> 【解析成功！共提取出 {len(claims)} 篇文献的独立责任主张（预期 4 篇）】\n")
    print(f"| 文献标号 | 论文实体角色 | 提取出的完整责任主张 (Attributed Claim) | 原文对应范围 |")
    print(f"| :---: | :--- | :--- | :--- |")
    for c in claims:
        key = c.get("target_key", "-")
        role = c.get("paper_role", "-")
        claim = c.get("attributed_claim", "")
        scope = c.get("context_scope", "")
        print(f"| {key} | {role} | {claim} | {scope} |")
    
    extracted_keys = {c.get("target_key") for c in claims}
    exact_match_count = len(claims) == len(CITED_KEYS)
    all_keys_covered = set(CITED_KEYS).issubset(extracted_keys)

    print("\n>> 【文献级 1:1 归纳审计 (Per-Citation 1:1 Audit)】:")
    print(f"  [1] 是否达成 1 篇文献对应 1 条主张: {'✓ 达成 (刚好 4 条)' if exact_match_count else f'✗ 未达成 ({len(claims)} 条)'}")
    print(f"  [2] 是否覆盖了全部 4 篇被引文献:     {'✓ 全部覆盖' if all_keys_covered else '✗ 存在遗漏'}")
    
    if exact_match_count and all_keys_covered:
        print("\n[✓ 结论]: 成功实现'一文献一条完整主张'，每个文献主张均可直接作为独立检索包！")
    else:
        print("\n[!] 提示: 主张数量与文献清单数量尚未完全对齐。")

except Exception as e:
    print(f"\n[!] JSON 解析异常: {e}")

print("=" * 80)
