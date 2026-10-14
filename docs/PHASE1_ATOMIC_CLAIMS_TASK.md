# 阶段 1 任务书：原子主张提取（Atomic Claim Extraction）

> 本文件是给执行者的**唯一**任务说明。只做完本文件列出的内容即算完成。
> 后续的证据检索（BM25 / chunks）、引用真伪判决、多 Agent 编排、向量库、最终报告**全部不在本轮范围内**。

---

## 0. 一句话目标

把 `cache/papers/<paper>/extracted_claims.json` 里的引用段落，交给 LLM 拆成「最小可独立核查的原子主张」，每条主张绑定它对应的引用编号；再用**纯程序**校验 LLM 没有篡改原文、没有增删引用编号、没有改动数字。产物写入同一论文目录，先只跑前 10 个段落。

---

## 1. 已核实的项目现状（照此写代码，不要另造接口）

以下事实均已在仓库中确认，**不需要再去探索**：

| 项 | 事实 |
|---|---|
| 解释器 | `/root/citetrace/.venv/bin/python`（Python 3.12.3）。系统 `python3` **没有** openai，必须用这个 venv |
| 依赖 | 已装：`openai 3.22.1`、`pydantic 2.13.5`、`pytest 9.1.1`、`httpx`、`langgraph`。**不需要 pip install 任何东西** |
| LLM 客户端 | `llm_config.py` 已存在。`get_llm_client(role) -> (LLMClient, model_name)`；`LLMClient.chat_completion(model=None, messages=..., temperature=0.0, max_tokens=1500, response_format=None) -> str`（返回**纯字符串**，内置 429 指数退避重试） |
| 角色 | 本轮用 `role="fast"`。`.env` 里 `COMMANDCODE_FAST_MODEL=inclusionai/ling-3.1-flash:free`，key 已配置可用。**不要修改 `.env`** |
| 输入文件 | `cache/papers/Attention_Is_All_You_Need/extracted_claims.json`，顶层是 dict：`paper_id / paper_title / pdf_sha256 / parser_version / extraction_method / total_claims / extracted_at / claims[]`，共 **36 条** claim |
| 单条 claim 字段 | `claim_id`（如 `claim_001`）、`context_id`、`content_type`（当前全为 `prose`）、`paragraph`（218–825 字符）、`citation_keys`（形如 `"[13]"`，**带方括号**）、`raw_citation_keys`、`citation_key_map`、`matched_references`、`source_section`、`source_file` |
| 前 10 条的引用数 | 8, 2, 3, 4, 4, 1, 3, 4, 2, 1（`claim_001` 是最难的 8 引用复合句，拿它当首个试金石） |
| 已有枚举 | `models/verification.py` 里的 `ClaimIntent` = `background / method_usage / factual_claim / metric_comparison / unverifiable`。**直接复用它**作为主张与引用的关系分类，不要新造一套词表 |
| 目录约定 | `models/`、`parsers/`、`scripts/`、`tests/` 已存在且都有 `__init__.py`；`prompts/` **不存在，需要新建**（纯数据目录，不放 `__init__.py`） |
| 输出目录 | 产物写回论文目录根：`cache/papers/<paper>/`（与 `extracted_claims.json`、`extracted_claims.md` 同级） |

### 1.1 一个必须一并修掉的坑

`citetrace_pdf/storage/writer.py:11` 的 `MarkdownWriter.KEEP_ROOT_MDS` 会在 PDF 重跑时**删除论文目录下所有不在白名单里的根级 `*.md`**。本轮要产出的 `atomic_claims_review.md` 会被误删。

→ 必须把 `"atomic_claims_review.md"` 加进这个集合（一行改动）。

---

## 2. 交付物清单（10 项，不多不少）

1. `models/claim.py` —— 新建
2. `models/__init__.py` —— 追加导出
3. `llm_config.py` —— 给 `LLMClient` **追加**一个 `generate_json()` 方法（不得改动已有方法的行为）
4. `prompts/atomic_claim_v1.txt` —— 新建，内容见 §3.3，**逐字照抄**
5. `parsers/claim_validator.py` —— 新建
6. `parsers/atomic_claim_extractor.py` —— 新建
7. `scripts/extract_atomic_claims.py` —— 新建
8. `tests/test_claim_validator.py` —— 新建（离线，不联网）
9. `citetrace_pdf/storage/writer.py:11` —— 一行改动（见 §1.1）
10. `parsers/__init__.py` —— 追加导出

---

## 3. 逐文件规格

### 3.1 `models/claim.py`

LLM 面向的模型（`AtomicClaimDraft`）与落盘模型（`AtomicClaim`）**分开**：
`claim_id`、`parent_claim_id`、`context_id` 由程序盖章，**不要让 LLM 生成 ID**（弱模型必然编号冲突/跳号）。

```python
from typing import List
from pydantic import BaseModel, Field

from .verification import ClaimIntent


class SourceSpan(BaseModel):
    """主张在原文段落中的字符区间。text 必须逐字来自 paragraph。"""
    start: int = Field(ge=0, description="起始下标（含）")
    end: int = Field(gt=0, description="结束下标（不含）")
    text: str = Field(description="paragraph[start:end] 的逐字内容")


class AtomicClaimDraft(BaseModel):
    """LLM 单次输出的原始结构（不含任何由程序盖章的字段）"""
    source_spans: List[SourceSpan]
    verbatim_claim: str
    normalized_claim: str
    citation_keys: List[str]
    relationship: ClaimIntent
    requires_review: bool = False
    review_reason: str = ""


class AtomicClaimResponse(BaseModel):
    """LLM 完整响应"""
    claims: List[AtomicClaimDraft]
    unassigned_citation_keys: List[str] = Field(default_factory=list)


class AtomicClaim(AtomicClaimDraft):
    """落盘结构：在 Draft 之上补程序盖章字段，并允许 validator 就地修正 span 下标"""
    claim_id: str                 # 形如 "claim_001.a01"，由 extractor 生成
    parent_claim_id: str          # 来自 extracted_claims.json 的 claim_id
    context_id: str               # 透传
```

`models/__init__.py` 追加导出 `SourceSpan / AtomicClaimDraft / AtomicClaimResponse / AtomicClaim`（保留已有导出）。

### 3.2 `llm_config.py`：追加 `LLMClient.generate_json()`

**只追加，不重写已有逻辑。** 先加一个模块级函数（`generate_json` 和 `--dry-run` 共用它，全仓库只能有这一份占位符替换实现）：

```python
def render_prompt(template: str, payload: Dict[str, Any]) -> str:
    """把 {{大写键名}} 占位符替换为 payload 中对应的值。"""
```

**必须用 `str.replace`，不要用 `str.format`**——prompt 里含大量 JSON 花括号，`format` 必炸。实现：对 `payload` 的每个键 `k`，执行 `template.replace("{{" + k.upper() + "}}", str(v))`，返回结果。

再给 `LLMClient` 追加方法，签名：

```python
def generate_json(
    self,
    *,
    prompt: str,
    payload: dict,
    schema,                 # type[BaseModel]
    system: str = "You are a precise academic citation parser. Output strictly valid JSON and nothing else.",
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 4000,
    repair_rounds: int = 1,
):  # -> BaseModel 实例
```

行为要求：

1. 方法第一步就是 `prompt = render_prompt(prompt, payload)`，后续全部基于渲染后的文本。
2. 调 `self.chat_completion(...)` 拿字符串。**v1 不要传 `response_format`**（当前代理端点对它支持不稳定）。
3. 剥壳：先 `strip()`；若被 ```` ```json ```` / ```` ``` ```` 包裹则取出围栏内内容；再截取第一个 `{` 到最后一个 `}` 之间的子串；然后 `json.loads`。
4. `schema.model_validate(data)` 返回实例。
5. **修复轮**：若 `json.loads` 或 `model_validate` 抛错，且 `repair_rounds > 0`，把原始输出 + 具体报错文本作为新的一轮 user 消息追加进 `messages`（"你的上一次输出无法解析，错误是 X。请只输出修正后的完整 JSON。"），重调一次。修复轮用尽仍失败 → 抛 `ValueError`，异常信息里带上原始响应前 500 字符（便于人工排查）。
6. 全过程用 `logger`（`citetrace.llm`）记录 warning，不要 `print`。

### 3.3 `prompts/atomic_claim_v1.txt`

**逐字写入以下内容**（这是本阶段质量的核心，不要自行"优化"、不要增删示例、不要加中文注释）：

```text
You are a citation-attribution parser for academic papers.

You receive ONE paragraph from a paper and the complete list of citation keys that appear in it.
Decompose the paragraph into atomic claims and bind each atomic claim to the citation key(s) the author attached to it.

You are not judging whether any claim is true. You only record what the author asserted, and which reference the author attached to that assertion.

--- Derivation procedure (follow it in this order) ---
Step 1. Locate every citation marker in the paragraph. A marker is a bracketed number group: either a single key or a comma-separated stack of keys.
Step 2. For each marker, identify the smallest phrase or clause the marker is attached to - the noun phrase, verb phrase or clause immediately preceding it. That phrase is the marker's scope.
Step 3. Emit one atomic claim per distinct scope. Two markers sharing one scope produce ONE claim carrying both keys. Two markers with different scopes produce TWO claims, each carrying only its own key.
Step 4. When a scope names several parallel items and each item carries its own marker, emit one claim per item. Keep each item's marker on its own item only.
Step 5. Rewrite the scope into a self-contained English declarative sentence:
        - resolve every pronoun and every elided subject to the noun it refers to inside this paragraph;
        - keep the author's own wording for technical terms, model names, dataset names and units;
        - keep every digit, number, percentage and inequality exactly as printed. Never convert, round, re-unit or re-derive a value;
        - if the author hedged, keep the hedge; if the author stated it flatly, do not add a hedge.
Step 6. Classify the citation intent with exactly one value:
        background        - the reference is cited as prior art or general context for a widely accepted statement, and the paragraph asserts nothing specific about that reference's own content.
        method_usage      - the paragraph states that the authors, or a named system, use, adopt, build on, extend or compare against the thing the reference introduces.
        factual_claim     - the paragraph asserts a specific property, mechanism, result or behaviour attributable to the content of the referenced work itself.
        metric_comparison - the claim carries an exact experimental number, score, or a complexity/measurement comparison.
        unverifiable      - the statement is about the present paper's own contribution, or carries no externally checkable content.
Step 7. Set requires_review to true when any of these holds, and put the condition that fired into review_reason:
        - the marker sits at a clause or sentence boundary and its scope could equally be the preceding or the following statement;
        - a stacked group of markers covers a scope that itself contains several distinguishable facts, and the paragraph gives no signal for splitting the stack;
        - the claim depends on a referent that this paragraph does not contain;
        - a number appears in the scope but its comparison target lies outside the paragraph.
        Otherwise set requires_review to false and leave review_reason empty.
Step 8. Every key in the input list must appear either in exactly one claim's citation_keys, or in unassigned_citation_keys. A key you could not attach to any claim goes into unassigned_citation_keys.

--- Field rules ---
source_spans     : the exact character range(s) of the paragraph this claim was derived from. Copy `text` verbatim from the paragraph, character for character, including punctuation and citation markers. `start` is inclusive, `end` is exclusive, both are indices into the paragraph. Use more than one span when the claim draws on non-adjacent parts of the paragraph.
verbatim_claim   : the text of the primary span, copied verbatim. It must be a substring of the paragraph.
normalized_claim : the self-contained sentence produced in Step 5. This is the only field you may rewrite.
citation_keys    : copied verbatim from the input list, brackets included.
relationship     : exactly one value from Step 6.
requires_review  : boolean from Step 7.
review_reason    : short string; empty when requires_review is false.

--- Output ---
Return a single JSON object and nothing else. No markdown fence, no commentary, no trailing text.
Shape:
{
  "claims": [
    {
      "source_spans": [ { "start": <int>, "end": <int>, "text": <string> } ],
      "verbatim_claim": <string>,
      "normalized_claim": <string>,
      "citation_keys": [<string>],
      "relationship": <background|method_usage|factual_claim|metric_comparison|unverifiable>,
      "requires_review": <bool>,
      "review_reason": <string>
    }
  ],
  "unassigned_citation_keys": [<string>]
}

--- Paragraph ---
{{PARAGRAPH}}

--- All citation keys present in this paragraph ---
{{CITATION_KEYS}}
```

> 说明（写给执行者，**不要写进 prompt 文件**）：prompt 里刻意不给任何填好值的示例主张。给示例会把弱模型的输出锚定到示例上，导致它照抄而不是照段落推导。`<...>` 只描述形状，不描述取值。

### 3.4 `parsers/claim_validator.py`

```python
@dataclass
class ValidationResult:
    valid: bool
    errors: List[str]      # 硬错误：整条 parent 判失败
    warnings: List[str]    # 软问题：可自动修复或可容忍
    repaired_spans: int = 0


class ClaimValidator:
    def validate_and_repair(
        self,
        paragraph: str,
        expected_keys: List[str],
        response: AtomicClaimResponse,
    ) -> ValidationResult:
        ...
```

校验规则（**逐条实现，错误信息用中文并带定位**，如 `第 3 条主张`）：

| # | 检查 | 不通过时的处理 |
|---|---|---|
| 1 | 每条 claim 的 `citation_keys` 必须 ⊆ `expected_keys` | **error**：`新增不存在的引用: [...]`（列出具体键） |
| 2 | `expected_keys` 中未被任何 claim 引用、且不在 `response.unassigned_citation_keys` 里的键 | **error**：`遗漏引用: [...]` |
| 3 | `expected_keys` 中出现在 `unassigned_citation_keys` 里的键 | **warning**：`未归属引用: [...]`（合法出口，不算失败） |
| 4 | 每个 `span.text` 必须能在 `paragraph` 中定位（用 `paragraph.find(text)`） | 找不到 → **error**：`第 i 条主张的原文片段不存在于段落中: "..."` |
| 5 | 定位成功但 `paragraph[start:end] != text` | **warning + 就地修正** `start/end` 为 `find` 的结果，`repaired_spans += 1`。**这不是错误** |
| 6 | `verbatim_claim` 必须是 `paragraph` 的子串；直接 `find` 失败时，用「空白折叠」后再比一次（把双方 `\s+` 压成单空格再定位） | 空白差异 → **warning + 修正**为原文真实子串；两种方式都定位不到 → **error** |
| 7 | 数字保真：先用正则 `\[\d+(?:\s*,\s*\d+)*\]` 抹掉两侧的引用标记，再抽 `\d+(?:[.,]\d+)*` 得到数字集合。`normalized_claim`/`verbatim_claim` 里出现的数字必须都出现在 `paragraph` 的数字集合中 | **error**：`第 i 条主张出现原文没有的数字: [...]` |
| 8 | `response.claims` 为空而 `expected_keys` 非空 | **error**：`未产出任何主张` |
| 9 | 某条 claim 的 `citation_keys` 为空 | **warning**：`第 i 条主张未绑定任何引用` |
| 10 | `requires_review=True` 但 `review_reason` 为空 | **warning + 修正**为 `"unspecified"` |

规则 5、6 之所以是「修复 + 警告」而不是「报错」：`ling-3.1-flash:free` 这类模型数字符下标的能力很差，但**它抄的文本本身通常是准的**。真正要防的是「片段凭空捏造」，这由 `find` 兜住；下标算错属于可自动纠正的噪声。若把它当硬错误，第一批 10 条会几乎全灭，阶段 1 永远无法验收。

**严禁**为了让校验通过而修改 `expected_keys`、放宽规则 1/2/7、或在 validator 里对 LLM 输出做「猜测性补全」。

### 3.5 `parsers/atomic_claim_extractor.py`

```python
@dataclass
class ExtractionOutcome:              # 用 dataclass，不要用 BaseModel
    parent_claim_id: str
    context_id: str
    status: str                       # "ok" | "validation_failed" | "llm_failed" | "skipped_no_citation"
    claims: List[AtomicClaim] = field(default_factory=list)
    unassigned_citation_keys: List[str] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    error: str = ""
    raw_response: str = ""            # 仅失败时保留，便于人工排查
    attempts: int = 1


class AtomicClaimExtractor:
    def __init__(self, llm, model_name: str, validator: ClaimValidator,
                 prompt_path: Path | None = None, max_tokens: int = 4000):
        ...

    def preview(self, context: dict) -> str:
        """返回该 context 拼装完成、占位符已全部替换的最终 prompt。
        供 --dry-run 使用；extract() 内部走同一条 _payload() 路径，不得出现第二份拼装逻辑。"""

    def extract(self, context: dict) -> ExtractionOutcome:
        """context 是 extracted_claims.json 里的一条 claim 记录（原样传入）"""
```

行为要求：

0. `ExtractionOutcome` 用 dataclass 而不是 BaseModel：`ValidationResult` 是 dataclass，塞进 pydantic 模型要额外开 `arbitrary_types_allowed`，没有必要。落盘时由 script 负责 `dataclasses.asdict()`（`AtomicClaim` 是 pydantic 模型，用 `.model_dump()`）。本文件需要的导入：`from dataclasses import dataclass, field`、`from typing import List, Optional`、`from pathlib import Path`、`from llm_config import render_prompt`。
1. `context["citation_keys"]` 为空 → 直接返回 `status="skipped_no_citation"`，不调用 LLM。
2. prompt 模板在 `__init__` 里读一次并缓存为 `self.prompt_text`，默认路径 `Path(__file__).resolve().parents[1] / "prompts" / "atomic_claim_v1.txt"`。另加私有方法 `_payload(context) -> dict`，返回 `{"paragraph": context["paragraph"], "citation_keys": ", ".join(context["citation_keys"])}`（一行、逗号分隔、带方括号，形如 `[13], [7], [35]`）。`preview()` 的实现就是 `render_prompt(self.prompt_text, self._payload(context))`。
3. 调 `llm.generate_json(prompt=self.prompt_text, payload=self._payload(context), schema=AtomicClaimResponse, max_tokens=self.max_tokens)`。
4. 拿到 response 后调 `validator.validate_and_repair(...)`。
5. **若 `valid=False`：做一次语义级重试**（总共最多 2 次尝试）。重试方式**不是**改 message 历史，而是在 prompt **模板**末尾追加纠正段后重新调 `generate_json`（`prompt_1` 指 `self.prompt_text` 模板本身，占位符替换仍由 `render_prompt` 完成）：

   ```
   prompt_2 = prompt_1 + "\n\n--- Correction required ---\nYour previous output was rejected by the checker for these reasons:\n" \
              + "\n".join(f"- {e}" for e in errors) \
              + "\nRe-read the paragraph and return the full corrected JSON object."
   ```

   `generate_json` 的 `repair_rounds` 保持默认 1（它负责 JSON 解析层面的修复）；本条负责的是**校验规则层面**的修复，两者职责不重叠。第二次仍 `valid=False` → `status="validation_failed"`，`error` 里带上两次的全部 errors，`raw_response` 保留第二次原文。
6. 成功时盖章生成 `AtomicClaim`：`AtomicClaim(**draft.model_dump(), claim_id=..., parent_claim_id=..., context_id=...)`，其中 `claim_id = f"{parent_claim_id}.a{idx:02d}"`（idx 从 1 开始，按 response 顺序）。
   注意 validator 是**就地修改** draft 的 span 的，所以必须在 validate 之后再 `model_dump()` 盖章。
7. LLM 调用抛异常 → `status="llm_failed"`，`error=str(e)`，**不要让异常冒泡中断整批**。

`parsers/__init__.py` 追加导出 `AtomicClaimExtractor`、`ClaimValidator`、`ValidationResult`。

### 3.6 `scripts/extract_atomic_claims.py`

命令行（用 `argparse`，`python -m scripts.extract_atomic_claims` 方式运行）：

```
--paper-folder PATH   必填，论文目录（含 extracted_claims.json）
--limit N             默认 10
--start N             默认 0
--role STR            默认 "fast"
--max-tokens N        默认 4000
--sleep SEC           每次 LLM 调用之间的间隔，默认 1.0（免费模型限流）
--out PATH            默认 <paper-folder>/atomic_claims.json
--review-out PATH     默认 <paper-folder>/atomic_claims_review.md
--dry-run             只打印 extractor.preview(claims[start]) 的结果（即第一条选中段落拼装完成的完整 prompt），然后退出码 0，不联网、不写任何文件
```

流程：读 `extracted_claims.json` → 取 `claims[start : start+limit]` → 逐条 `extractor.extract()` → 汇总落盘 → 在 stdout 打印一张摘要表（每个 parent 一行：`claim_id / status / 主张数 / 涉及引用数 / errors 数 / warnings 数 / requires_review 数`）+ 总计行。

退出码：全部 `ok`（含 `skipped_no_citation`）→ 0；有任何 `validation_failed` / `llm_failed` → 1。

**`atomic_claims.json` 输出结构**（严格按此写，后续阶段依赖它）：

```json
{
  "paper_id": "Attention_Is_All_You_Need",
  "paper_title": "...",
  "source_file": "cache/papers/Attention_Is_All_You_Need/extracted_claims.json",
  "extractor_version": "atomic_claim_v1",
  "model": "inclusionai/ling-3.1-flash:free",
  "role": "fast",
  "generated_at": "2026-10-13T17:00:00",
  "range": {"start": 0, "limit": 10},
  "summary": {
    "parents_processed": 10,
    "parents_ok": 10,
    "parents_failed": 0,
    "atomic_claims_total": 0,
    "requires_review_total": 0,
    "unassigned_keys_total": 0,
    "repaired_spans_total": 0,
    "errors_total": 0,
    "warnings_total": 0
  },
  "items": [
    {
      "parent_claim_id": "claim_001",
      "context_id": "ctx_001",
      "source_section": "1 Introduction",
      "content_type": "prose",
      "paragraph": "...",
      "expected_citation_keys": ["[13]", "[7]"],
      "matched_references": {"[13]": "..."},
      "status": "ok",
      "attempts": 1,
      "validation": {"valid": true, "errors": [], "warnings": [], "repaired_spans": 2},
      "unassigned_citation_keys": [],
      "claims": [
        {
          "claim_id": "claim_001.a01",
          "parent_claim_id": "claim_001",
          "context_id": "ctx_001",
          "source_spans": [{"start": 0, "end": 60, "text": "..."}],
          "verbatim_claim": "...",
          "normalized_claim": "...",
          "citation_keys": ["[13]"],
          "relationship": "background",
          "requires_review": false,
          "review_reason": ""
        }
      ],
      "error": "",
      "raw_response": ""
    }
  ]
}
```

JSON 一律 `ensure_ascii=False, indent=2`；写文件用 `encoding="utf-8", newline="\n"`；先写 `.tmp` 再 `replace`（与 `writer.py` 一致）。

**`atomic_claims_review.md`**（人工验收用）：每个 parent 一节，包含
① `claim_id` + `source_section` + status；② 原段落（引用块）；③ 期望引用键清单；④ 主张表格 `| claim_id | relationship | citation_keys | normalized_claim | review |`；⑤ 该 parent 的 errors / warnings 列表（有才写）；⑥ 若有 `unassigned_citation_keys` 或 `raw_response`，原样附上。文件开头放一张全局汇总表。

### 3.7 `tests/test_claim_validator.py`

离线单测（**不联网、不调 LLM**，直接构造 `AtomicClaimResponse`）。至少覆盖：

1. 正常通过：一段含 2 个引用标记的合成段落，键与 span 全部正确 → `valid=True, errors=[]`。
2. 编造引用：claim 里出现 `expected_keys` 之外的键 → `valid=False`，errors 含「新增不存在的引用」。
3. 遗漏引用：`expected_keys` 有一个键既没被引用也没进 `unassigned_citation_keys` → `valid=False`，errors 含「遗漏引用」。
4. 合法未归属：该键进了 `unassigned_citation_keys` → `valid=True`，warnings 含「未归属引用」。
5. 下标错但文本对：`start/end` 故意偏移 → `valid=True`，`repaired_spans >= 1`，且 span 的 `start/end` 已被就地修正为正确值（断言修正后的 `paragraph[start:end] == text`）。
6. 文本捏造：`span.text` 不在段落中 → `valid=False`。
7. 数字篡改：段落写 `24.6`，`normalized_claim` 写 `25.6` → `valid=False`，errors 含「原文没有的数字」。
8. 引用编号里的数字不算篡改：`normalized_claim` 含 `[13]` 而段落数字集合里没有裸 `13` 时，不得误报（验证规则 7 的抹标记逻辑）。

### 3.8 `citetrace_pdf/storage/writer.py`

第 11 行改为：

```python
KEEP_ROOT_MDS = {"verification_report.md", "extracted_claims.md", "atomic_claims_review.md", "readme.md"}
```

---

## 4. 执行顺序与验证命令

在 WSL 里执行（Windows 侧用 `wsl.exe -d Ubuntu-24.04 bash -lc "cd /root/citetrace && <命令>"`）：

```bash
cd /root/citetrace

# ① 离线单测先绿
./.venv/bin/python -m pytest tests/test_claim_validator.py -q

# ② 不联网，肉眼确认 prompt 拼装正确（占位符必须全部被替换，不得残留 {{...}}）
./.venv/bin/python -m scripts.extract_atomic_claims \
  --paper-folder cache/papers/Attention_Is_All_You_Need --limit 1 --dry-run

# ③ 单条真跑（claim_001 是 8 引用复合句，最难，先啃它）
./.venv/bin/python -m scripts.extract_atomic_claims \
  --paper-folder cache/papers/Attention_Is_All_You_Need --limit 1

# ④ 打开 atomic_claims_review.md 人工看一遍，确认拆解粒度与引用绑定合理后再跑满 10 条
./.venv/bin/python -m scripts.extract_atomic_claims \
  --paper-folder cache/papers/Attention_Is_All_You_Need --limit 10

# ⑤ 回归：确认没有碰坏已有测试
./.venv/bin/python -m pytest tests -q
```

③ 之后必须先自查再进 ④。若 ③ 的 `status` 不是 `ok`，先读 `atomic_claims.json` 里该 item 的 `validation.errors` 和 `raw_response`，判断是 prompt 表述问题还是 validator 规则实现问题，**改完再重跑 ③**，不要直接跳到 ④。

---

## 5. 验收标准（全部满足才算完成）

- [ ] `pytest tests/test_claim_validator.py -q` 全绿
- [ ] `pytest tests -q` 无新增失败
- [ ] `--dry-run` 输出的 prompt 中不残留任何 `{{...}}` 占位符
- [ ] `--limit 10` 跑完，10 个 parent 的 `status` 均为 `ok`（允许个别 `skipped_no_citation`）
- [ ] `atomic_claims.json` 的 `summary.errors_total == 0`
- [ ] `summary.parents_failed == 0`，即**没有凭空增加的引用、没有遗漏的引用、没有数字篡改**
- [ ] 每条主张的 `citation_keys` 都带方括号且能在 `expected_citation_keys` 中找到
- [ ] `claim_001` 的 8 个键全部有归属（在 claims 里或在 unassigned 里），且 `[13]`（LSTM）与 `[7]`（GRU）**没有被合并进同一条主张**——这是本阶段最关键的能力信号
- [ ] 不确定的归属被标了 `requires_review=true` 且 `review_reason` 非空
- [ ] `atomic_claims_review.md` 已生成，人工可读
- [ ] `repaired_spans_total` 可以大于 0（正常现象），但规则 4/6 的硬错误必须为 0

---

## 6. 本轮禁止事项

- 不要实现证据检索、BM25、chunks 匹配、`retrieval/` 目录
- 不要实现引用真伪判决、`CitationVerdict` 相关逻辑
- 不要引入 LangGraph / 多 Agent / 向量数据库 / 任何新依赖
- 不要改 `.env`、不要改 `llm_config.py` 里已有方法的行为、不要改 `extracted_claims.json`
- 不要生成最终报告、不要写 README、不要新建本任务书之外的 `.md`
- 不要为了让校验通过而放宽 §3.4 的规则 1/2/7，或篡改 `expected_keys`
- 不要给 LLM 传 `claim_id` 让它自己编号（ID 一律程序盖章）
- 不要在 prompt 文件里添加填好值的示例主张

---

## 7. 完成后向用户回报什么

1. 实际执行的命令与退出码。
2. `atomic_claims.json` 的 `summary` 全文。
3. `claim_001` 的完整拆解结果（所有 `normalized_claim` + `citation_keys` + `relationship` + `requires_review`），逐条列出。
4. `warnings` 的分类计数（哪些是下标修复、哪些是未归属引用、哪些是空白差异）。
5. 你在实现中对本文档做出的**任何**偏离，以及原因。没偏离就明确说"无偏离"。
6. `atomic_claims_review.md` 的路径。
