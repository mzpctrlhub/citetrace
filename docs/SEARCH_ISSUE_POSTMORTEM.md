# 文献检索不完整问题排查与修复记录 (Postmortem)

> **记录时间**: 2026-10-04  
> **基准测试论文**: *Attention Is All You Need* (`1706.03762`, 共 35 处引用段落、40 篇被引文献、72 条引文断言)  
> **修复效果**: 被引文献获取率由 `33/40 (82.5%)` 提升至 **`40/40 (100%)`**；`证据不足 (Insufficient)` 由 `11` 降至 **`0`**；`检索未完成 (Search Incomplete)` 由 `15` 降至 **`0`**。

---

## 一、 问题现象

在运行 `python demo_verifier.py` 对《Attention Is All You Need》执行全量引用核查时，存在两类“文献搜索不完整”现象：

1. **外部文献抓取不完整（7 篇被引文献未获取到 PDF）**
   - 未能获取的文献标号：`[12]`、`[13]`、`[14]`、`[25]`、`[26]`、`[27]`、`[29]`。
   - 导致所有引用上述 7 篇文献的断言直接因缺失正文而落入 `⚠️ 证据不足 (Insufficient): 11`。
2. **单篇文献内部跨片段检索不完整（15 条引文报 `🔎 检索未完成 Search Incomplete`）**
   - 即使被引文献 PDF 已成功下载切分，仍有 15 条断言在达到最大检索轮次后未搜完所有证据块（例如提示 `达到最大检索轮次，仍有 59 个片段未检索`），另有 5 条断言被误判为 `🔴 数据不正确 (Data Error)`。

---

## 二、 根因剖析与逐项解决方案

### （一） 外部被引文献获取失败的 5 个根因与修复

| 序号 | 涉及文献 | 根本原因 | 解决方案与修改位置 |
| :---: | :--- | :--- | :--- |
| **1** | `[13]`, `[14]`, `[26]`, `[29]` | **多梯队检索流水线过早短路**：<br>`fetch_openalex_by_title` 在仅拿到 `doi` 但无可用开放 `pdf_url`（例如 `dl.acm.org` 付费墙链接被剔除后 `pdf_url=None`）时仍返回非空字典。`enrich_citation_with_cache` 仅判断 `if not paper_info:`，导致下游 Semantic Scholar、Crossref + Unpaywall 阶梯被全部跳过，并将 `"pdf_url": null` 写入 `.metadata_cache`。 | **修改 `tools/academic_fetcher.py` (`enrich_citation_with_cache`)**：<br>将短路条件改为 `_has_pdf()`（即 `bool(paper_info and paper_info.get("pdf_url"))`）。若前序阶梯仅拿到 `doi` 而无有效 `pdf_url`，保留 `ref.doi` 并继续向下游阶梯穿透检索。 |
| **2** | `[14]`, `[25]`, `[26]`, `[29]` | **早期计算语言学论文的 ACM 镜像付费墙与旧版链接**：<br>2015 年前的 ACL / EMNLP / NAACL / COLING / CL 论文在 Crossref/OpenAlex 中登记的是 `10.3115/...`（指向 `dl.acm.org` 付费墙），而 `[25]` 在 OpenAlex 中的链接为不带 `.pdf` 的旧版摘要页 `http://www.aclweb.org/anthology/J93-2004`。 | **修改 `tools/academic_fetcher.py`**：<br>1. 新增 `fetch_acl_anthology_by_title`，按会议与年份直查 **ACL Anthology 官方开放论文库** (`aclanthology.org/events/{venue}-{year}/`)，剥离标题内嵌 HTML 标签（如 `<span class=acl-fixed-case>PCFG</span>`）后匹配并返回官方直链 `https://aclanthology.org/{id}.pdf`。<br>2. 在 OpenAlex 候选链接提取与本地缓存读取时统一经过 `_normalize_pdf_url` 规范化。 |
| **3** | `[27]` | **参考文献省略副标题导致标题相似度被拒**：<br>原文将 `[27]` 简写为 *"A decomposable attention model"*，而 arXiv (`1606.01933`) 与 ACL Anthology 上的完整标题为 *"A Decomposable Attention Model for Natural Language Inference"*，对称 Jaccard 仅为 $3/6 = 0.50 < 0.75$。 | **修改 `tools/academic_fetcher.py` (`is_title_match`)**：<br>在保持词序严格一致的前提下，支持“主标题精确前缀 + `:` 或 `for` 副标题引导词（且主标题词数 $\ge 3$、覆盖率 $\ge 0.5$）”匹配，阶梯 3 直接从 arXiv 命中 `1606.01933v2.pdf`。 |
| **4** | `[12]`, `[13]` | **早期奠基文献商业出版商闭源 + 标题末尾年份粘连**：<br>`[12]`（*Gradient Flow in Recurrent Nets*, 2001）与 `[13]`（*Long Short-Term Memory*, 1997）在 IEEE/MIT Press 处为闭源状态；同时 `[12]` 因无独立期刊字段，解析出的标题末尾粘连了 `, 2001.`。 | **修改 `parsers/citation_parser.py` 与 `tools/academic_fetcher.py`**：<br>1. 在 `parse_reference_entry` 中用正则剥离标题末尾残留的 `, YYYY.`。<br>2. 新增 `fetch_jku_archive_by_title`，从作者 Sepp Hochreiter 的 JKU Linz 实验室公开归档库直接获取原文 PDF（`ch7.pdf` 与 `2604.pdf`）。 |
| **5** | `[29]` 等 | **OpenAlex 逗号触发 `HTTP 400` & Unpaywall 占位邮箱触发 `HTTP 422`**：<br>标题含英文逗号时拼入 OpenAlex `filter=title.search:...` 会被解析为多过滤条件分隔符并报 `HTTP 400`；Unpaywall API 拒绝 `academic_bot@example.com` 占位邮箱并返回 `HTTP 422`。 | **修改 `tools/academic_fetcher.py`**：<br>1. `fetch_openalex_by_title` 构造查询词前先做字母数字清洗。<br>2. `fetch_unpaywall_by_doi` 使用 `urllib.parse.quote(clean_doi, safe='/')` 与合法邮箱域名，并过滤付费墙网关链接。 |

---

### （二） 论文内部跨片段检索未完成 (`Search Incomplete`) 的 2 个根因与修复

1. **超长段落滑窗切块的尾部死循环碎片 Bug（最核心原因）**
   - **问题定位**：在 `citetrace_pdf/chunker.py` 的 `_split_long(text, max_chars=2200, overlap=200)` 中：
     ```python
     while start < len(text):
         end = min(start + max_chars, len(text))
         ...
         pieces.append(text[start:end].strip())
         start = max(end - overlap, start + 1)
     ```
     当滑窗到达段落末尾（`end == len(text)`）时，由于缺少 `if end >= len(text): break`，此后 `end - overlap` 恒定不变，一旦 `start >= len(text) - overlap`，`max(end - overlap, start + 1)` 就会退化为 `start + 1`，导致循环在段落最后 200 个字符上**逐字符递增空转 200 次**！
   - **后果**：任何包含超过 2200 字符长段落的论文都会凭空多出 200 个碎片 chunk（例如 `[19]`、`[35]`、`[38]` 被膨胀到近 300 个 chunk，`[13]` 甚至膨胀到 5061 个 chunk），导致有限检索轮次永远搜不完所有 chunk，最终落入 `SEARCH_INCOMPLETE`。
   - **修复方案**：在 `pieces.append(...)` 后立即判断 `if end >= len(text): break`，消除尾部重复切片。

2. **多句长段落对单句引用的重合度稀释与引用标号干扰**
   - **问题定位**：
     - 在 `agents/graph.py` 的 `HeuristicGraphClient` 中，原逻辑以整个引用段落的总词数（常达 45～60 个词）作为分母计算重合率，而单篇被引文献往往只支撑段落中的某一句话（重合 10～13 个实质词），导致得分如 $13/60 = 0.217 < 0.22$ 在第 1 轮被误拒。
     - 原逻辑直接用 `\b\d+(?:\.\d+)?\b` 提取段落中所有数字，将引用标号（如 `[34]`、`[29]`、`[37]`）或本论文超参数（如 `4-layer`、`dmodel = 1024`）当成待核查实验指标，误报 `data_error`。
   - **修复方案**：
     - 在 `HeuristicGraphClient` 中先剥离引用标号 `\[[^\]]+\]`，将重合度分母上限封顶，并仅对显式百分比指标（`\b\d+(?:\.\d+)?%\b`）执行数值冲突比对。
     - 将 `verifier.py` 与 `demo_verifier.py` 的默认检索参数提升为 `max_search_rounds=10, top_k_sections=5`。

---

## 三、 修复前后对比

| 指标项 | 修复前 | 修复后 |
| :--- | :---: | :---: |
| **被引文献成功抓取数** | `33 / 40` (缺 `[12]`, `[13]`, `[14]`, `[25]`, `[26]`, `[27]`, `[29]`) | **`40 / 40` (100%)** |
| **🟢 正确引用 (`Correct`)** | `41` | **`72`** |
| **🔴 数据不正确 (`Data Error`)** | `5` (引用标号误判) | **`0`** |
| **⚠️ 证据不足 (`Insufficient`)** | `11` | **`0`** |
| **🔎 检索未完成 (`Search Incomplete`)** | `15` | **`0`** |
| **单元测试 (`pytest -v`)** | `42 passed` | **`42 passed`** |


---

## 五、章节切分（Section Splitting）缺失与错位问题复盘（`[13]`、`[16]`、`[18]` 等）

在对 40 篇被引文献的 `metadata.json` 大纲逐篇审计后，发现 `[13]`、`[16]`、`[18]`（以及同类的 `[14]`、`[17]`、`[5]`、`[22]`、`[24]`、`[27]`、`[39]`）存在三类典型的 PDF 章节切分缺陷：

### 1. `[13]` (*Long Short-Term Memory*, 1997) —— 远古 `dvips` Type-3 点阵字体导致正文 1~8 节全部漏切
- **现象**：32 页的论文仅切出 `00_Abstract.md` 与 `References.md` 两个文件，`paper_title` 被误提为 `'nc.dvi'`，正文 `1 INTRODUCTION` 至 `8 ACKNOWLEDGMENTS` 及 `APPENDIX` 全部丢失。
- **根因**：
  1. **Type-3 控制字符编码**：1997 年由 `dvips` 生成的 Type-3 点阵字体未内嵌标准 Unicode CMap，数字 `1`~`9` 被编码为 ASCII 控制字符 `\x01`~`\x09`（如 `'\x01 INTRODUCTION'`、`'\x04 LONG SHORT-TERM MEMORY'`），连字符合字（`ff/fi/fl/ffi/ffl`）被编码为 `\x0b`~`\x0f`。
  2. **字距微调（Kerning）断行与虚拟字号失真**：所有字符的 `span["size"]` 均为 `0.12`，且纵向包围盒高度被硬编码为 `120pt`（`y1 - y0 = 120`）。`pymupdf4llm` 因垂直重叠将整页压成单行，而 `pymupdf` 原生行提取则把同一视觉行内的每一次 TeX 字距微调拆成了独立行（如 `'SHOR\nT-TERM'`、`'INTR\nODUCTION'`、`'PREVIOUS\nW\nORK'`），导致章节正则完全无法匹配。
- **修复方案**（`citetrace_pdf/pdf/extractor.py`、`citetrace_pdf/parser/cleaner.py`）：
  - 新增 `_is_type3_dvips_doc()` 自动检测 `Type3` 且 `size < 1.0`（或含 `\x01`~`\x08` 控制字符）的旧版 TeX PDF；
  - 新增 `clean_dvips_text()` 将 `\x01`~`\x09` 还原为 `'1'`~`'9'`，并将 `\x0b`~`\x0f` 还原为 `ff/fi/fl/ffi/ffl`；
  - 新增 `_extract_type3_page()` 按视觉行坐标 `y0` 聚类，并根据水平间距 `x0 - prev_x1` 智能区分词内字距微调（`< 1.5pt` 直接拼接，如 `'SHOR' + 'T-TERM' -> 'SHORT-TERM'`）与词间空格（`>= 1.5pt`），将 `[13]` 完整恢复为 **22 个章节（含 `1`~`8` 节、`3.1`~`3.2`、`5.1`~`5.7`、`APPENDIX`、`A.1`、`A.2` 与 `References`）**。

### 2. `[16]` (*Can Active Memory Replace Attention?*) —— 图表序号 `5. Output` 污染单调性守卫导致第 2 节漏切
- **现象**：大纲从 `1 Introduction` 直接跳到伪章节 `5. Output`，随后是 `3 Experiments`、`4 Discussion`，真正的 **`2 Active Memory Models`（及 `2.1`、`2.2`、`2.3`）全部丢失**。同理，`[17]` 在 `3.2` 后混入伪章节 `1. Output`，`[5]` 在 `4.2` 后混入列表项 `2. Baseline + RNN`、`3. Baseline + CSLM + RNN`、`4. Baseline + CSLM + RNN`。
- **根因**：
  1. `[16]` 第 2 页架构示意图中有独立数字行 `"5"` 紧邻下一行 `"Output"`，`SectionParser._detect_line_heading` 将其拼接为 `"5. Output"`；
  2. `_MonotonicGuard` 原先设置 `max_jump = 4`，在 `1 Introduction`（`_max_major_seen = 1`）后接受了 `major = 5`，将 `_max_major_seen` 抬高污染至 `5`；
  3. 当第 3 页真正的 `# 2 Active Memory Models`（`major = 2`）到达时，触发 `major < self._max_major_seen - 2`（`2 < 5 - 2`），导致真章节被单调性守卫误杀！
  4. 此外，原 `_MonotonicGuard` 未区分一级章节（`len(parts) == 1`）与子章节，允许一级章节号小于等于 `_max_major_seen`，导致 `[17]` 的 `1. Output` 和 `[5]` 的 `2./3./4. Baseline` 列表项混入大纲。
- **修复方案**（`citetrace_pdf/parser/section.py`、`citetrace_pdf/parser/heading.py`）：
  - 收紧 `_MonotonicGuard`：一级主章节（`len(parts) == 1`）必须**严格递增**（`major > _max_major_seen`）且跳跃步长不超过 `max_jump = 2`；首个编号章节必须从 `1` 或 `2` 开始；
  - 在 `_detect_line_heading` 跨行拼接数字与下一行时，排除 `Input`/`Output`/`State` 等常见示意图标签；
  - 限制字母型附录标题（`A`~`G`）必须区分大小写且仅在正文已推进至 `>= 2` 节后生效，消除了 `[22]`、`[27]` 以 `"A ..."` 开头的论文主标题及 `[24]`、`[39]` 正文公式碎片被误判为附录的问题。

### 3. `[18]` (*Neural Machine Translation in Linear Time*) 与 `[14]` —— 加粗章节标题被合并进段落内部导致第 6 节 / 第 4 节漏切
- **现象**：`[18]` 从 `5. Character Prediction` 直接跳到 `7. Conclusion`，**漏掉了第 6 页的 `6. Character-Level Machine Translation`**；`[14]` 从 `3 Parser Self-Training` 直接跳到 `5 Results`，**漏掉了第 4 页的 `4 Experimental Setup`**。
- **根因**：
  - `pymupdf4llm.to_markdown` 在双栏排版或紧凑段间距下，将加粗章节标题直接内联拼接到了前一段落或底部脚注的同一行文本中间：
    - `[18]` 第 6 页：`"...achieves 1.31 bits/character on the test set. **6. Character-Level Machine Translation** We evaluate the full ByteNet..."`
    - `[14]` 第 4 页：`"> 4This amount was constrained... to exploit **4 Experimental Setup** more unlabeled data."`
  - 由于 `SectionParser.parse` 仅按 `\n` 切分，嵌在段落中间的 `**6. Character-Level Machine Translation**` 无法作为独立行进入 `_detect_line_heading`。
- **修复方案**（`citetrace_pdf/parser/section.py`）：
  - 在 `SectionParser.parse` 逐块拆行前，通过正则 `_INLINE_BOLD_HEADING` 将内嵌在段落中间的加粗编号章节标题（`**\d{1,2}(?:\.\d+)*\.?\s+[A-Z]...**`）自动前插/后插换行符拆分为独立行，使 `[18]` 的第 6 节（`6. Character-Level Machine Translation`）与 `[14]` 的第 4 节（`4 Experimental Setup`）完整切出。
