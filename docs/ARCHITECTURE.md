# CiteTrace 代码结构导读 (极简版)

项目已全面清理冗余文件，**只保留真正参与 4 步核查主线的代码**。平时开发只看标 ⭐ 的 6 个核心文件即可。

---

## 一、 全流程 4 步走

```
输入论文 PDF
   │
   ▼
[第 1 步：拆解待查论文]  citetrace_pdf/processor.py  +  parsers/folder_extractor.py
   │  把 PDF 按章节切成 .md，提取正文里所有带 [1]、[2] 的引文句子，对齐文末参考文献表
   ▼
[第 2 步：下载并切分被引论文]  tools/academic_fetcher.py  +  citetrace_pdf/chunker.py
   │  联网下载被引论文 PDF → 切章节 .md → 再切成 400~2200 字符的证据块 (chunks.json)
   ▼
[第 3 步：LangGraph 闭环核查]  agents/graph.py (retrieve -> judge -> validate -> finalize)
   │  按关键词检索 Top-K 证据块 → 模型研判 → 程序逐字校验 key_quote（防幻觉）→ 终局判定
   ▼
[第 4 步：汇总生成报告]  verifier.py  (命令行入口: demo_verifier.py)
   │  输出 verification_report.md 与 verification_report.json
```

---

## 二、 目录全景（无任何冗余历史文件）

```text
/root/citetrace
├── demo_verifier.py          ⭐ [入口] 命令行一键运行脚本（选 PDF -> 调 verifier -> 打印汇总）
├── verifier.py               ⭐ [总控] 串联第 1~4 步的主控制器 CiteTraceVerifier
│
├── agents/
│   └── graph.py              ⭐ [核查核心] LangGraph 状态图 + OpenAICompatibleClient / HeuristicGraphClient
│
├── citetrace_pdf/            📦 [PDF 切分与切块包]
│   ├── processor.py          ⭐ 把任意 PDF 解析并落地成“文献夹”（各章节 .md + metadata.json）
│   ├── chunker.py            ⭐ 把文献夹里的章节切成带 chunk_id 的小段证据 (chunks.json)
│   ├── pdf/                  🔒 (黑盒) PyMuPDF 底层文本提取、单双栏检测、PDF 下载缓存
│   ├── parser/               🔒 (黑盒) 标题字号识别、章节切分、参考文献清洗格式化
│   ├── model/                🔒 (黑盒) PDF 内部块/行/文档数据结构
│   └── storage/              🔒 (黑盒) 文献夹 Markdown 与 metadata.json 落盘读写
│
├── parsers/                  🔍 [引文与参考文献提取]
│   ├── folder_extractor.py   ⭐ 从文献夹中提取每处引文上下文 (citations/1.md, 2.md...) 并对齐参考文献
│   ├── link_extractor.py     🔒 (底层) 利用 PDF 内部超链接热区精确提取引用锚点
│   └── citation_parser.py    🔒 (底层) 单条参考文献正则解析（提标题、年份、DOI、arXiv）
│
├── tools/
│   └── academic_fetcher.py   ⭐ [联网抓取] 按 arXiv -> OpenAlex -> S2 -> Crossref 搜寻并下载被引论文 PDF
│
├── models/                   📋 [数据字典] 仅 2 个文件
│   ├── citation.py              CitationRef（单条参考文献结构）与 CitationStatus
│   └── verification.py          CitationVerdict（6 种判决结论）与 FailureReason（9 种失败原因）
│
└── tests/                    🧪 [单元测试] 运行 `pytest -v` 即可验证全部离线测试
```
