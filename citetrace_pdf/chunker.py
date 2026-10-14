# citetrace_pdf/chunker.py
"""
段落级证据切块 (Evidence Chunking)

把一篇论文专属文献夹中的各章节 Markdown 切分为带稳定 ID 的 chunk，
并落地为 chunks.json，供后续证据检索与大模型引文定位使用。

设计目标：
1. 每个 chunk 携带稳定 ID (document_id::section_file::序号)、章节标题与页码，
   大模型给出的任何证据引文都必须能回溯到具体 chunk；
2. 纯文本处理、零外部依赖（不依赖 pymupdf），可独立测试；
3. 切块参数：短段合并 (~400 字符起)，长段滑窗 (~2200 字符, 重叠 ~200)，
   对应约 100~600 tokens 的证据粒度；
4. Markdown 表格是原子证据块：不与散文合并、不从行中间切断，超长时按行分片
   且每片重复表头，带上紧邻的表题 (Table N: ...)，并严格保持在章节原文中的出现位置。
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import pymupdf
from .pdf.table_reconstructor import reconstruct_tables_in_markdown
from typing import Any, Dict, List, Optional

# 表题形如 "Table 2: ..."、pymupdf4llm 产出的 "_Table 1._ ..." 或 "**Table 1**"
_INTRO_VERBS = {
    "lists", "shows", "presents", "reports", "illustrates", "compares",
    "gives", "summarizes", "contains", "displays", "depicts", "provides",
    "demonstrates", "indicates", "describes", "details", "is", "are", "were", "was"
}

_TABLE_CAPTION_PREFIX_RE = re.compile(
    r"^(?:"
    r"[\*_]{1,2}\s*Table\s*\d+\.?[\*_]{1,2}(?:\s*[:.\-–—]|\s+(?P<word>[A-Za-z]+)|\s*$)|\s*"
    r"Table\s*\d+\s*(?:[:.\-–—]|\s+(?P<word2>[A-Za-z]+)|\s*$)"
    r")",
    re.I,
)
_TABLE_CAPTION_RE = _TABLE_CAPTION_PREFIX_RE


def is_table_caption(text: str) -> bool:
    """判断一段文本是否为表格标题（表题），严格区分叙述性引导句。"""
    t = text.strip()
    if not t:
        return False
    # 引导句典型特征：以冒号结尾，如 "Table 1 lists the translation accuracy across benchmarks:"
    if t.endswith(":") and not re.match(r"^[\*_]{0,2}\s*Table\s*\d+[\*_]{0,2}\s*:$", t, re.I):
        return False
    m = _TABLE_CAPTION_PREFIX_RE.match(t)
    if not m:
        return False
    word = m.group("word") or m.group("word2")
    if word and word.lower() in _INTRO_VERBS:
        return False
    return True


_TABLE_SEPARATOR_RE = re.compile(r"^\|[\s\-:|]+\|$")
_NOISE_LINE_RE = re.compile(
    r"^\s*(?:---+|#{1,6}|###\s*本章子章节导引|本章节具体内容请查阅各子章节.*)\s*$"
)


def strip_md_header(section_text: str) -> str:
    """剥离章节 Markdown 文件头部的元信息行与尾部的子章节导引块"""
    lines: List[str] = []
    past_header = False
    for line in section_text.splitlines():
        s = line.strip()
        if not past_header:
            if s.startswith("#") or s.startswith(">") or s == "---" or not s:
                continue
            past_header = True
        lines.append(line)
    body = "\n".join(lines).strip()
    # 剥离 writer.py 注入的尾部子章节导引块
    body = re.split(r"\n---\n+\s*###\s*本章子章节导引\b", body, maxsplit=1)[0]
    body = re.split(r"\n*###\s*本章子章节导引\b", body, maxsplit=1)[0]
    return body.strip()


def _split_paragraphs(section_text: str) -> List[str]:
    """剥离章节文件头部的元信息行，按双换行拆自然段

    Markdown 表格（整块以 | 开头）作为一个完整段落保留，交由表格切块逻辑
    单独处理；此处过滤导引章节卡片、纯链接列表与残留分隔符/空标题行。
    """
    body = strip_md_header(section_text)
    raw_paras = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    cleaned_paras: List[str] = []
    for p in raw_paras:
        if (
            p.startswith("> **章节类型")
            or p.startswith("- [")
            or p.startswith("### 本章子章节导引")
            or p.startswith("本章节具体内容请查阅各子章节")
        ):
            continue
        if p.startswith("|"):
            cleaned_paras.append(p)
            continue
        clean_lines = [
            ln for ln in p.splitlines()
            if not _NOISE_LINE_RE.match(ln)
            and not ln.strip().startswith("> **章节类型")
            and not ln.strip().startswith("- [")
        ]
        cp = "\n".join(clean_lines).strip()
        if cp:
            cleaned_paras.append(cp)
    return cleaned_paras


def _iter_ordered_blocks(paras: List[str]) -> List[tuple[str, str]]:
    """按原文顺序将段落流组织为 [('prose', text) | ('table', table_with_caption), ...]。

    pymupdf4llm 常把表题放在表格之后，手写 Markdown 则放在之前，两个方向都要认。
    表题并入所属表格块顶部，同时保持表格在章节中的原始位置不变。
    """
    is_table = [p.startswith("|") for p in paras]
    caption_for: Dict[int, int] = {}  # 表格段落下标 -> 表题段落下标
    claimed: set[int] = set()
    for i, p in enumerate(paras):
        if is_table[i] or not is_table_caption(p):
            continue
        for t in (i + 1, i - 1):
            if 0 <= t < len(paras) and is_table[t] and t not in claimed:
                caption_for[t] = i
                claimed.add(t)
                break

    captions = set(caption_for.values())
    ordered: List[tuple[str, str]] = []
    for i, p in enumerate(paras):
        if i in captions:
            continue
        if is_table[i]:
            c = caption_for.get(i)
            ordered.append(("table", f"{paras[c]}\n{p}" if c is not None else p))
        else:
            ordered.append(("prose", p))
    return ordered


def _split_table_blocks(paras: List[str]) -> tuple[List[str], List[str]]:
    """把段落流拆成 (散文段落, 表格块)，并把紧邻的表题并入所属表格块。"""
    ordered = _iter_ordered_blocks(paras)
    prose = [text for kind, text in ordered if kind == "prose"]
    tables = [text for kind, text in ordered if kind == "table"]
    return prose, tables


def _split_table(block: str, max_chars: int = 2200) -> List[str]:
    """表格原子切块：能整块装下就整块输出，超长则按行分片且每片重复表头。

    绝不从行中间硬切——半个单元格比没有表格更糟，会把数值证据变成噪声。
    """
    lines = block.splitlines()
    prefix: List[str] = []
    while lines and not lines[0].lstrip().startswith("|"):
        head_line = lines.pop(0).strip()
        if head_line:
            prefix.append(head_line)
    if not lines:
        return []

    header = [lines[0]]
    rows = lines[1:]
    if rows and _TABLE_SEPARATOR_RE.match(rows[0].strip()):
        header.append(rows.pop(0))
    head = "\n".join(prefix + header)

    if not rows:
        return [head]

    pieces: List[str] = []
    buf: List[str] = []
    size = len(head)
    for row in rows:
        if buf and size + len(row) + 1 > max_chars:
            pieces.append("\n".join([head] + buf))
            buf, size = [], len(head)
        buf.append(row)
        size += len(row) + 1
    if buf:
        pieces.append("\n".join([head] + buf))
    return pieces


def _merge_short(paras: List[str], min_chars: int = 400) -> List[str]:
    """将过短的自然段合并，避免碎片化 chunk"""
    merged: List[str] = []
    buf = ""
    for p in paras:
        buf = f"{buf}\n{p}".strip() if buf else p
        if len(buf) >= min_chars:
            merged.append(buf)
            buf = ""
    if buf:
        if merged and len(buf) < min_chars:
            merged[-1] = f"{merged[-1]}\n{buf}"
        else:
            merged.append(buf)
    return [m for m in merged if len(m) >= 80]


def _split_long(text: str, max_chars: int = 2200, overlap: int = 200) -> List[str]:
    """超长段落滑窗切分，尽量在句子边界断开并保留重叠"""
    if len(text) <= max_chars:
        return [text]
    pieces: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        if end < len(text):
            cut = max(text.rfind(". ", start, end), text.rfind(".\n", start, end))
            if cut > start + max_chars // 2:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return pieces


def chunk_paper_folder(folder: str | Path, max_chars: int = 2200) -> List[Dict[str, Any]]:
    """
    主入口：读取文献夹 metadata.json 与各章节 Markdown，
    生成段落级 chunk 并写入 chunks.json，返回 chunk 列表。
    """
    folder = Path(folder)
    metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
    doc_id = metadata.get("paper_id") or folder.name
    doc_sha256 = metadata.get("pdf_sha256")
    from citetrace_pdf.storage.cache import MetadataCache
    chunker_ver = MetadataCache.CHUNKER_VERSION
    chunks: List[Dict[str, Any]] = []

    pdf_path = folder / "raw.pdf"
    pdf_doc = pymupdf.open(pdf_path) if pdf_path.is_file() else None

    def _append_section_chunks(
        fname: str, title: str, page_start: Optional[int], page_end: Optional[int]
    ) -> None:
        sec_path = folder / "sections" / fname
        if not sec_path.is_file():
            sec_path = folder / fname
        if not sec_path.is_file():
            return

        raw_sec_text = sec_path.read_text(encoding="utf-8")
        if "|" in raw_sec_text and "---" in raw_sec_text:
            sec_pages = None
            if pdf_doc is not None and page_start and page_end:
                s_p = max(1, int(page_start))
                e_p = min(len(pdf_doc), int(page_end))
                if s_p <= e_p:
                    sec_pages = [pdf_doc[p - 1] for p in range(s_p, e_p + 1)]
            raw_sec_text = reconstruct_tables_in_markdown(raw_sec_text, pages=sec_pages)

        ordered_blocks = _iter_ordered_blocks(
            _split_paragraphs(raw_sec_text)
        )
        counter = 0

        def _emit(
            text: str,
            *,
            is_prose: bool,
            caption: Optional[str] = None,
            table_id: Optional[str] = None,
            oversized: bool = False,
            oversized_reason: Optional[str] = None,
        ) -> None:
            nonlocal counter
            seq_idx = len(chunks)
            counter += 1
            chunk_dict: Dict[str, Any] = {
                "chunk_id": f"{doc_id}::{fname}::{counter:03d}",
                "chunk_content_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "document_id": doc_id,
                "document_sha256": doc_sha256,
                "section_id": fname,
                "section_title": title,
                "section_file": fname,
                "sequence_index": seq_idx,
                "page_start": page_start,
                "page_end": page_end,
                "content_type": "prose" if is_prose else "table",
                "text": text,
                "char_count": len(text),
                "chunker_version": chunker_ver,
            }
            if not is_prose:
                if caption:
                    chunk_dict["caption"] = caption
                if table_id:
                    chunk_dict["table_id"] = table_id
            if oversized:
                chunk_dict["oversized"] = True
                if oversized_reason:
                    chunk_dict["oversized_reason"] = oversized_reason
            chunks.append(chunk_dict)

        def _flush_prose() -> None:
            nonlocal prose_buf
            if not prose_buf:
                return
            merged = _merge_short(prose_buf)
            if merged:
                for para in merged:
                    for piece in _split_long(para, max_chars):
                        _emit(piece, is_prose=True)
            else:
                text = "\n".join(prose_buf).strip()
                if text:
                    for piece in _split_long(text, max_chars):
                        _emit(piece, is_prose=True)
            prose_buf = []

        prose_buf: List[str] = []
        for kind, block in ordered_blocks:
            if kind == "prose":
                prose_buf.append(block)
            else:
                # 遇到表格：表格作为硬性顺序屏障！
                # 强制输出前面所有散文，短句即便小于 80 字符也严禁跨越表格与表后散文合并
                _flush_prose()

                # 提取表题与 table_id（若存在）
                lines = block.splitlines()
                prefix = []
                for ln in lines:
                    if ln.lstrip().startswith("|"):
                        break
                    if ln.strip():
                        prefix.append(ln.strip())
                caption = "\n".join(prefix) if prefix else None
                table_id = None
                if caption:
                    m = re.search(r"Table\s*(\d+)", caption, re.I)
                    if m:
                        table_id = f"table_{m.group(1)}"

                for piece in _split_table(block, max_chars):
                    oversized = len(piece) > max_chars
                    _emit(
                        piece,
                        is_prose=False,
                        caption=caption,
                        table_id=table_id,
                        oversized=oversized,
                        oversized_reason="single_table_row" if oversized else None,
                    )

        # 章节末尾输出剩余散文
        _flush_prose()

    try:
        for item in metadata.get("outline", []):
            fname = item.get("file", "")
            if fname.lower() in ("references.md", "other.md"):
                continue
            _append_section_chunks(
                fname, item.get("title", ""), item.get("start_page"), item.get("end_page")
            )

        if not chunks:
            _append_section_chunks("other.md", "Main Content", 1, metadata.get("total_pages", 1))
    finally:
        if pdf_doc is not None:
            pdf_doc.close()

    out_path = folder / "chunks.json"
    out_path.write_text(
        json.dumps(chunks, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    meta_path = folder / "metadata.json"
    if meta_path.is_file():
        try:
            m_data = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(m_data, dict):
                from citetrace_pdf.storage.cache import MetadataCache
                m_data["chunker_version"] = MetadataCache.CHUNKER_VERSION
                meta_path.write_text(json.dumps(m_data, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:
            pass
    return chunks


# 引文定位的 token 化。数字必须整体成块并保住小数点：旧实现把小数点当格式噪声删掉，
# 于是 "24.6" 成了 "24.61" 的前缀，一个写错的数值能蒙过【逐字】校验（实测占证据数字的 35.7%）。
# 数字只允许一个分隔组，防止 [-0.2,0.2] 或 IP 串被贪婪吞成一个 token 而误判为不存在。
_QUOTE_TOKEN_RE = re.compile(
    r"\d{1,3}(?:,\d{3})+(?:\.\d+)?"   # 千分位: 3,028 / 1,234.5
    r"|\d+(?:\.\d+)?"                 # 普通数字: 24.61 / 40
    r"|[一-鿿]"                       # 汉字逐字，保留中文子串语义
    r"|[^\W\d_]+"                     # 拉丁等文字的整词
)


def _canon_number(token: str) -> str:
    """数值归一：去千分位逗号与末尾多余的 0，使 28.10 与 28.1 判等、24.6 与 24.61 判不等"""
    t = token.replace(",", "")
    if "." in t:
        t = t.rstrip("0").rstrip(".")
    return t or "0"


def _quote_tokens(text: str) -> List[str]:
    """切成可比对的 token：数字整体一块、拉丁词整体一块、汉字逐字一块，其余符号仅作分隔"""
    return [
        _canon_number(m.group(0)) if m.group(0)[0].isdigit() else m.group(0)
        for m in _QUOTE_TOKEN_RE.finditer(text.lower())
    ]


def _contains_tokens(hay: List[str], needle: List[str]) -> bool:
    """needle 须作为连续 token 序列出现 —— 边界必须对齐，不许从数字中间截半截"""
    n = len(needle)
    if n > len(hay):
        return False
    return any(hay[i:i + n] == needle for i in range(len(hay) - n + 1))


def locate_quote(chunks: List[Dict[str, Any]], quote: str) -> Optional[Dict[str, Any]]:
    """
    程序化引文校验：判断一句引文是否【逐字】存在于某个 chunk 中。
    供 validate 节点使用 —— 大模型给出的 key_quote 必须能通过本检验，
    否则视为幻觉证据并驳回判决。
    """
    q = _quote_tokens(quote or "")
    if not q:
        return None
    for c in chunks:
        if _contains_tokens(_quote_tokens(c.get("text", "")), q):
            return c
    return None
