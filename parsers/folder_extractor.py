# citetrace/parsers/folder_extractor.py
"""
从已落地的专属文献夹中，逐章节扫描正文引用段落，
通过序列号 [n] 或作者-年份 [Author, Year] 精准对齐到 References.md 中的参考文献条目，
最终在文献夹下生成：
  - extracted_claims.json  (结构化数据，供 Layer 4/5 自动化核查)
  - extracted_claims.md    (人类可读清单：段落 + 对应参考文献卡片)

双引擎支持：
1. Tier 1（首选）：若存在 raw.pdf，直接通过 PDF 矢量超链接层进行像素级锚定对齐 (100% 精度)
2. Tier 2（兜底）：通过 Markdown 章节段落与 References.md 进行语义与正则匹配对齐
"""
import re
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

from citetrace_pdf.chunker import strip_md_header
from .citation_parser import (
    YEAR_PATTERN,
    DOI_PATTERN,
    ARXIV_PATTERN,
    parse_reference_entry,
)
from .link_extractor import extract_citations_from_pdf_links


def pdf_sha256(path: Path) -> Optional[str]:
    """计算 PDF 文件 SHA-256 指纹，用于数据契约溯源（文件缺失或读取失败返回 None）"""
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for block in iter(lambda: f.read(1 << 20), b""):
                h.update(block)
        return h.hexdigest()
    except Exception:
        return None

logger = logging.getLogger(__name__)

# 匹配正文中出现的 [n]、[n, m]、[n-m] 或 [Author et al., 2015] 等引用标记
IN_TEXT_CITATION_RE = re.compile(
    r"\[([0-9\s,\-]+|[A-Z][^\]\n]*(?:19|20)\d{2}[a-z]?[^\]\n]*)\]"
)


def expand_citation_keys(raw: str) -> List[str]:
    """
    将复杂引用符号展开为独立标准 key 列表
    例：
        '1, 2'  -> ['[1]', '[2]']
        '1-3'   -> ['[1]', '[2]', '[3]']
        '13, 7' -> ['[13]', '[7]']
        'Laurent et al., 2015, Amodei et al., 2015' -> ['[Laurent et al., 2015]', '[Amodei et al., 2015]']
    """
    cleaned = raw.strip()
    if not cleaned:
        return []

    # 1. 纯数字或数字范围
    if re.match(r"^[\d\s,\-]+$", cleaned):
        keys: List[str] = []
        parts = [p.strip() for p in cleaned.split(",") if p.strip()]
        for part in parts:
            if "-" in part:
                ends = [r.strip() for r in part.split("-") if r.strip()]
                if len(ends) == 2 and ends[0].isdigit() and ends[1].isdigit():
                    for n in range(int(ends[0]), int(ends[1]) + 1):
                        keys.append(f"[{n}]")
                    continue
            if part.isdigit():
                keys.append(f"[{part}]")
        return list(dict.fromkeys(keys))  # 保序去重

    # 2. 作者-年份复合引用 (可能由分号或逗号分隔)
    sub_cits = re.split(r";\s*|,\s*(?=[A-Z][a-zA-Z]+(?:\s+et\s+al|\s+and|\s*,))", cleaned)
    keys = []
    for sc in sub_cits:
        sc_clean = sc.strip().rstrip(",;")
        if sc_clean:
            keys.append(f"[{sc_clean}]")
    return list(dict.fromkeys(keys))


def _make_catalog_item(
    idx: int, canonical_id: str, content: str, custom_key: Optional[str] = None
) -> Dict[str, Any]:
    parsed = parse_reference_entry(content)
    inferred_key = (
        parsed.key
        if parsed and parsed.key and "Author" not in parsed.key and "n.d." not in parsed.key
        else None
    )
    ay_key = custom_key or inferred_key

    year = parsed.year if parsed else None
    year_suffix = None
    for src in (ay_key, canonical_id, content):
        if src:
            m_ys = YEAR_PATTERN.search(src)
            if m_ys:
                if not year:
                    try:
                        year = int(m_ys.group(1))
                    except ValueError:
                        pass
                if m_ys.group(2):
                    year_suffix = m_ys.group(2)
                    break

    doi = parsed.doi if parsed else None
    if not doi:
        m_doi = DOI_PATTERN.search(content)
        if m_doi:
            doi = m_doi.group(1).strip().rstrip(".,;")

    arxiv_id = parsed.arxiv_id if parsed else None
    if not arxiv_id:
        m_arx = ARXIV_PATTERN.search(content)
        if m_arx:
            arxiv_id = m_arx.group(1).strip().rstrip(".,;")

    return {
        "index": idx,
        "canonical_id": canonical_id,
        "author_year_key": ay_key,
        "title": (parsed.title or "") if parsed else "",
        "raw_text": content,
        "authors": parsed.authors if parsed else [],
        "year": year,
        "year_suffix": year_suffix,
        "doi": doi,
        "arxiv_id": arxiv_id,
    }


def parse_reference_catalog_from_md(md_path: Path) -> List[Dict[str, Any]]:
    """
    从 References.md 中解析出有序结构化条目清单 (每条包含 index, canonical_id, author_year_key, title, raw_text)。
    可直接序列化落地为 references.json，供大模型核查及前端展示。
    """
    if not md_path.exists():
        return []

    body = strip_md_header(md_path.read_text(encoding="utf-8"))
    if not body:
        return []

    # 预清理 Markdown 列表符号（如 '- [1]'）
    body = re.sub(r"(?:^|\n)\s*[-*•]\s*(?=\[\d+\])", "\n", body)

    catalog: List[Dict[str, Any]] = []
    seen_canonical_ids: set[str] = set()
    seen_fingerprints: set[str] = set()

    if re.search(r"\[\d+\]", body):
        for seg in re.split(r"(?=\[\d+\])", body):
            seg = seg.strip()
            if not seg:
                continue
            m = re.match(r"^(\[\d+\])\s*(.*)", seg, re.DOTALL)
            if not m:
                continue
            key = m.group(1)
            num_match = re.search(r"\d+", key)
            num = int(num_match.group(0)) if num_match else len(catalog) + 1
            content = re.sub(r"\s*\n\s*", " ", m.group(2)).strip()
            content = re.sub(r"^(?:[-*•]\s*|\d+\.\s*)+", "", content).strip()
            content = re.sub(r"(?:\s*[-*•]\s*)+$", "", content).strip()

            custom_key = None
            inner_tag = re.match(r"^(\[[^\]]+\])\s*(.*)", content)
            if inner_tag:
                custom_key = inner_tag.group(1).strip()
                content = inner_tag.group(2).strip()

            item = _make_catalog_item(num, key, content, custom_key)
            canon_id = item["canonical_id"]
            fp = re.sub(r"\W+", "", content.lower())[:80]
            if canon_id in seen_canonical_ids or (len(fp) >= 15 and fp in seen_fingerprints):
                continue
            seen_canonical_ids.add(canon_id)
            if len(fp) >= 15:
                seen_fingerprints.add(fp)

            catalog.append(item)
    else:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
        idx = 1
        for p in paragraphs:
            content = re.sub(r"\s*\n\s*", " ", p).strip()
            content = re.sub(r"^(?:[-*•]\s*|\d+\.\s*)+", "", content).strip()
            content = re.sub(r"(?:\s*[-*•]\s*)+$", "", content).strip()
            if content:
                fp = re.sub(r"\W+", "", content.lower())[:80]
                if len(fp) >= 15 and fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)
                catalog.append(_make_catalog_item(idx, f"[{idx}]", content))
                idx += 1

    return catalog


def parse_references_from_md(
    ref_md_path: Path, catalog: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, str]:
    """
    从 References.md 中解析出 {序列号或作者-年份Key: 原始文献条目全文} 的字典。
    直接复用 parse_reference_catalog_from_md 与 _build_catalog_lookup，无需二次读取文件。
    """
    if catalog is None:
        catalog = parse_reference_catalog_from_md(ref_md_path)
    return {k: item["raw_text"] for k, item in _build_catalog_lookup(catalog).items()}


def extract_cited_paragraphs(
    section_path: Path,
    section_title: str
) -> List[Dict]:
    """
    扫描一个章节 Markdown 文件，提取所有包含引用的段落（含正文散文与 Markdown 表格行）。
    返回结构化清单，表格引用包含 content_type="table", table_caption, table_headers。
    """
    if not section_path.exists():
        return []

    raw_text = section_path.read_text(encoding="utf-8")
    from citetrace_pdf.chunker import _split_paragraphs, _iter_ordered_blocks
    ordered_blocks = _iter_ordered_blocks(_split_paragraphs(raw_text))

    results = []
    for kind, block in ordered_blocks:
        if kind == "prose":
            if block.startswith("> **章节类型") or block.startswith("- ["):
                continue
            matches = IN_TEXT_CITATION_RE.findall(block)
            if not matches:
                continue
            all_keys: List[str] = []
            for raw_m in matches:
                all_keys.extend(expand_citation_keys(raw_m))
            all_keys = list(dict.fromkeys(all_keys))
            if not all_keys:
                continue
            results.append({
                "content_type": "prose",
                "paragraph": block,
                "citation_keys": all_keys,
                "source_section": section_title,
                "source_file": section_path.name,
            })
        elif kind == "table":
            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            caption_lines: List[str] = []
            table_lines: List[str] = []
            for ln in lines:
                if ln.startswith("|"):
                    table_lines.append(ln)
                elif not table_lines:
                    caption_lines.append(ln)
            caption = " ".join(caption_lines).strip() if caption_lines else None

            if len(table_lines) >= 2:
                header_line = table_lines[0]
                headers = [c.strip() for c in header_line.split("|") if c.strip()]
                sep_line = table_lines[1]
                is_sep = bool(re.match(r"^\|[\s\-:|]+\|$", sep_line))
                data_rows = table_lines[2:] if is_sep else table_lines[1:]

                for row in data_rows:
                    matches = IN_TEXT_CITATION_RE.findall(row)
                    if not matches:
                        continue
                    row_keys: List[str] = []
                    for raw_m in matches:
                        row_keys.extend(expand_citation_keys(raw_m))
                    row_keys = list(dict.fromkeys(row_keys))
                    if not row_keys:
                        continue

                    cells = [c.strip() for c in row.split("|") if c.strip()]
                    row_str = " | ".join(cells) if cells else row

                    results.append({
                        "content_type": "table",
                        "table_caption": caption,
                        "table_headers": headers,
                        "paragraph": row_str,
                        "citation_keys": row_keys,
                        "source_section": section_title,
                        "source_file": section_path.name,
                    })

    return results


def _find_section_for_page(page_no: int, outline: List[Dict]) -> Tuple[str, str]:
    """根据页码从大纲 outline 中匹配对应的章节标题与文件名。

    匹配规则：
    1. 跳过 References / other 等非正文章节；
    2. 页码落在章节 [start_page, end_page] 区间内即为候选；
    3. 同一页存在多个候选章节时，取 (start_page, 大纲序) 最大的章节，
       即最晚开始、最贴近该页内容的章节；
    4. 无任何命中时回退 Main Content / main.md。
    """
    best = None
    best_key = None
    for pos, item in enumerate(outline):
        fname = item.get("file", "").lower()
        if fname in ("references.md", "other.md"):
            continue
        sp = item.get("start_page", 1)
        ep = item.get("end_page", 999)
        if sp <= page_no <= ep:
            key = (sp, pos)
            if best_key is None or key > best_key:
                best, best_key = item, key
    if best is not None:
        return best.get("title", "Main Content"), best.get("file", "main.md")
    return "Main Content", "main.md"


def _norm_alnum(text: str) -> str:
    """合并跨行连字符并剥离 Markdown 符号/标点，仅保留小写字母数字序列用于跨格式子串对齐"""
    dehyph = re.sub(r"([a-zA-Z])-\s+([a-z])", r"\1\2", text)
    dehyph = re.sub(r"(\d)\^(\d)", r"\1\2", dehyph)
    return re.sub(r"[^a-z0-9]+", " ", dehyph.lower()).strip()


def _find_section_for_claim(
    claim_paragraph: str,
    page_no: int,
    folder: Path,
    outline: List[Dict]
) -> Tuple[str, str]:
    """
    为待核查论断匹配最准确的章节标题与文件名：
    1. 优先通过去标点/去 Markdown 格式后的前 8 个词在各章节 Markdown 中做精确子串匹配（解决同一页多子节、粗体及表格管道符导致的章节错位）
    2. 兜底通过页码匹配
    """
    words = _norm_alnum(claim_paragraph).split()
    probes = [p for p in (" ".join(words[:8]), " ".join(words[-8:])) if p]
    if probes:
        for item in outline:
            fname = item.get("file", "")
            if fname.lower() in ("references.md", "other.md"):
                continue
            sec_file = (folder / "sections" / fname) if (folder / "sections" / fname).exists() else (folder / fname)
            if sec_file.exists():
                try:
                    norm_sec = _norm_alnum(sec_file.read_text(encoding="utf-8"))
                    if any(p in norm_sec for p in probes):
                        return item.get("title", "Main Content"), fname or "main.md"
                except Exception:
                    pass

    return _find_section_for_page(page_no, outline)


def _enrich_table_claim(claim_paragraph: str, folder: Path, sec_file: str) -> str:
    """若引用段落来自 Markdown 表格行，补充 Table 标题/列头并还原带 | 分隔符的完整单元格行；若跨页散文段首被截断，从所属章节还原完整段落。"""
    if not sec_file or "Table " in claim_paragraph[:30]:
        return claim_paragraph
    sec_path = (folder / "sections" / sec_file) if (folder / "sections" / sec_file).exists() else (folder / sec_file)
    if not sec_path.is_file():
        return claim_paragraph
    try:
        sec_text = sec_path.read_text(encoding="utf-8")
    except Exception:
        return claim_paragraph

    norm_claim = _norm_alnum(claim_paragraph)
    words = norm_claim.split()
    probes = [p for p in (" ".join(words[:6]), " ".join(words[-6:])) if len(p) >= 6]
    if not probes:
        return claim_paragraph

    lines = sec_text.splitlines()
    for idx, line in enumerate(lines):
        s = line.strip()
        if not (s.startswith("|") and s.endswith("|")):
            continue
        norm_line = _norm_alnum(s)
        if not any(p in norm_line for p in probes):
            continue
        # Walk backwards to find table header and Table N caption
        t_start = idx
        while t_start > 0 and lines[t_start - 1].strip().startswith("|"):
            t_start -= 1
        t_end = idx
        while t_end + 1 < len(lines) and lines[t_end + 1].strip().startswith("|"):
            t_end += 1
        header_line = lines[t_start].strip()
        caption_line = ""
        for c_idx in range(max(0, t_start - 5), t_start):
            cand = lines[c_idx].strip()
            if re.match(r"^(?:\*\*)?Table\s+\d+", cand, re.I):
                caption_line = " ".join(
                    lines[k].strip() for k in range(c_idx, t_start) if lines[k].strip()
                )
                break
        # Collect all data rows from this table whose normalized content appears in norm_claim
        matched_rows: List[str] = []
        for r_idx in range(t_start + 2, t_end + 1):
            r_str = lines[r_idx].strip()
            r_norm = _norm_alnum(r_str)
            if r_norm and r_norm in norm_claim:
                matched_rows.append(r_str)
        if not matched_rows and idx >= t_start + 2:
            matched_rows.append(s)
        row_body = " ".join(matched_rows) if matched_rows else claim_paragraph
        prefix_parts = []
        if caption_line:
            prefix_parts.append(caption_line)
        if header_line:
            prefix_parts.append(f"Columns: {header_line}")
        if prefix_parts:
            return f"[{' | '.join(prefix_parts)}] {row_body}"
        break

    # Cross-page truncated prose claim recovery (e.g. starting with lowercase "length n is smaller...")
    if claim_paragraph and claim_paragraph[0].islower():
        body = strip_md_header(sec_text)
        for para in re.split(r"\n\s*\n", body):
            para = para.strip()
            if not para or para.startswith("|"):
                continue
            if probes[0] in _norm_alnum(para) and len(para) > len(claim_paragraph):
                clean_p = re.sub(r"</?(?:sup|sub|u)>|_", "", para)
                clean_p = re.sub(r"\s+", " ", clean_p).strip()
                return clean_p

    return claim_paragraph


def _build_claim_record(
    counter: int,
    paragraph: str,
    canonical_keys: List[str],
    raw_keys: List[str],
    citation_map: Dict[str, str],
    matched_refs: Dict[str, str],
    source_section: str,
    source_file: str,
    content_type: str = "prose",
    table_caption: Optional[str] = None,
    table_headers: Optional[List[str]] = None,
) -> Dict[str, Any]:
    record: Dict[str, Any] = {
        "claim_id": f"claim_{counter:03d}",
        "context_id": f"ctx_{counter:03d}",
        "content_type": content_type,
        "paragraph": paragraph,
        "citation_keys": list(dict.fromkeys(canonical_keys)),
        "raw_citation_keys": raw_keys,
        "citation_key_map": citation_map,
        "matched_references": matched_refs,
        "source_section": source_section,
        "source_file": source_file,
    }
    if content_type == "table":
        if table_caption:
            record["table_caption"] = table_caption
        if table_headers is not None:
            record["table_headers"] = table_headers
    return record


def _format_key_labels(keys: List[str], c_map: Dict[str, str]) -> List[str]:
    return [
        f"`{k}` ({c_map[k]})" if c_map.get(k) and c_map[k] != k else f"`{k}`"
        for k in keys
    ]


def _extract_first_author_surname(item: Dict[str, Any]) -> Optional[str]:
    """从参考文献条目中提取第一作者姓氏"""
    authors = item.get("authors") or []
    if authors:
        first = authors[0].strip()
        if "," in first:
            sur = first.split(",")[0].strip()
            if sur:
                return re.sub(r"[^\w-]", "", sur)
        words = [w for w in first.split() if w]
        if words:
            last_w = words[-1]
            if (len(last_w) <= 2 or last_w.endswith(".")) and len(words) > 1:
                return re.sub(r"[^\w-]", "", words[0])
            return re.sub(r"[^\w-]", "", last_w)

    ay_key = item.get("author_year_key") or ""
    m = re.match(r"^\[?([A-Za-z\-]+)", ay_key)
    if m:
        return m.group(1)

    c_id = item.get("canonical_id") or ""
    m = re.match(r"^\[?([A-Za-z\-]+)", c_id)
    if m and not m.group(1).isdigit():
        return m.group(1)

    raw = item.get("raw_text") or ""
    m = re.match(r"^\s*(?:\[\d+\]|\d+\.)?\s*([A-Za-z\-]+)", raw)
    if m:
        return m.group(1)

    return None


class AdvancedCatalogLookup(dict):
    """
    增强版参考文献字典查找器：
    1. 保持完全向后兼容的 dict 特性：lookup[k]、lookup.get(k)、items() 均可正常调用
    2. 维护按 DOI、arXiv ID、作者-年份多维度倒排索引
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.by_doi: Dict[str, Dict[str, Any]] = {}
        self.by_arxiv: Dict[str, Dict[str, Any]] = {}
        self.by_author_year: Dict[Tuple[str, Optional[int], Optional[str]], List[Dict[str, Any]]] = {}


def _build_advanced_catalog_lookup(ref_catalog: List[Dict[str, Any]]) -> AdvancedCatalogLookup:
    """构建多维倒排参考文献索引表，支持 DOI、arXiv、作者-年份精准匹配与消歧"""
    lookup = AdvancedCatalogLookup()
    author_year_counts: Dict[Tuple[str, int], int] = {}
    for item in ref_catalog:
        fa = _extract_first_author_surname(item)
        yr = item.get("year")
        if fa and yr:
            key_pair = (fa.lower(), yr)
            author_year_counts[key_pair] = author_year_counts.get(key_pair, 0) + 1

    for item in ref_catalog:
        canon_id = item.get("canonical_id")
        if canon_id:
            lookup[canon_id] = item
            lookup[canon_id.strip("[]")] = item
        if "index" in item:
            lookup[f"[ref_{item['index']}]"] = item
            lookup[f"ref_{item['index']}"] = item

        # 1. DOI 倒排索引
        doi = item.get("doi")
        if doi:
            norm_doi = doi.lower().strip().rstrip(".,;)]")
            lookup.by_doi[norm_doi] = item
            lookup[norm_doi] = item
            lookup[f"[{norm_doi}]"] = item
            lookup[f"doi:{norm_doi}"] = item
            lookup[f"DOI:{norm_doi}"] = item
            lookup[f"DOI: {doi}"] = item
            lookup[doi] = item

        # 2. arXiv 倒排索引
        arxiv_id = item.get("arxiv_id")
        if arxiv_id:
            norm_arx = re.sub(r"v\d+$", "", arxiv_id.lower().strip().rstrip(".,;)]"))
            lookup.by_arxiv[norm_arx] = item
            lookup.by_arxiv[arxiv_id.lower().strip()] = item
            lookup[arxiv_id] = item
            lookup[norm_arx] = item
            lookup[f"arXiv:{arxiv_id}"] = item
            lookup[f"arxiv:{norm_arx}"] = item
            lookup[f"[{arxiv_id}]"] = item
            lookup[f"[arXiv:{arxiv_id}]"] = item

        # 3. 作者-年份倒排索引
        fa = _extract_first_author_surname(item)
        yr = item.get("year")
        ys = item.get("year_suffix")
        if fa and yr:
            lookup.by_author_year.setdefault((fa.lower(), yr, ys), []).append(item)
            lookup.by_author_year.setdefault((fa.lower(), yr, None), []).append(item)

        ay_key = item.get("author_year_key")
        if ay_key:
            lookup[ay_key] = item
            lookup[ay_key.strip("[]")] = item
            m_author = re.match(r"^\[?([A-Za-z]+)", ay_key)
            year_match = YEAR_PATTERN.search(ay_key)
            if m_author and year_match:
                fa_m = m_author.group(1)
                yr_m = int(year_match.group(1))
                ys_m = year_match.group(2)
                if ys_m:
                    lookup[f"[{fa_m}, {yr_m}{ys_m}]"] = item
                    lookup[f"[{fa_m} et al., {yr_m}{ys_m}]"] = item
                    lookup[f"[{fa_m} et al. {yr_m}{ys_m}]"] = item
                    lookup[f"({fa_m} et al., {yr_m}{ys_m})"] = item
                    lookup[f"({fa_m}, {yr_m}{ys_m})"] = item
                elif author_year_counts.get((fa_m.lower(), yr_m), 0) == 1:
                    lookup[f"[{fa_m}, {yr_m}]"] = item
                    lookup[f"[{fa_m} et al., {yr_m}]"] = item

    return lookup


_build_catalog_lookup = _build_advanced_catalog_lookup


def _align_citation_keys(
    raw_keys: List[str],
    ref_catalog: List[Dict[str, Any]],
    catalog_lookup: Dict[str, Dict[str, Any]],
    raw_matched: Optional[Dict[str, str]] = None,
    default_missing: str = "",
) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
    """
    分层对齐正文引用标记到 References.md 条目。
    按 5 级优先级严格判定，杜绝同作者跨年份或不同合作者的误归并与贪婪匹配：
    1. 优先级 1: PDF 物理链接目标 (raw_matched) 最高优先匹配
    2. 优先级 2: DOI / arXiv ID 强特征精准对齐
    3. 优先级 3: 标准引用键与带有 a/b 后缀的键 (Smith et al., 2020a vs 2020b) 严格隔离不混淆
    4. 优先级 4: 同姓氏不同合作者 (Smith & Doe vs Smith & Brown) 能够精准消歧，杜绝贪婪误匹配
    5. 优先级 5: 标点变体自适应与子串模糊兜底
    """
    canonical_keys: List[str] = []
    citation_map: Dict[str, str] = {}
    matched_refs: Dict[str, str] = {}

    for k in raw_keys:
        cat_item: Optional[Dict[str, Any]] = None

        # -------------------------------------------------------------
        # 优先级 1: PDF 物理链接目标 (raw_matched) 匹配
        # -------------------------------------------------------------
        if raw_matched is not None and k in raw_matched and raw_matched[k]:
            phys_target = raw_matched[k].strip()
            # 1.1 检查目标文本中的 DOI / arXiv ID
            m_d = DOI_PATTERN.search(phys_target)
            if m_d:
                d_norm = m_d.group(1).lower().rstrip(".,;)]")
                if hasattr(catalog_lookup, "by_doi") and d_norm in catalog_lookup.by_doi:
                    cat_item = catalog_lookup.by_doi[d_norm]
            if not cat_item:
                m_a = ARXIV_PATTERN.search(phys_target)
                if m_a:
                    a_norm = re.sub(r"v\d+$", "", m_a.group(1).lower().rstrip(".,;)]"))
                    if hasattr(catalog_lookup, "by_arxiv") and a_norm in catalog_lookup.by_arxiv:
                        cat_item = catalog_lookup.by_arxiv[a_norm]

            # 1.2 检查目标文本与 catalog 条目文本的重合度
            if not cat_item:
                norm_target = _norm_alnum(phys_target)
                best_match = None
                best_overlap = 0
                for it in ref_catalog:
                    norm_it = _norm_alnum(it.get("raw_text", ""))
                    if norm_target in norm_it or norm_it in norm_target:
                        overlap = len(min(norm_target, norm_it, key=len))
                        if overlap > best_overlap:
                            best_overlap = overlap
                            best_match = it
                if best_match and best_overlap >= 15:
                    cat_item = best_match

        # -------------------------------------------------------------
        # 优先级 2: DOI / arXiv ID 强特征匹配
        # -------------------------------------------------------------
        if not cat_item:
            # 2.1 DOI
            m_doi = DOI_PATTERN.search(k) or re.search(r"\b10\.\d{4,9}/[^\s\]\)\,\;]+", k)
            if m_doi:
                doi_cand = (m_doi.group(1) if m_doi.lastindex else m_doi.group(0)).lower().rstrip(".,;)]")
                if hasattr(catalog_lookup, "by_doi") and doi_cand in catalog_lookup.by_doi:
                    cat_item = catalog_lookup.by_doi[doi_cand]
                else:
                    for it in ref_catalog:
                        it_doi = (it.get("doi") or "").lower()
                        if it_doi and it_doi == doi_cand:
                            cat_item = it
                            break
                        if doi_cand in it.get("raw_text", "").lower():
                            cat_item = it
                            break

        if not cat_item:
            # 2.2 arXiv
            m_arx = ARXIV_PATTERN.search(k) or re.search(r"\b\d{4}\.\d{4,5}(?:v\d+)?\b", k)
            if m_arx:
                arx_cand = (m_arx.group(1) if m_arx.lastindex else m_arx.group(0)).lower().rstrip(".,;)]")
                arx_norm = re.sub(r"v\d+$", "", arx_cand)
                if hasattr(catalog_lookup, "by_arxiv") and (arx_norm in catalog_lookup.by_arxiv or arx_cand in catalog_lookup.by_arxiv):
                    cat_item = catalog_lookup.by_arxiv.get(arx_norm) or catalog_lookup.by_arxiv.get(arx_cand)
                else:
                    for it in ref_catalog:
                        it_arx = (it.get("arxiv_id") or "").lower()
                        if it_arx:
                            it_norm = re.sub(r"v\d+$", "", it_arx)
                            if it_norm == arx_norm or it_arx == arx_cand:
                                cat_item = it
                                break
                        if arx_norm in it.get("raw_text", "").lower():
                            cat_item = it
                            break

        # -------------------------------------------------------------
        # 优先级 3: 严格引用键 (Exact Key Match)
        # -------------------------------------------------------------
        if not cat_item:
            cat_item = catalog_lookup.get(k)
            if not cat_item:
                if not k.startswith("["):
                    cat_item = catalog_lookup.get(f"[{k}]")
                else:
                    cat_item = catalog_lookup.get(k.strip("[]"))

        # -------------------------------------------------------------
        # 优先级 4: 作者-年份消歧与标点变体归一化匹配
        # -------------------------------------------------------------
        if not cat_item:
            m_year = YEAR_PATTERN.search(k)
            if m_year:
                q_year = int(m_year.group(1))
                q_suffix = m_year.group(2) or ""
                q_author_part = k[:m_year.start()].strip(" [](),;:")
                q_has_et_al = bool(re.search(r"\bet\s+al\.?\b", q_author_part, re.IGNORECASE))
                q_author_clean = re.sub(r"\bet\s+al\.?\b", "", q_author_part, flags=re.IGNORECASE).strip(" ,;:")

                coauthor_tokens = [
                    re.sub(r"[^\w-]", "", c.strip().split()[-1].lower())
                    for c in re.split(r"\s+(?:and|&)\s+|,\s*", q_author_clean)
                    if c.strip()
                ]
                q_first_author = coauthor_tokens[0] if coauthor_tokens else ""
                q_other_coauthors = set(coauthor_tokens[1:])

                if q_first_author:
                    scored_candidates = []
                    for it in ref_catalog:
                        # 4.1 年份校验
                        it_year = it.get("year")
                        if not it_year:
                            m_y = YEAR_PATTERN.search(it.get("raw_text", ""))
                            it_year = int(m_y.group(1)) if m_y else None
                        if it_year and it_year != q_year:
                            continue

                        # 4.2 第一作者校验
                        it_fa = _extract_first_author_surname(it)
                        if not it_fa or it_fa.lower() != q_first_author.lower():
                            continue

                        # 4.3 计算匹配得分
                        score = 20.0

                        it_suffix = it.get("year_suffix") or ""
                        if not it_suffix:
                            m_ys = YEAR_PATTERN.search(
                                it.get("canonical_id", "") or it.get("author_year_key", "") or it.get("raw_text", "")
                            )
                            it_suffix = m_ys.group(2) if m_ys else ""

                        if q_suffix:
                            if it_suffix == q_suffix:
                                score += 60.0
                            else:
                                score -= 100.0  # 后缀冲突重罚
                        else:
                            if not it_suffix:
                                score += 15.0
                            else:
                                score -= 10.0

                        it_raw_lower = it.get("raw_text", "").lower()
                        it_key_lower = (it.get("author_year_key") or "").lower()
                        if q_other_coauthors:
                            for co in q_other_coauthors:
                                if co and (co in it_raw_lower or co in it_key_lower):
                                    score += 50.0
                                else:
                                    score -= 60.0  # 指定合作者未命中重罚
                        else:
                            if q_has_et_al:
                                if "et al" in it_key_lower or len(it.get("authors", [])) > 2:
                                    score += 10.0

                        if score > 0:
                            scored_candidates.append((score, it))

                    if scored_candidates:
                        scored_candidates.sort(key=lambda x: x[0], reverse=True)
                        cat_item = scored_candidates[0][1]

        # -------------------------------------------------------------
        # 优先级 5: 模糊子串与字符归一化兜底
        # -------------------------------------------------------------
        if not cat_item:
            clean_k = _norm_alnum(k)
            if len(clean_k) >= 4:
                for rk, ci in catalog_lookup.items():
                    clean_rk = _norm_alnum(rk)
                    if clean_rk and (clean_k == clean_rk or clean_k in clean_rk or clean_rk in clean_k):
                        cat_item = ci
                        break

        # -------------------------------------------------------------
        # 结果装配
        # -------------------------------------------------------------
        if cat_item:
            c_id = cat_item["canonical_id"]
            canonical_keys.append(c_id)
            citation_map[c_id] = k
            matched_refs[c_id] = cat_item["raw_text"]
            matched_refs[k] = cat_item["raw_text"]
        else:
            canonical_keys.append(k)
            matched_refs[k] = (raw_matched or {}).get(k, default_missing)

    return canonical_keys, citation_map, matched_refs


def extract_from_paper_folder(
    paper_dir: str,
    paper_id: Optional[str] = None
) -> Dict:
    """
    主入口：从论文专属文献夹中提取所有引用段落，
    并通过序列号或超链接精准对齐参考文献，最终落盘为 .json + .md 两份文件。

    执行策略：
    1. 优先检测文献夹内是否存在 raw.pdf。若存在且带有跳转链接，则启用高精度 PDFLinkCitationExtractor；
    2. 若不存在或链接解析无结果，自动优雅降级为 Markdown 章节遍历与正则对齐。
    """
    folder = Path(paper_dir)
    meta_path = folder / "metadata.json"

    # 1. 读取 metadata.json
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {folder}")

    with meta_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    paper_title = metadata.get("paper_title", "Unknown")
    paper_id = paper_id or metadata.get("paper_id", folder.name)
    outline = metadata.get("outline", [])
    all_claims = []

    ref_path = (folder / "sections" / "References.md") if (folder / "sections" / "References.md").exists() else (folder / "References.md")
    ref_catalog = parse_reference_catalog_from_md(ref_path) if ref_path.exists() else []

    # 落地结构化 references.json
    if ref_catalog:
        ref_json_path = folder / "references.json"
        with ref_json_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(ref_catalog, f, ensure_ascii=False, indent=2)

    # 建立目录快速索引：支持通过 [1]、[Krizhevsky et al., 2012]、[Krizhevsky, 2012] 等多键反查
    catalog_lookup = _build_catalog_lookup(ref_catalog)
    ref_entries = {k: item["raw_text"] for k, item in catalog_lookup.items()}

    def _align_keys(
        raw_keys: List[str],
        raw_matched: Optional[Dict[str, str]] = None,
        default_missing: str = "",
    ) -> Tuple[List[str], Dict[str, str], Dict[str, str]]:
        return _align_citation_keys(
            raw_keys=raw_keys,
            ref_catalog=ref_catalog,
            catalog_lookup=catalog_lookup,
            raw_matched=raw_matched,
            default_missing=default_missing,
        )

    # 2. 检查是否可使用 Tier 1 (PDF 矢量超链接层提取)
    raw_pdf_path = folder / "raw.pdf"
    used_link_extraction = False

    if raw_pdf_path.exists():
        try:
            link_claims, _ = extract_citations_from_pdf_links(raw_pdf_path)
            if link_claims:
                logger.info(f"[{paper_id}] 成功启用 Tier 1 矢量链接提取，获得 {len(link_claims)} 个段落")
                for claim_counter, lc in enumerate(link_claims, 1):
                    src_page = lc.get("source_page", 1)
                    sec_title, sec_file = _find_section_for_claim(
                        lc["paragraph"], src_page, folder, outline
                    )
                    enriched_para = _enrich_table_claim(lc["paragraph"], folder, sec_file)
                    raw_keys = lc["citation_keys"]
                    canonical_keys, citation_map, matched_refs = _align_keys(
                        raw_keys, raw_matched=lc["matched_references"]
                    )
                    all_claims.append(_build_claim_record(
                        claim_counter, enriched_para, canonical_keys,
                        raw_keys, citation_map, matched_refs, sec_title, sec_file,
                    ))
                used_link_extraction = True
        except Exception as e:
            logger.warning(f"[{paper_id}] PDF 链接层提取异常，自动回退到 Markdown 模式: {e}")

    # 3. 若未启用链接提取，执行 Tier 2 (Markdown 章节与 References.md 对齐)
    if not used_link_extraction:
        logger.info(f"[{paper_id}] Tier 2 从 References.md 解析出 {len(ref_entries)} 条参考文献")

        claim_counter = 1
        for item in outline:
            filename = item.get("file", "")
            title = item.get("title", "")

            if filename.lower() in ("references.md", "other.md"):
                continue

            section_path = (folder / "sections" / filename) if (folder / "sections" / filename).exists() else (folder / filename)
            for cp in extract_cited_paragraphs(section_path, title):
                raw_keys = cp["citation_keys"]
                canonical_keys, citation_map, matched_refs = _align_keys(
                    raw_keys, default_missing="(该参考文献在 References.md 中未找到)"
                )
                all_claims.append(_build_claim_record(
                    claim_counter, cp["paragraph"], canonical_keys,
                    raw_keys, citation_map, matched_refs, cp["source_section"], cp["source_file"],
                    content_type=cp.get("content_type", "prose"),
                    table_caption=cp.get("table_caption"),
                    table_headers=cp.get("table_headers"),
                ))
                claim_counter += 1

    # 4. 统计指标
    unique_ref_keys = {k for c in all_claims for k in c["citation_keys"]}

    total_refs_count = len(ref_catalog) if ref_catalog else len(ref_entries)

    raw_pdf = folder / "raw.pdf"
    result = {
        "paper_id": paper_id,
        "paper_title": paper_title,
        "pdf_sha256": pdf_sha256(raw_pdf) if raw_pdf.exists() else None,
        "parser_version": metadata.get("parser_version"),
        "extraction_method": "pdf_vector_links" if used_link_extraction else "markdown_syntax",
        "total_claims": len(all_claims),
        "total_unique_refs_cited": len(unique_ref_keys),
        "total_refs_in_paper": total_refs_count,
        "extracted_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "claims": all_claims
    }

    # 5. 写入 extracted_claims.json
    json_path = folder / "extracted_claims.json"
    with json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    # 6. 写入 extracted_claims.md
    md_path = folder / "extracted_claims.md"
    md_lines = _render_claims_markdown(result, ref_catalog)
    with md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(md_lines))

    # 7. 写入 citations/ 独立文件夹（各引用条目按 1 开始顺序独立成文件）
    _write_citations_directory(folder, result, ref_catalog, ref_entries)

    logger.info(f"[{paper_id}] 提取完成: {len(all_claims)} 处引用段落, 涉及 {len(unique_ref_keys)} 篇被引文献 (方法: {result['extraction_method']})")
    return result


def _render_claims_markdown(result: Dict, ref_catalog: List[Dict]) -> List[str]:
    """渲染人类可读的 Markdown 引用清单"""
    total_refs_count = result["total_refs_in_paper"]
    lines = [
        f"# 论文引用段落与参考文献对齐清单 (Extracted Claims & Citations)",
        f"",
        f"> **所属论文**: {result['paper_title']} ({result['paper_id']})",
        f"> **抽取引擎**: {result.get('extraction_method', 'standard')}",
        f"> **共提取引用段落**: {result['total_claims']} 处",
        f"> **关联被引文献**: {result['total_unique_refs_cited']} / {total_refs_count} 篇",
        f"> **提取时间**: {result['extracted_at']}",
        f"",
        f"---",
        f""
    ]

    for claim in result["claims"]:
        lines.extend([
            f"### [{claim['claim_id'].upper()}] 来自章节：`{claim['source_file']}`（{claim['source_section']}）",
            f"",
            f"**正文引用段落**:",
            f"",
        ])

        para_lines = claim["paragraph"].split("\n")
        for pl in para_lines:
            lines.append(f"> {pl}")
        lines.append(f"")

        c_map = claim.get("citation_key_map", {})
        key_labels = _format_key_labels(claim["citation_keys"], c_map)

        lines.append(f"**对齐序列号 / 标识**: {', '.join(key_labels)}")
        lines.append(f"")

        lines.append(f"**关联参考文献**:")
        lines.append(f"")
        for k in claim["citation_keys"]:
            ref_text = claim["matched_references"].get(k, "")
            raw_k = c_map.get(k)
            prefix = f"* **{k}** `{raw_k}`" if (raw_k and raw_k != k) else f"* **{k}**"
            lines.append(f"{prefix}: {ref_text}")
        lines.append(f"")
        lines.append(f"---")
        lines.append(f"")

    cited_keys = {k for c in result["claims"] for k in c["citation_keys"]}

    if ref_catalog:
        uncited = [it for it in ref_catalog if it["canonical_id"] not in cited_keys]
        if uncited:
            lines.extend([
                f"## 附录：正文中未检测到引用的参考文献 ({len(uncited)} 篇)",
                f""
            ])
            for it in uncited[:30]:
                ay_info = f" `{it['author_year_key']}`" if it.get("author_year_key") else ""
                lines.append(f"* **{it['canonical_id']}**{ay_info}: {it['raw_text']}")
            lines.append(f"")

    lines.append(f"*由 CiteTrace 引用证据链核查系统自动提取生成。*")
    return lines


def _clean_citations_directory(citations_dir: Path) -> None:
    """清理 citations 目录下的旧产物 (*.md, *.json)，杜绝重新生成时的历史残留文件"""
    if not citations_dir.exists():
        return
    for p in citations_dir.glob("*.md"):
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass
    for p in citations_dir.glob("*.json"):
        try:
            if p.is_file():
                p.unlink()
        except OSError:
            pass


def _write_citations_directory(
    folder: Path,
    result: Dict[str, Any],
    ref_catalog: List[Dict[str, Any]],
    ref_entries: Dict[str, str]
) -> Path:
    """
    在论文处理目录下创建 citations 文件夹：
    文件夹存放该论文的引用部分以及对应参考文献的东西，并且引用的文件用 1 开始排序 (1.md, 2.md, ...)
    生成前自动清理旧文件，杜绝历史残留文件。
    """
    citations_dir = folder / "citations"
    citations_dir.mkdir(parents=True, exist_ok=True)
    _clean_citations_directory(citations_dir)

    catalog_by_canon = _build_catalog_lookup(ref_catalog)

    all_claims = result.get("claims", [])
    paper_title = result.get("paper_title", folder.name)

    for idx, claim in enumerate(all_claims, 1):
        c_map = claim.get("citation_key_map", {})
        matched_refs = claim.get("matched_references", {})
        keys = claim.get("citation_keys", [])

        key_labels = _format_key_labels(keys, c_map)

        md_lines = [
            f"# 引用 #{idx} (Citation {idx})",
            "",
            f"> **所属论文**: {paper_title}",
            f"> **出现章节**: `{claim.get('source_file', '')}`（{claim.get('source_section', '')}）",
            f"> **对应序列号 / 标识**: {', '.join(key_labels)}",
        ]
        if claim.get("content_type") == "table":
            if claim.get("table_caption"):
                md_lines.append(f"> **关联表题**: {claim['table_caption']}")
            if claim.get("table_headers"):
                md_lines.append(f"> **表头**: {' | '.join(claim['table_headers'])}")
        md_lines.extend([
            "",
            "---",
            "",
            "### 一、正文引用段落原文",
            "",
        ])

        para_lines = claim.get("paragraph", "").split("\n")
        for pl in para_lines:
            md_lines.append(f"> {pl}")
        md_lines.extend(["", "---", "", f"### 二、本处引用关联的参考文献 (共 {len(keys)} 篇)", ""])

        json_refs: Dict[str, Any] = {}

        for k in keys:
            raw_k = c_map.get(k, k)
            cat_item = catalog_by_canon.get(k) or catalog_by_canon.get(raw_k)
            raw_text = matched_refs.get(k) or (cat_item["raw_text"] if cat_item else ref_entries.get(k, ""))

            clean_digits = re.search(r'\d+', k)
            canonical_id = f"[{clean_digits.group(0)}]" if clean_digits else k

            ref_title = cat_item.get("title", "") if cat_item else ""
            if not ref_title and raw_text:
                parsed = parse_reference_entry(raw_text)
                ref_title = (parsed.title or "") if parsed else ""

            header_label = f"{canonical_id}"
            if raw_k and raw_k != canonical_id:
                header_label += f" `{raw_k}`"
            if ref_title:
                header_label += f": {ref_title}"

            md_lines.append(f"#### {header_label}")
            md_lines.append(f"* **规范序号**: `{canonical_id}`")
            if raw_k and raw_k != canonical_id:
                md_lines.append(f"* **原文标记**: `{raw_k}`")
            if ref_title:
                md_lines.append(f"* **论文标题**: {ref_title}")
            md_lines.append(f"* **完整参考文献条目**: {raw_text}")
            md_lines.append(f"* **本地文献处理目录**: [`../cited_papers/{canonical_id}/`](../cited_papers/{canonical_id}/)")
            md_lines.append(f"* **本地文献全文 PDF**: [`../cited_papers/{canonical_id}/raw.pdf`](../cited_papers/{canonical_id}/raw.pdf)")
            md_lines.append("")

            json_refs[canonical_id] = {
                "key": canonical_id,
                "raw_key": raw_k,
                "title": ref_title,
                "raw_text": raw_text,
                "cited_paper_folder": f"cited_papers/{canonical_id}",
                "cited_paper_pdf": f"cited_papers/{canonical_id}/raw.pdf"
            }

        md_lines.extend([
            "---",
            "",
            "### 三、核查状态 (Verification Status)",
            "",
            "* **状态**: 待核查 (Pending Verification)",
            ""
        ])

        # 写入 1.md, 2.md ...
        item_md_path = citations_dir / f"{idx}.md"
        with item_md_path.open("w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(md_lines))

        # 写入 1.json, 2.json ...
        item_json_path = citations_dir / f"{idx}.json"
        claim_json = {
            "index": idx,
            "file": f"{idx}.md",
            "claim_id": claim.get("claim_id", f"claim_{idx:03d}"),
            "source_file": claim.get("source_file", ""),
            "source_section": claim.get("source_section", ""),
            "paragraph": claim.get("paragraph", ""),
            "citation_keys": keys,
            "citation_key_map": c_map,
            "matched_references": json_refs,
            "verification": {
                "status": "PENDING"
            }
        }
        if "content_type" in claim:
            claim_json["content_type"] = claim["content_type"]
        if "table_caption" in claim:
            claim_json["table_caption"] = claim["table_caption"]
        if "table_headers" in claim:
            claim_json["table_headers"] = claim["table_headers"]

        with item_json_path.open("w", encoding="utf-8", newline="\n") as f:
            json.dump(claim_json, f, ensure_ascii=False, indent=2)

    # 写入 citations/index.md
    index_md_path = citations_dir / "index.md"
    index_lines = [
        f"# 论文正文引用清单与对应参考文献索引 (Citations Index)",
        "",
        f"> **所属论文**: {paper_title}",
        f"> **引用总数**: {len(all_claims)} 处引用",
        f"> **排序方式**: 从 1 开始自然顺序编号 (`1.md` ~ `{len(all_claims)}.md`)",
        "",
        "---",
        "",
        "| 序号 | 来源章节 | 关联参考文献 | 独立详情文件 |",
        "| :--- | :--- | :--- | :--- |"
    ]
    for idx, c in enumerate(all_claims, 1):
        keys_str = ", ".join(c.get("citation_keys", []))
        index_lines.append(f"| **{idx}** | `{c.get('source_file', '')}` | {keys_str} | [{idx}.md]({idx}.md) |")

    index_lines.extend(["", "---", "", "*由 CiteTrace 系统自动生成*"])
    with index_md_path.open("w", encoding="utf-8", newline="\n") as f:
        f.write("\n".join(index_lines))

    # 写入 citations/index.json
    index_json_path = citations_dir / "index.json"
    with index_json_path.open("w", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return citations_dir
