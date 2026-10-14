import re
from typing import Dict, List, Optional
from models.citation import CitationRef, CitationStatus

# 匹配年份，如 "(2017)", "2017." 或 "2017a"
YEAR_PATTERN = re.compile(r"\b(19\d{2}|20\d{2})([a-z])?\b")
NUMBERED_ENTRY_RE = re.compile(r"^\s*(?:[-*•]\s*)?(?:\[\d+\]|\d+\.)")
# 匹配 DOI
DOI_PATTERN = re.compile(r"(?:DOI:\s*|https?://(?:dx\.)?doi\.org/)?(10\.\d{4,9}/[-._;()/:A-Za-z0-9]+)", re.IGNORECASE)
# 匹配 arXiv 编号
ARXIV_PATTERN = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5}(?:v\d+)?)", re.IGNORECASE)
# 匹配期刊/会议出处前缀（用于过滤非标题分段）
VENUE_KEYWORDS = re.compile(
    r"^(?:in\s+|proceedings\s+|advances\s+in|ieee|acm|arxiv|springer|elsevier|nature|science|journal\s+of|trans\b|transactions\b|\d{4})",
    re.IGNORECASE,
)


def parse_reference_entry(raw_entry: str, default_key: Optional[str] = None) -> Optional[CitationRef]:
    """
    解析单条参考文献文本，提取结构化元数据。
    全面支持：
    1. 数字编号型: "[1] Vaswani et al. Attention..."
    2. 作者-年份型 (无标号): "Alex Krizhevsky, Ilya Sutskever... Imagenet..."
    """
    entry_clean = raw_entry.strip()
    if not entry_clean:
        return None

    # 1. 提取引用锚点 key，如 [1] 或 1.
    key = None
    body_text = entry_clean

    bracket_match = re.match(r"^\s*(?:[-*•]\s*)?\[([^\]]+)\]\s*(.*)", entry_clean, re.DOTALL)
    if bracket_match:
        inner = bracket_match.group(1).strip()
        num_clean = inner.lstrip("0") if inner.isdigit() else inner
        key = f"[{num_clean or '0'}]"
        body_text = bracket_match.group(2).strip()
    else:
        num_match = re.match(r"^\s*(?:[-*•]\s*)?(\d+)\.\s*(.*)", entry_clean, re.DOTALL)
        if num_match:
            key = f"[{int(num_match.group(1))}]"
            body_text = num_match.group(2).strip()

    # 2. 提取发表年份及后缀 (如 2020a)
    year = None
    year_suffix = None
    year_match = YEAR_PATTERN.search(body_text)
    if year_match:
        try:
            year = int(year_match.group(1))
            year_suffix = year_match.group(2) or None
        except ValueError:
            pass

    # 3. 提取 DOI
    doi = None
    doi_match = DOI_PATTERN.search(body_text)
    if doi_match:
        doi = doi_match.group(1).strip().rstrip(".,;")

    # 4. 提取 arXiv ID
    arxiv_id = None
    arxiv_match = ARXIV_PATTERN.search(body_text)
    if arxiv_match:
        arxiv_id = arxiv_match.group(1).strip().rstrip(".,;")

    # 5. 标题启发式抽取：
    title = None

    # A. 优先检查带引号的标题: "Title here" 或 “Title here”
    quote_match = re.search(r'["“]([^"”]{6,150})["”]', body_text)
    if quote_match:
        title = quote_match.group(1).strip()

    # B. 若有年份括号如 (2017). Title here.
    if not title and year:
        after_year = re.split(r"\(\s*" + str(year) + r"[a-z]?\s*\)[\.,]?\s*", body_text, maxsplit=1)
        if len(after_year) > 1:
            title_candidate = re.split(r"[\.\?\!]\s+", after_year[1], maxsplit=1)[0]
            if len(title_candidate.strip()) >= 8:
                title = title_candidate.strip()

    # 预切分句子（对作者中间名首字母缩写如 V. 或 E. 及常见缩写进行保护，供标题与作者提取复用）
    p_text = re.sub(r'\b([A-Z])\.\s+', r'\1._DOT_ ', body_text)
    p_text = re.sub(r'\bi\.e\.\s+', 'i.e._DOT_ ', p_text)
    p_text = re.sub(r'\be\.g\.\s+', 'e.g._DOT_ ', p_text)
    parts = [p.replace('._DOT_', '.').strip() for p in re.split(r"[\.\?\!]\s+", p_text) if p.strip()]

    # C. IEEE/NeurIPS 格式: Authors. Title. In Venue, Year.
    if not title:
        for cand in parts[1:]:
            cand_clean = cand.strip()
            if len(cand_clean) >= 6 and len(cand_clean.split()) >= 2:
                if not VENUE_KEYWORDS.match(cand_clean) and not re.match(r"^[\d\s,\-–]+$", cand_clean):
                    title = re.sub(r",\s*(?:19|20)\d{2}[a-z]?\.?$", "", cand_clean).strip()
                    break

    # 6. 作者列表提取
    authors: List[str] = []
    if parts:
        authors_clean = re.sub(r"\s+&\s+|\s+and\s+", ", ", parts[0])
        authors = [a.strip() for a in authors_clean.split(",") if a.strip() and len(a.strip()) > 1]

    # 7. 若仍无 key (无标号作者-年份型)，优先根据第一作者姓氏与年份自动生成
    if not key:
        first_author = authors[0] if authors else "Author"
        tokens = first_author.split()
        last_name = tokens[-1] if tokens else first_author
        is_multi = len(authors) > 1 or "et al" in body_text.lower()
        key_year = f"{year}{year_suffix}" if (year and year_suffix) else (str(year) if year else "n.d.")
        if last_name != "Author" and key_year != "n.d.":
            key = f"[{last_name} et al., {key_year}]" if is_multi else f"[{last_name}, {key_year}]"
        elif default_key:
            key = default_key if default_key.startswith("[") else f"[{default_key}]"
        else:
            key = f"[{last_name}, {key_year}]"

    return CitationRef(
        key=key,
        raw_text=entry_clean,
        title=title,
        authors=authors,
        year=year,
        doi=doi,
        arxiv_id=arxiv_id,
        status=CitationStatus.RESOLVED if (title or doi or arxiv_id) else CitationStatus.UNRESOLVED
    )


def _is_same_citation_entry(a: CitationRef, b: CitationRef) -> bool:
    """判断两条 CitationRef 是否实质指向同一篇文献（用于去重）"""
    if a.doi and b.doi and a.doi.lower().strip() == b.doi.lower().strip():
        return True
    if a.arxiv_id and b.arxiv_id and a.arxiv_id.strip() == b.arxiv_id.strip():
        return True
    if a.title and b.title:
        norm_a = re.sub(r"\W+", "", a.title.lower())
        norm_b = re.sub(r"\W+", "", b.title.lower())
        if norm_a == norm_b or (len(norm_a) >= 15 and (norm_a in norm_b or norm_b in norm_a)):
            return True
    norm_raw_a = re.sub(r"\W+", "", a.raw_text.lower())[:80]
    norm_raw_b = re.sub(r"\W+", "", b.raw_text.lower())[:80]
    return norm_raw_a == norm_raw_b


def parse_references_section(ref_section_text: str) -> Dict[str, CitationRef]:
    """
    解析整段 References 文本，返回引用字典，Key 为如 '[1]' 或 '[Author, Year]'，Value 为 CitationRef
    支持数字标号型与无标号作者-年份型，自动去重与冲突消歧。
    """
    citations: Dict[str, CitationRef] = {}
    lines = ref_section_text.strip().split("\n")

    def _add_citation(ref: Optional[CitationRef]) -> None:
        if not ref or not ref.key:
            return
        if ref.key in citations:
            existing = citations[ref.key]
            if _is_same_citation_entry(existing, ref):
                # 完全相同文献重复出现，保留第一条
                return
            # 不同文献发生 key 碰撞（如同作者同年不同文献）-> 自动消歧
            m_key = re.match(r"^\[(.*?)(\s*,\s*(?:19|20)\d{2})([a-z])?\]$", ref.key)
            if m_key:
                author_part = m_key.group(1)
                year_part = m_key.group(2)
                if existing.key == ref.key:
                    existing_new_key = f"[{author_part}{year_part}a]"
                    existing.key = existing_new_key
                    citations[existing_new_key] = existing
                    del citations[ref.key]
                for letter in "bcdefghijklmnopqrstuvwxyz":
                    candidate_key = f"[{author_part}{year_part}{letter}]"
                    if candidate_key not in citations:
                        ref.key = candidate_key
                        citations[candidate_key] = ref
                        return
            # 其它 key 冲突，追加序号
            sub_idx = 2
            while f"{ref.key[:-1]}_{sub_idx}]" in citations:
                sub_idx += 1
            ref.key = f"{ref.key[:-1]}_{sub_idx}]"
            citations[ref.key] = ref
        else:
            citations[ref.key] = ref

    # 检查是否包含数字编号开头 (如 [1] 或 1. 或 - [1])
    has_numbered_bullets = any(NUMBERED_ENTRY_RE.match(l.strip()) for l in lines)

    if has_numbered_bullets:
        current_entry_lines: List[str] = []

        def commit_entry():
            if current_entry_lines:
                full_entry = " ".join(l.strip() for l in current_entry_lines if l.strip())
                ref = parse_reference_entry(full_entry)
                _add_citation(ref)
                current_entry_lines.clear()

        for line in lines:
            stripped = line.strip()
            if not stripped:
                continue

            # 判断是否是新引用的开头（如 [1] 或 1. 或 - [1]）
            is_new_entry = bool(NUMBERED_ENTRY_RE.match(stripped))
            if is_new_entry:
                commit_entry()
                current_entry_lines.append(stripped)
            else:
                if current_entry_lines:
                    current_entry_lines.append(stripped)

        commit_entry()
    else:
        # 作者-年份型或非编号型排版：按双换行切分段落
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", ref_section_text) if p.strip()]
        for idx, p in enumerate(paragraphs, 1):
            single_line_entry = " ".join(p.split())
            ref = parse_reference_entry(single_line_entry, default_key=f"[ref_{idx}]")
            _add_citation(ref)

    return citations
