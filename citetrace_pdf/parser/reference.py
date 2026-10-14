from __future__ import annotations

import re

from .cleaner import TextCleaner


class ReferenceFormatter:
    ENTRY_MARKER = re.compile(r"(\[\d+\])")
    BULLET_START = re.compile(r"^[-*]\s+[A-Z\u00C0-\u017F]")
    AUTHOR_YEAR_START = re.compile(
        r"^(?:[-*]\s*)?([A-Z\u00C0-\u017F][^(\n]{1,300}?\s*\((?:19|20)\d{2}[a-z]?\)\.?)"
    )
    # 兜底守卫：过滤条目末尾误混入的图表题注、表格分隔线、附录标题、Checklist、致谢等
    NON_REF_TRAILING = re.compile(
        r"(?:\n\s*|\s+)(?:(?:Figure|Fig\.|Table)\s+\d+[:.]|\|\s*---\s*\||"
        r"(?:Attention\s+Visualizations|Appendices|Appendix\b|NeurIPS\s+Checklist|Checklist\b|"
        r"Reproducibility\s+Checklist|Acknowledgments\b|Acknowledgements\b).*).*",
        re.DOTALL | re.I,
    )

    @classmethod
    def _is_pseudo_reference(cls, body: str) -> bool:
        b = body.strip()
        return bool(re.match(
            r"^(?:Appendix|[A-G]\.?\s+[A-Z]|(?:Attention\s+)?Visualizations|"
            r"Checklist|NeurIPS\s+Checklist|Reproducibility\s+Checklist|"
            r"Acknowledgments|Acknowledgements)\b",
            b,
            re.I,
        ))

    @classmethod
    def _clean_entry(cls, raw: str) -> str:
        body = TextCleaner.dehyphenate(raw)
        body = cls.NON_REF_TRAILING.sub("", body)
        body = re.sub(r"\s+", " ", body).strip()
        body = re.sub(r"^(?:\s*[-*•]\s*)+", "", body)
        # 剥离多余的前置数字编号如 "1. "，若在 [1] 之后残留
        body = re.sub(r"^\d+\.\s*", "", body).strip()
        body = re.sub(r"(?:\s*[-*•]\s*)+$", "", body).strip()
        if cls._is_pseudo_reference(body):
            return ""
        return body

    @classmethod
    def format(cls, text: str) -> str:
        if not text:
            return ""

        # 1. 预清理 Markdown 列表符号（如 '- [1]', '* [1]', '• [1]'）
        cleaned_text = re.sub(r"(?:^|\n)\s*[-*•]\s*(?=\[\d+\])", "\n", text)
        # 2. 将带列表符号的普通编号转换为标准编号: "- 1. ", "* 1. ", "• 1. " -> "\n1. "
        cleaned_text = re.sub(r"(?:^|\n)\s*[-*•]\s*(?=\d+\.)", "\n", cleaned_text)
        # 3. 将行首普通数字编号 "1. " 或 "1 " 规整化为 "[1] "（当其后紧随大写字母或作者名）
        cleaned_text = re.sub(r"(?:^|\n)\s*(\d+)\.\s+(?=[A-Z\u00C0-\u017F])", r"\n[\1] ", cleaned_text)
        # 4. 将补零标号如 [01], [04] 归一化为 [1], [4]
        cleaned_text = re.sub(r"\[0*(\d+)\]", r"[\1]", cleaned_text)

        # 1. 存在 [n] 数字标号时，按 [n] 切分
        if cls.ENTRY_MARKER.search(cleaned_text):
            parts = cls.ENTRY_MARKER.split(cleaned_text)
            entries: list[str] = []
            current_tag: str | None = None
            seen_tags: set[str] = set()
            seen_fingerprints: set[str] = set()

            for part in parts:
                value = part.strip()
                if not value:
                    continue
                if cls.ENTRY_MARKER.fullmatch(value):
                    current_tag = value
                    continue
                body = cls._clean_entry(value)
                if body:
                    fp = re.sub(r"\W+", "", body.lower())[:80]
                    tag = current_tag or ""
                    is_dup = False
                    if tag and tag in seen_tags:
                        is_dup = True
                    if fp and len(fp) >= 15 and fp in seen_fingerprints:
                        is_dup = True

                    if not is_dup:
                        if tag:
                            seen_tags.add(tag)
                        if fp and len(fp) >= 15:
                            seen_fingerprints.add(fp)
                        entries.append(f"{tag} {body}" if tag else body)
                current_tag = None
            if current_tag and current_tag not in seen_tags:
                entries.append(current_tag)
            return "\n\n".join(entries) if entries else TextCleaner.clean(text)

        # 2. 无 [n] 标号的作者-年份排版，通过作者-年份、列表符号或段落边界自适应切分
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        if not lines:
            return TextCleaner.clean(text)

        has_bullets = any(cls.BULLET_START.match(l) for l in lines)
        has_author_year = any(cls.AUTHOR_YEAR_START.match(l) for l in lines)

        cleaned_entries: list[str] = []
        if has_bullets or has_author_year:
            curr: list[str] = []
            for line in lines:
                is_new = (has_bullets and cls.BULLET_START.match(line)) or bool(cls.AUTHOR_YEAR_START.match(line))
                if is_new:
                    if curr:
                        cleaned_entries.append(" ".join(curr))
                    curr = [line.lstrip("-* \t")]
                else:
                    if curr:
                        curr.append(line)
                    else:
                        curr = [line]
            if curr:
                cleaned_entries.append(" ".join(curr))
        else:
            if "\n\n" in text:
                for p in text.split("\n\n"):
                    p_clean = re.sub(r"\s+", " ", p).strip()
                    if len(p_clean) >= 15:
                        cleaned_entries.append(p_clean)
            else:
                curr = []
                for line in lines:
                    if not curr:
                        curr.append(line)
                        continue
                    prev = curr[-1]
                    ends_ref = bool(re.search(r'(?:\b(19\d{2}|20\d{2})[a-z]?[\.\)]|arXiv:\d{4}\.\d{4,5}[,\.]?)\s*$', prev.strip()))
                    starts_author = bool(re.match(r'^[A-Z\u00C0-\u017F][A-Za-z\u00C0-\u017F\.\s,\'\-]{2,}', line))
                    if ends_ref and starts_author:
                        cleaned_entries.append(" ".join(curr))
                        curr = [line]
                    else:
                        curr.append(line)
                if curr:
                    cleaned_entries.append(" ".join(curr))

        # 格式化并自动统一编号，并进行去重
        numbered_entries = []
        seen_fingerprints: set[str] = set()
        seen_tags: set[str] = set()
        idx = 1
        for entry in cleaned_entries:
            e_clean = cls._clean_entry(entry)
            if len(e_clean) >= 15 and not cls._is_pseudo_reference(e_clean):
                fp = re.sub(r"\W+", "", e_clean.lower())[:80]
                if fp and len(fp) >= 15 and fp in seen_fingerprints:
                    continue
                seen_fingerprints.add(fp)

                m_tag = re.match(r"^\[(\d+)\]", e_clean)
                if m_tag:
                    canon_tag = f"[{int(m_tag.group(1))}]"
                    body_only = e_clean[m_tag.end():].strip()
                    if canon_tag not in seen_tags:
                        seen_tags.add(canon_tag)
                        numbered_entries.append(f"{canon_tag} {body_only}")
                else:
                    canon_tag = f"[{idx}]"
                    seen_tags.add(canon_tag)
                    numbered_entries.append(f"{canon_tag} {e_clean}")
                    idx += 1

        return "\n\n".join(numbered_entries) if numbered_entries else TextCleaner.clean(text)
