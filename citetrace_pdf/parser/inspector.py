from __future__ import annotations

import re
from dataclasses import dataclass

import pymupdf

from ..model.document import Section
from .auditor import AuditReport, OutlineAuditor
from .cleaner import clean_dvips_text, normalize_text


@dataclass(frozen=True, slots=True)
class CandidateHeading:
    """从 PDF 底层 dict 提取的单条候选标题骨架（供本地自愈与第 3 步 LLM 裁判共用）。"""
    page: int
    y0: float
    font_size: float
    is_bold: bool
    text: str


class PageInspector:
    """
    Agentic PDF 第二步：根据 OutlineAuditor 的诊断报告，
    定向透视异常页的底层字体/字号/加粗 span，在本地自动修复伪章节与漏切章节。
    """

    _BOLD_FONT_RE = re.compile(r"(?:bold|medi|cmbx|demi|black|heavy|sfbx)", re.I)

    @classmethod
    def _is_span_bold(cls, span: dict) -> bool:
        return bool(
            cls._BOLD_FONT_RE.search(str(span.get("font", "")))
            or (int(span.get("flags", 0)) & 16)
        )

    @classmethod
    def _iter_visual_lines(cls, page: pymupdf.Page) -> list[tuple[float, float, bool, str, list[dict]]]:
        """遍历页面所有文本行，并将同一视觉行 (y0 差值 < 2.0pt，如章节号 '2' 与标题 'Active Memory Models') 自动合并。"""
        merged: list[tuple[float, float, bool, str, list[dict]]] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                text = clean_dvips_text("".join(s.get("text", "") for s in line.get("spans", []))).strip()
                if not text:
                    continue
                y0 = float(line.get("bbox", [0, 0, 0, 0])[1])
                max_sz = max(float(s.get("size", 0.0)) for s in spans)
                is_bold = any(cls._is_span_bold(s) for s in spans)
                if merged and abs(y0 - merged[-1][0]) < 2.0:
                    py0, psz, pbold, ptxt, pspans = merged[-1]
                    merged[-1] = (py0, max(psz, max_sz), pbold or is_bold, f"{ptxt} {text}", pspans + spans)
                else:
                    merged.append((y0, max_sz, is_bold, text, spans))
        return merged

    @classmethod
    def extract_skeleton(
        cls,
        doc: pymupdf.Document,
        page_range: tuple[int, int] | None = None,
    ) -> list[CandidateHeading]:
        """
        提取指定页码区间内的候选标题骨架：
        计算正文基准字号，筛选出【加粗】或【字号显著大于正文】的短行 (< 85 字符)。
        """
        start_p = max(1, page_range[0]) if page_range else 1
        end_p = min(len(doc), page_range[1]) if page_range else len(doc)

        sizes: list[float] = []
        raw_lines: list[tuple[int, float, float, bool, str]] = []

        for pno in range(start_p, end_p + 1):
            for y0, max_sz, is_bold, text, _ in cls._iter_visual_lines(doc[pno - 1]):
                if len(text) >= 20:
                    sizes.append(max_sz)
                raw_lines.append((pno, y0, max_sz, is_bold, text))

        body_size = sorted(sizes)[len(sizes) // 2] if sizes else 10.0

        skeleton: list[CandidateHeading] = []
        for pno, y0, sz, is_bold, text in raw_lines:
            clean = re.sub(r"\s+", " ", text).strip()
            if len(clean) < 3 or len(clean) > 85:
                continue
            if re.fullmatch(r"\d{1,3}", clean):
                continue
            is_larger = sz > body_size + 0.4
            looks_numbered = bool(
                re.match(r"^(?:\d+(?:\.\d+)*\.?\s+[A-Z]|Abstract$|References$|Appendix\b)", clean, re.I)
            )
            if sz >= body_size - 0.2 and (is_bold or is_larger or looks_numbered):
                skeleton.append(CandidateHeading(
                    page=pno,
                    y0=round(y0, 1),
                    font_size=round(sz, 2),
                    is_bold=is_bold,
                    text=clean,
                ))
        return skeleton

    @classmethod
    def find_missing_numbered_heading(
        cls,
        doc: pymupdf.Document,
        page_range: tuple[int, int],
        expected_number: str,
    ) -> CandidateHeading | None:
        """
        定向在 page_range 页码区间内搜寻缺失的第 expected_number 节标题
        （既检查合并后的视觉行，也检查被合并在段落内部的独立加粗 span）。
        """
        start_p = max(1, page_range[0])
        end_p = min(len(doc), page_range[1])
        pattern = re.compile(
            rf"^(?:\*{{0,2}}#*\s*)?(?:[-•]\s+)?({re.escape(expected_number)}[.:]?\s+(?:[A-Z]|\d+\s+[A-Z]{{2,}})[A-Za-z0-9\s\-–—:()/'\"?&+]{{2,75}})$"
        )

        for pno in range(start_p, end_p + 1):
            for y0, max_sz, is_bold, text, spans in cls._iter_visual_lines(doc[pno - 1]):
                m_line = pattern.match(normalize_text(text))
                if m_line:
                    return CandidateHeading(
                        page=pno,
                        y0=round(y0, 1),
                        font_size=round(max_sz, 2),
                        is_bold=is_bold,
                        text=m_line.group(1).strip(),
                    )
                for span in spans:
                    stext = normalize_text(span.get("text", "")).strip()
                    m_span = pattern.match(stext)
                    if m_span:
                        return CandidateHeading(
                            page=pno,
                            y0=round(float(span.get("bbox", [0, 0, 0, 0])[1]), 1),
                            font_size=round(float(span.get("size", 10.0)), 2),
                            is_bold=cls._is_span_bold(span),
                            text=m_span.group(1).strip(),
                        )
        return None

    @classmethod
    def auto_repair(
        cls,
        doc: pymupdf.Document,
        sections: list[Section],
        report: AuditReport,
    ) -> tuple[list[Section], AuditReport]:
        """
        根据 AuditReport 执行本地确定性自愈：
        1. 先移除明显的伪标题（如 '5. Output'），再复检并移除真实倒退章节；
        2. 定向在异常页中根据底层视觉行/span 找回并切开 missing_section；
        3. 返回修复后的 sections 与二次复检的 AuditReport。
        """
        if report.passed:
            return sections, report

        repaired = list(sections)

        # Step A: 分两阶段移除伪标题与非单调章节，将其正文归还给上一章节
        for target_kind in ("suspicious_title", "non_monotonic"):
            cur_rep = OutlineAuditor.audit(repaired, len(doc), report.paper_title)
            drop_indices = {
                iss.suspect_section_index
                for iss in cur_rep.issues
                if iss.kind == target_kind and iss.suspect_section_index is not None
            }
            if drop_indices:
                merged: list[Section] = []
                for idx, sec in enumerate(repaired):
                    if idx in drop_indices and merged:
                        prev = merged[-1]
                        combined_content = f"{prev.content}\n\n{sec.title}\n\n{sec.content}".strip()
                        merged[-1] = Section(
                            title=prev.title,
                            start_page=prev.start_page,
                            end_page=max(prev.end_page, sec.end_page),
                            content=combined_content,
                            level=prev.level,
                            number=prev.number,
                        )
                    else:
                        merged.append(sec)
                repaired = merged

        # Step B: 针对缺失的主章节号，定向在异常页码区间的底层 span 中定位并切分
        recheck = OutlineAuditor.audit(repaired, len(doc), report.paper_title)
        for iss in recheck.issues:
            if iss.kind != "missing_section" or not iss.expected_number:
                continue
            cand = cls.find_missing_numbered_heading(doc, iss.page_range, iss.expected_number)
            if cand is None:
                continue
            for idx, sec in enumerate(repaired):
                if not (sec.start_page <= cand.page <= sec.end_page):
                    continue
                clean_heading = re.sub(r"\*+", "", cand.text).strip()
                pattern_in_content = re.compile(
                    rf"(?:\*{{0,2}}#*\s*)?{re.escape(clean_heading)}(?:\*{{0,2}})?"
                )
                m_pos = pattern_in_content.search(sec.content)
                if m_pos:
                    before = sec.content[:m_pos.start()].strip()
                    after = sec.content[m_pos.end():].strip()
                    repaired[idx] = Section(
                        title=sec.title,
                        start_page=sec.start_page,
                        end_page=cand.page,
                        content=before,
                        level=sec.level,
                        number=sec.number,
                    )
                    repaired.insert(idx + 1, Section(
                        title=clean_heading,
                        start_page=cand.page,
                        end_page=sec.end_page,
                        content=after,
                        level=len(iss.expected_number.split(".")),
                        number=iss.expected_number,
                    ))
                    break

        final_report = OutlineAuditor.audit(repaired, len(doc), report.paper_title)
        return repaired, final_report
