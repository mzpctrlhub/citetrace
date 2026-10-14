from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..model.document import Section
from ..model.layout import PageLayout
from ..pdf.layout import LayoutNormalizer
from .cleaner import TextCleaner
from .heading import Heading, HeadingDetector
from .reference import ReferenceFormatter


@dataclass
class _SectionBuffer:
    title: str
    start_page: int
    end_page: int
    heading: Heading | None = None
    paragraphs: list[str] = field(default_factory=list)


class _MonotonicGuard:
    """
    对候选章节号实施单调性约束，防止跑分、图表序号、列表项等被误识别为章节标题。

    策略：
    - 首个编号章节只允许从 1 或 2 开始，防止图表中散落的数字（如 "5. Output"）污染单调上界。
    - 一级主章节号（如 1, 2, 3）必须严格递增（major > _max_major_seen），且跳跃不超过 max_jump (2)。
      这消除了正文中编号列表项（如在 4.2 节后出现 "2. Baseline"）或示意图标签（如 "1. Output"）的误判。
    - 子章节（如 3.1, 3.2）的主章节号必须在 [_max_major_seen - 1, _max_major_seen + 1] 范围内。
    - 字母型附录标题（如 A, B, A.1）必须在正文已推进到至少第 2 节后才允许生效，防止以 "A ..." 开头的论文主标题被误判为附录 A。
    """

    def __init__(self, max_jump: int = 2) -> None:
        self.max_jump = max_jump
        self._max_major_seen: int = 0

    def accept(self, heading: Heading) -> bool:
        """返回 True 表示此 heading 通过单调性检查，应予接受。"""
        if heading.kind == "appendix" and heading.number and re.fullmatch(r"[A-G](?:\.\d+)*", heading.number):
            return self._max_major_seen >= 2

        if heading.kind != "numbered" or heading.number is None:
            return True

        parts = heading.number.split(".")
        major = int(parts[0])
        is_top_level = len(parts) == 1

        # 首个编号章节只允许从 1 或 2 开始
        if self._max_major_seen == 0:
            if major > 2:
                return False
            self._max_major_seen = major
            return True

        if is_top_level:
            # 一级主章节必须严格递增且不发生大幅跳跃
            if major <= self._max_major_seen:
                return False
            if major > self._max_major_seen + self.max_jump:
                return False
        else:
            # 子章节主号必须与当前主号一致或最多偏差 1
            if major > self._max_major_seen + 1:
                return False
            if major < self._max_major_seen - 1:
                return False

        if major > self._max_major_seen:
            self._max_major_seen = major
        return True


class SectionParser:
    """Turn normalized page blocks into sections through a small references-aware state machine."""

    _NON_HEADING_PREFIXES = ("Figure", "Fig.", "Table", "|", "[")
    _DIAGRAM_LABELS = {"input", "output", "inputs", "outputs", "state", "states"}
    _INLINE_BOLD_HEADING = re.compile(
        r"(?<!^)(?<!\n)\s*(\*{2}\d{1,2}(?:\.\d+)*\.?\s+[A-Z][A-Za-z0-9\s\-–—:()/'\"?&+]{2,60}\*{2})"
    )

    def __init__(self, detector: HeadingDetector | None = None) -> None:
        self.detector = detector or HeadingDetector()

    @staticmethod
    def _is_appendix_heading(head: Heading | None, guard: _MonotonicGuard) -> bool:
        if head is None:
            return False
        return (
            (head.kind in ("appendix", "acknowledgments") and guard.accept(head))
            or (head.kind == "numbered" and guard.accept(head) and "appendix" in head.text.lower())
        )

    def _detect_line_heading(self, lines: list[str], index: int) -> tuple[Heading | None, int]:
        line = lines[index]
        if line.startswith(self._NON_HEADING_PREFIXES) or line.endswith((",", ";")):
            return None, 1
        clean_line = line.rstrip(".").strip()
        direct = self.detector.detect(clean_line)
        if direct:
            return direct, 1
        if index + 1 < len(lines) and self.detector.NUMBER_ONLY.fullmatch(line):
            next_line = lines[index + 1]
            next_clean = next_line.rstrip(".").strip()
            if (
                len(next_clean) <= 80
                and re.match(r"^[A-Z]", next_clean)
                and next_clean.lower() not in self._DIAGRAM_LABELS
                and not next_line.startswith(self._NON_HEADING_PREFIXES)
                and not next_line.endswith((",", ";"))
            ):
                combined = f"{line}. {next_clean}" if not line.endswith(".") else f"{line} {next_clean}"
                cand = self.detector.detect(combined)
                if cand:
                    return cand, 2
        return None, 1

    def parse(self, pages: list[PageLayout]) -> list[Section]:
        sections: list[Section] = []
        current = _SectionBuffer("Abstract", 1, 1)
        in_references = False
        reference_last_page = 0
        guard = _MonotonicGuard(max_jump=2)

        def finish(*, keep_empty: bool = False) -> None:
            nonlocal current
            raw = "\n\n".join(current.paragraphs)
            title_is_default = current.title.lower() in {"abstract", "other"}
            if raw or keep_empty or not title_is_default:
                content = ReferenceFormatter.format(raw) if current.title.lower() == "references" else raw
                heading = current.heading
                sections.append(Section(
                    title=current.title, start_page=current.start_page, end_page=current.end_page,
                    content=content, level=heading.level if heading else 0,
                    number=heading.number if heading else None,
                ))

        for page in pages:
            ordered = LayoutNormalizer.reorder(page.blocks, page.width)
            if not ordered:
                continue

            # 若处于参考文献章节，新页面检查是否已离开 References
            if in_references and page.number > reference_last_page:
                first_meaningful_line = next(
                    (l.strip() for b in ordered for l in b.text.splitlines() if l.strip()), ""
                )
                head = self.detector.detect(first_meaningful_line)
                if not self._is_appendix_heading(head, guard):
                    head = None

                # 兜底：若前序已明确为 [n] 标号型参考文献，但新页面完全没有任何 [n]，且首行非文献出处/会议接续
                if head is None and any(re.search(r"\[\d+\]", p) for p in current.paragraphs):
                    has_citation_markers = any(
                        re.search(r"^\s*\[\d+\]", l) for b in ordered for l in b.text.splitlines()
                    )
                    if not has_citation_markers:
                        clean_first = first_meaningful_line.rstrip(".").strip()
                        if (
                            clean_first
                            and len(clean_first) <= 60
                            and not clean_first.startswith(self._NON_HEADING_PREFIXES)
                            and not clean_first.endswith((",", ";", ":", "-", "–"))
                            and not self.detector.is_citation_noise(clean_first)
                            and not self.detector.BIB_NOISE.search(clean_first)
                            and re.match(r"^[A-Z]", clean_first)
                        ):
                            head = Heading(clean_first, 1, None, "appendix")

                if head is not None:
                    finish()
                    in_references = False
                    current = _SectionBuffer(head.text, page.number, page.number, head)

            for block in ordered:
                raw_text = block.text.strip()
                if not raw_text:
                    continue
                if block.kind == "table":
                    if in_references:
                        # 参考文献内部绝不可能包含独立的表格，若出现表格说明已进入附录
                        finish()
                        in_references = False
                        current = _SectionBuffer("Appendix", page.number, page.number, Heading("Appendix", 1, None, "appendix"))
                    current.paragraphs.append(raw_text)
                    current.end_page = page.number
                    continue

                # 拆分被 pymupdf4llm 合并进段落或脚注内部的加粗编号章节标题（如 **6. Character-Level Machine Translation**）
                raw_text = self._INLINE_BOLD_HEADING.sub(r"\n\1\n", raw_text)
                lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
                buffer_lines: list[str] = []
                index = 0
                while index < len(lines):
                    line = lines[index]
                    clean_line = line.rstrip(".").strip()

                    if in_references:
                        appendix, advance = self._detect_line_heading(lines, index)
                        if len(clean_line) <= 80 and self._is_appendix_heading(appendix, guard):
                            self._append_buffer(current, buffer_lines, page.number)
                            finish()
                            current = _SectionBuffer(appendix.text, page.number, page.number, appendix)
                            in_references = False
                        else:
                            buffer_lines.append(line)
                            reference_last_page = page.number
                        index += advance
                        continue

                    # Some PDF extractors combine a parent and child heading on one line.
                    parent_child = re.match(
                        r"^(\d+\.?\s+[A-Z][A-Za-z0-9\s\-–—]+?[.:]?)\s+(\d+\.\d+.*)$", line
                    )
                    if parent_child:
                        lines.insert(index + 1, parent_child.group(2).strip())
                        line = parent_child.group(1).rstrip(".:").strip()
                        lines[index] = line

                    if self.detector.is_citation_noise(line):
                        buffer_lines.append(line)
                        index += 1
                        continue

                    candidate, advance = self._detect_line_heading(lines, index)
                    run_in_remainder: str | None = None

                    # Inline subsection headings: separate heading from same-line body.
                    if candidate is None:
                        clean_for_runin = self.detector.clean_markdown_markers(line)
                        run_in = self.detector.RUN_IN.match(clean_for_runin)
                        if run_in:
                            candidate = self.detector.detect(f"{run_in.group(1)} {run_in.group(2).rstrip('.: –—').strip()}")
                            if candidate:
                                run_in_remainder = run_in.group(3).strip()

                    # ── 单调性守卫：过滤跑分、图注序号等误判的伪章节 ──────────
                    if candidate is not None and not guard.accept(candidate):
                        candidate = None
                        run_in_remainder = None

                    if candidate:
                        if candidate.text.strip().lower() == current.title.strip().lower():
                            if run_in_remainder:
                                buffer_lines.append(run_in_remainder)
                            index += advance
                            continue
                        self._append_buffer(current, buffer_lines, page.number)
                        finish()
                        current = _SectionBuffer(candidate.text, page.number, page.number, candidate)
                        if run_in_remainder:
                            buffer_lines.append(run_in_remainder)
                        if candidate.kind == "references":
                            in_references = True
                            reference_last_page = page.number
                    else:
                        buffer_lines.append(line)
                    index += advance

                self._append_buffer(current, buffer_lines, page.number)

        finish()
        return self._rebind_floating_tables(sections)

    @staticmethod
    def _rebind_floating_tables(sections: list[Section]) -> list[Section]:
        """Rebind page-top LaTeX floating tables (e.g. Table 2 on page 8) to the same-page section that actually cites them."""
        table_cap_re = re.compile(r"(?:^|\n\n)(Table\s+(\d+)[:.][^\n]+\n\n(?:\|[^\n]+\n)+)", re.M)
        for i, sec in enumerate(sections):
            for match in list(table_cap_re.finditer(sec.content)):
                full_table_block = match.group(1).strip()
                table_num = match.group(2)
                prose_without_table = (sec.content[:match.start()] + " " + sec.content[match.end():]).strip()
                ref_re = re.compile(rf"\bTable\s+{re.escape(table_num)}\b", re.I)
                if ref_re.search(prose_without_table):
                    continue
                # Look for a subsequent section on the same page range that explicitly cites Table N
                for j in range(i + 1, min(len(sections), i + 5)):
                    cand_sec = sections[j]
                    if cand_sec.start_page > sec.end_page:
                        break
                    if ref_re.search(cand_sec.content):
                        sections[i] = Section(
                            title=sec.title,
                            start_page=sec.start_page,
                            end_page=sec.end_page,
                            content=prose_without_table,
                            level=sec.level,
                            number=sec.number,
                        )
                        sec = sections[i]
                        sections[j] = Section(
                            title=cand_sec.title,
                            start_page=cand_sec.start_page,
                            end_page=cand_sec.end_page,
                            content=f"{full_table_block}\n\n{cand_sec.content}".strip(),
                            level=cand_sec.level,
                            number=cand_sec.number,
                        )
                        break
        return sections

    @staticmethod
    def _append_buffer(section: _SectionBuffer, lines: list[str], page_no: int) -> None:
        if lines:
            cleaned = TextCleaner.clean("\n".join(lines))
            if cleaned:
                section.paragraphs.append(cleaned)
            section.end_page = page_no
            lines.clear()
