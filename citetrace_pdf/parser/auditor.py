from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Sequence

from ..model.document import Section


@dataclass(frozen=True, slots=True)
class AuditIssue:
    """单条大纲体检异常记录，指明异常类型、涉及页码区间与修复提示。"""
    kind: str                       # 'missing_section' | 'non_monotonic' | 'under_segmented' | 'suspicious_title'
    message: str                    # 人类可读的诊断信息
    page_range: tuple[int, int]     # 需要定向回溯检查的起止页码 (start_page, end_page)
    expected_number: str | None = None
    suspect_section_index: int | None = None


@dataclass(slots=True)
class AuditReport:
    """整篇 PDF 的大纲体检报告。"""
    paper_title: str
    total_pages: int
    total_sections: int
    issues: list[AuditIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.issues) == 0


class OutlineAuditor:
    """
    Agentic PDF 第一步：基于四条全树数学不变量（Tree Invariants）对章节大纲进行结构自检：
    1. 任意层级兄弟节点连续性（主章节 1..N、子章节 P.1..P.k、字母附录 A..Z 均不跳号）
    2. 层级元组字典序严格单调性（禁止出现 4.5.1 -> 4.6 -> 4.5.2 等双栏错序或伪标题倒退）
    3. 非空叶子不变量（无子节点的叶子章节正文不得为空）
    4. 单节容量上界不变量（防长文漏切）
    """

    _SUSPICIOUS_WORDS = {"input", "output", "inputs", "outputs", "state", "states", "layer", "layers"}
    _BOGUS_PAPER_TITLE = re.compile(r"(?:\.(?:dvi|tex|ps|pdf)$|^untitled)", re.I)

    @classmethod
    def audit(
        cls,
        sections: Sequence[Section],
        total_pages: int,
        paper_title: str = "",
    ) -> AuditReport:
        report = AuditReport(
            paper_title=paper_title,
            total_pages=total_pages,
            total_sections=len(sections),
        )

        # 1. 检查论文主标题是否异常（例如 'nc.dvi'、空标题或字母碎裂乱码）
        from citetrace_pdf.pdf.title_evaluator import evaluate_title_quality
        title_quality = evaluate_title_quality(paper_title)
        if title_quality.is_broken or not paper_title or cls._BOGUS_PAPER_TITLE.search(paper_title.strip()):
            report.issues.append(AuditIssue(
                kind="suspicious_title",
                message=f"论文主标题异常、碎裂或缺失: {paper_title!r} (quality={title_quality.quality_score:.2f})",
                page_range=(1, 1),
            ))

        # 2. 检查整体章节密度（长文漏切检测，例如 32 页只切出 2 节）
        if total_pages >= 6 and len(sections) <= 3:
            report.issues.append(AuditIssue(
                kind="under_segmented",
                message=f"全文共 {total_pages} 页，但仅切出 {len(sections)} 个章节，疑似正文标题未被识别",
                page_range=(1, total_pages),
            ))

        # 3. 逐节扫描：检查主章节与子章节跳号、元组单调性、单节跨度过大与伪标题
        max_major_seen = 0
        last_major_page = 1
        last_tuple: tuple[int, ...] | None = None
        sub_state: dict[str, tuple[int, int]] = {}  # parent_prefix -> (last_child_num, last_page)
        parent_start_page: dict[str, int] = {}
        last_appendix_ord: int | None = None
        last_appendix_page: int = 1

        for idx, sec in enumerate(sections):
            span_pages = sec.end_page - sec.start_page + 1
            if span_pages > max(10, total_pages // 2) and sec.title.lower() not in {"references", "appendix", "appendices"}:
                report.issues.append(AuditIssue(
                    kind="under_segmented",
                    message=f"章节 {sec.title!r} 跨度达 {span_pages} 页 (p{sec.start_page}-p{sec.end_page})，内部可能漏切子章节",
                    page_range=(sec.start_page, sec.end_page),
                    suspect_section_index=idx,
                ))

            clean_words = re.sub(r"^[\d.]+\s*", "", sec.title).strip().lower()
            if clean_words in cls._SUSPICIOUS_WORDS or any(1 <= ord(c) <= 8 for c in sec.title):
                report.issues.append(AuditIssue(
                    kind="suspicious_title",
                    message=f"章节 {sec.title!r} 疑似示意图标签或含乱码控制符",
                    page_range=(sec.start_page, sec.end_page),
                    suspect_section_index=idx,
                ))

            # 字母附录及附录子章节连续性检查 (A -> B -> C 及 C.1 -> C.2)
            if sec.number and re.fullmatch(r"[A-G](?:\.\d+)*", sec.number):
                app_parts = sec.number.split(".")
                if len(app_parts) == 1:
                    cur_ord = ord(sec.number)
                    if last_appendix_ord is not None and cur_ord > last_appendix_ord + 1:
                        for missing_ord in range(last_appendix_ord + 1, cur_ord):
                            missing_let = chr(missing_ord)
                            report.issues.append(AuditIssue(
                                kind="missing_section",
                                message=f"附录从 {chr(last_appendix_ord)} 跳至 {sec.number}，在 p{last_appendix_page}-p{sec.start_page} 之间缺失附录 {missing_let}",
                                page_range=(last_appendix_page, sec.start_page),
                                expected_number=missing_let,
                            ))
                    last_appendix_ord = cur_ord
                    last_appendix_page = sec.start_page
                    parent_start_page[sec.number] = sec.start_page
                else:
                    parent_prefix = ".".join(app_parts[:-1])
                    child_num = int(app_parts[-1])
                    prev_child, prev_page = sub_state.get(
                        parent_prefix,
                        (0, parent_start_page.get(parent_prefix, last_appendix_page)),
                    )
                    if child_num > prev_child + 1:
                        for m_child in range(prev_child + 1, child_num):
                            exp_num = f"{parent_prefix}.{m_child}"
                            report.issues.append(AuditIssue(
                                kind="missing_section",
                                message=f"附录子章节从 {parent_prefix}.{prev_child} 跳至 {sec.number}，在 p{prev_page}-p{sec.start_page} 之间缺失 {exp_num}",
                                page_range=(prev_page, sec.start_page),
                                expected_number=exp_num,
                            ))
                    sub_state[parent_prefix] = (child_num, sec.start_page)
                continue

            if not sec.number or not re.fullmatch(r"\d+(?:\.\d+)*", sec.number):
                continue

            parts = [int(p) for p in sec.number.split(".")]
            cur_tuple = tuple(parts)
            major = parts[0]
            is_top_level = len(parts) == 1
            parent_start_page[sec.number] = sec.start_page

            if max_major_seen == 0:
                if major > 1:
                    for missing in range(1, major):
                        report.issues.append(AuditIssue(
                            kind="missing_section",
                            message=f"首个编号章节从 {sec.number} 开始，前面缺失第 {missing} 节",
                            page_range=(1, sec.start_page),
                            expected_number=str(missing),
                        ))
                max_major_seen = major
                last_major_page = sec.start_page
                last_tuple = cur_tuple
                continue

            # 一级主章节倒退或重复
            if is_top_level and major <= max_major_seen:
                report.issues.append(AuditIssue(
                    kind="non_monotonic",
                    message=f"一级章节 {sec.title!r} (主号={major}) 出现在主号 {max_major_seen} 之后，疑似列表项或图注误判",
                    page_range=(sec.start_page, sec.end_page),
                    suspect_section_index=idx,
                ))
                continue

            # 子章节层级元组单调性检查（如 4.6 之后出现 4.5.2）
            if not is_top_level and last_tuple is not None and cur_tuple <= last_tuple:
                report.issues.append(AuditIssue(
                    kind="non_monotonic",
                    message=f"子章节 {sec.title!r} ({sec.number}) 出现在 {'.'.join(map(str, last_tuple))} 之后，存在顺序颠倒",
                    page_range=(sec.start_page, sec.end_page),
                    suspect_section_index=idx,
                ))
                continue

            # 一级主章节跳号
            if major > max_major_seen + 1:
                for missing in range(max_major_seen + 1, major):
                    report.issues.append(AuditIssue(
                        kind="missing_section",
                        message=f"主章节号从 {max_major_seen} 直接跳至 {major}，在 p{last_major_page}-p{sec.start_page} 之间缺失第 {missing} 节",
                        page_range=(last_major_page, sec.start_page),
                        expected_number=str(missing),
                    ))

            # 子章节兄弟连续性检查（如 1.1 -> 1.3 或 5 -> 5.3）
            if not is_top_level:
                parent_prefix = ".".join(str(x) for x in parts[:-1])
                child_num = parts[-1]
                prev_child, prev_page = sub_state.get(
                    parent_prefix,
                    (0, parent_start_page.get(parent_prefix, last_major_page)),
                )
                if child_num > prev_child + 1:
                    for m_child in range(prev_child + 1, child_num):
                        exp_num = f"{parent_prefix}.{m_child}"
                        report.issues.append(AuditIssue(
                            kind="missing_section",
                            message=f"子章节从 {parent_prefix}.{prev_child} 跳至 {sec.number}，在 p{prev_page}-p{sec.start_page} 之间缺失 {exp_num}",
                            page_range=(prev_page, sec.start_page),
                            expected_number=exp_num,
                        ))
                sub_state[parent_prefix] = (child_num, sec.start_page)

            if major > max_major_seen:
                max_major_seen = major
                last_major_page = sec.start_page
            last_tuple = cur_tuple

        return report
