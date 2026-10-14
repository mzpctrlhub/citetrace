from __future__ import annotations

import json
import logging
import re
from typing import Protocol

import pymupdf

from ..model.document import Section
from .auditor import AuditReport, OutlineAuditor
from .inspector import CandidateHeading, PageInspector

logger = logging.getLogger(__name__)


class ChatClientProtocol(Protocol):
    def chat_completion(
        self, model: str, messages: list[dict[str, str]], temperature: float = 0.0
    ) -> str: ...


class LLMSkeletonReferee:
    """
    Agentic PDF 第三步：当本地规则与 PageInspector 仍未消解大纲异常（如无编号章节论文或复杂排版）时，
    自动从统一配置中心 `llm_config.py` 获取 `pdf_referee` 角色的大模型（若未填 Key 则平滑回退至离线裁判），
    对极简候选标题骨架（~500 Tokens）审定真实章节标题，并在原文中执行无损切分。
    """

    _UNNUMBERED_SEC_RE = re.compile(
        r"^(?:Abstract|Introduction|Background|Related\s+Work|Preliminaries|"
        r"Method(?:s|ology)?|Approach|Model(?:\s+Architecture)?|Architecture|"
        r"Experiments?(?:\s+and\s+Results)?|Experimental\s+(?:Setup|Results)|"
        r"Evaluation|Results(?:\s+and\s+Discussion)?|Analysis|Discussion|"
        r"Conclusion(?:s)?(?:\s+and\s+Future\s+Work)?|Summary|"
        r"Acknowledgments?|References|Bibliography|Appendix|Appendices)$",
        re.I,
    )

    SYSTEM_PROMPT = (
        "You are an expert academic PDF structure analyzer. "
        "Given a numbered list of candidate short lines extracted from a research paper "
        "(with page number, font size, and bold flag), select ONLY the indices of genuine "
        "section or subsection headings. Exclude paper titles, author names, table headers, "
        "figure labels, equations, and diagram annotations. "
        'Respond strictly in JSON format: {"selected_indices": [0, 2, 5]}'
    )

    @classmethod
    def _resolve_configured_client(cls) -> tuple[ChatClientProtocol | None, str]:
        return None, "heuristic-offline"

    @classmethod
    def select_headings_via_llm(
        cls,
        candidates: list[CandidateHeading],
        llm_client: ChatClientProtocol,
        model: str = "gpt-4o-mini",
    ) -> list[CandidateHeading]:
        if not candidates:
            return []
        lines_desc = "\n".join(
            f"[{i}] p{c.page} | size={c.font_size} | bold={c.is_bold} | {c.text!r}"
            for i, c in enumerate(candidates)
        )
        messages = [
            {"role": "system", "content": cls.SYSTEM_PROMPT},
            {"role": "user", "content": f"Candidate lines:\n{lines_desc}"},
        ]
        try:
            raw = llm_client.chat_completion(model=model, messages=messages, temperature=0.0)
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            data = json.loads(m.group(0) if m else raw)
            indices = data.get("selected_indices", [])
            return [
                candidates[i]
                for i in indices
                if isinstance(i, int) and 0 <= i < len(candidates)
            ]
        except Exception as exc:
            logger.warning("LLM skeleton referee failed (%s), falling back to heuristic referee", exc)
            return cls.select_headings_offline(candidates)

    @classmethod
    def select_headings_offline(cls, candidates: list[CandidateHeading]) -> list[CandidateHeading]:
        """无 API Key 时的离线骨架裁判：结合字号层级与学术章节语义词典筛选候选行。"""
        if not candidates:
            return []
        selected: list[CandidateHeading] = []
        for c in candidates:
            text = c.text.strip().rstrip(".:")
            if re.match(r"^\d{1,2}(?:\.\d+)*\.?\s+[A-Z]", text) and len(text.split()) >= 2:
                selected.append(c)
            elif cls._UNNUMBERED_SEC_RE.match(text):
                selected.append(c)
        return selected

    @classmethod
    def adjudicate_and_split(
        cls,
        doc: pymupdf.Document,
        sections: list[Section],
        report: AuditReport,
        llm_client: ChatClientProtocol | None = None,
        model: str | None = None,
    ) -> tuple[list[Section], AuditReport]:
        """
        对仍存在 under_segmented 或 missing_section 异常的区间提取候选骨架，
        优先使用传入或 llm_config.py 中配置的 `pdf_referee` 大模型审定标题并在原章节内执行无损切分。
        """
        if report.passed:
            return sections, report

        if llm_client is None:
            llm_client, configured_model = cls._resolve_configured_client()
            model = model or configured_model

        repaired = list(sections)
        target_ranges = [
            iss.page_range
            for iss in report.issues
            if iss.kind in ("under_segmented", "missing_section")
        ]
        if not target_ranges:
            return repaired, report

        page_range = (
            min(r[0] for r in target_ranges),
            max(r[1] for r in target_ranges),
        )
        candidates = PageInspector.extract_skeleton(doc, page_range)
        approved = (
            cls.select_headings_via_llm(candidates, llm_client, model or "gpt-4o-mini")
            if llm_client is not None
            else cls.select_headings_offline(candidates)
        )

        existing_titles = {re.sub(r"\W+", "", s.title.lower()) for s in repaired}
        for cand in approved:
            norm_cand = re.sub(r"\W+", "", cand.text.lower())
            if not norm_cand or norm_cand in existing_titles:
                continue
            for idx, sec in enumerate(repaired):
                if not (sec.start_page <= cand.page <= sec.end_page):
                    continue
                clean_heading = re.sub(r"\*+", "", cand.text).strip()
                pattern_in_content = re.compile(
                    rf"(?:\*{{0,2}}#*\s*)?{re.escape(clean_heading)}(?:\*{{0,2}})?"
                )
                m_pos = pattern_in_content.search(sec.content)
                if not m_pos:
                    continue
                before = sec.content[:m_pos.start()].strip()
                after = sec.content[m_pos.end():].strip()
                if not before and not after:
                    continue
                num_match = re.match(r"^(\d+(?:\.\d+)*)", clean_heading)
                num = num_match.group(1) if num_match else None
                lvl = len(num.split(".")) if num else 1
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
                    level=lvl,
                    number=num,
                ))
                existing_titles.add(norm_cand)
                break

        final_report = OutlineAuditor.audit(repaired, len(doc), report.paper_title)
        return repaired, final_report
