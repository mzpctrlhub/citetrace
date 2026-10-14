from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Heading:
    text: str
    level: int = 1
    number: str | None = None
    kind: str = "numbered"


class HeadingDetector:
    """Centralized heading rules; numeric section labels are not capped at 12."""

    NUMBERED = re.compile(r"^(\d+(?:\.\d+)*)[.:]?\s+((?:[A-Z]|\d+[\s\-]+[A-Z]{2,}|\d+[A-Za-z]+\b).+)$")
    LETTERED = re.compile(r"^([A-G](?:\.\d+)*)[.:]?\s+((?:[A-Z][A-Za-z]{2,}|\d+[\s\-]+[A-Z]{2,})[\w\s\-–—:()/'\"?&+]*)$")
    APPENDIX = re.compile(r"^(?:APPENDI(?:X|CES)|Appendi(?:x|ces))(?:\s+([A-G0-9]+(?:\.\d+)*))?(?:[:.]?\s*|\s+)?(.*)$")
    RUN_IN = re.compile(
        r"^(\d+(?:\.\d+)+)[.:]?\s+([A-Z][A-Za-z0-9\s,:\-–—()/'\"?&+]*?[. :–—])\s+(.+)$"
    )
    SECTION_START = re.compile(r"^\d{1,2}(?:\.\d+)*[.:]?\s+[A-Z]")
    NUMBER_ONLY = re.compile(r"^(?:\d+(?:\.\d+)*\.?|[A-G]\.?)$", re.I)
    CITATION_NOISE = re.compile(
        r" (?:et al|arXiv|IEEE|CVPR|ICCV|NeurIPS|Proc\.|vol\.|pp\.|doi:|, and | [12]\d{3} )", re.I
    )
    BIB_NOISE = re.compile(
        r" (?:conference|proceedings|proc\.|vol\.|volume|pages|pp\.|ieee|acm|arxiv|springer|elsevier|journal|transactions|symposium|press|editor|edition) ",
        re.I
    )
    UNNUMBERED_APPENDIX = re.compile(
        r"^(?:(?:Attention\s+)?Visualizations?|"
        r"Broader\s+Impacts?(?:\s+Statement)?|"
        r"Limitations(?:\s+and\s+Societal\s+Impacts?)?|"
        r"Ethical\s+Considerations?|Ethics\s+Statement|"
        r"(?:NeurIPS\s+|Paper\s+|Reproducibility\s+)?Checklist|"
        r"Author\s+Contributions?|Funding\s+Transparency|"
        r"Appendices|Appendix(?:\s+[A-Z0-9]+)?)$",
        re.I
    )

    # 段落中含有引用括号，如 [1]、[39] 等，不可能是章节标题
    _INLINE_CITATION = re.compile(r"\[\d+\]")

    @staticmethod
    def clean_markdown_markers(text: str) -> str:
        """Strip markdown heading hashes (#), list bullets before section ids, bold (**), italic (* or _), and backticks."""
        text = re.sub(r"^#+\s*", "", text.strip())
        text = re.sub(r"\*\*|__|[*_`]", "", text)
        text = re.sub(r"^[-•]\s+(?=(?:\d+(?:\.\d+)*|[A-G](?:\.\d+)*)(?:[.:]?\s+))", "", text.strip())
        return text.strip()

    def detect(self, text: str) -> Heading | None:
        text = self.clean_markdown_markers(text)
        text = re.sub(r"\s+", " ", text.strip().rstrip(".").strip())
        if len(text) > 80:
            return None

        # ── 过滤 1：含内联引用括号（如 [39]），绝非章节标题 ──────────────
        if self._INLINE_CITATION.search(text):
            return None

        # ── 过滤 2：含文献出处/会议期刊等元数据噪声词，绝非章节标题 ──────────
        if self.BIB_NOISE.search(text):
            return None

        if re.fullmatch(r"Abstract", text, re.I):
            return Heading(text, 0, None, "abstract")
        if re.fullmatch(r"References", text, re.I):
            return Heading(text, 0, None, "references")
        if re.fullmatch(r"Acknowledg(?:e)?ments?", text, re.I):
            return Heading(text, 0, None, "acknowledgments")
        if re.fullmatch(r"Supplementary\s+Material(?:s)?", text, re.I):
            return Heading(text, 1, "Supplementary", "appendix")
        if self.UNNUMBERED_APPENDIX.fullmatch(text):
            return Heading(text, 1, text, "appendix")
        appendix = self.APPENDIX.fullmatch(text)
        if appendix and "for details" not in text.lower():
            return Heading(text, 1, appendix.group(1), "appendix")
        lettered = self.LETTERED.fullmatch(text)
        if lettered:
            num = lettered.group(1).upper()
            return Heading(text, len(num.split(".")), num, "appendix")
        numbered = self.NUMBERED.fullmatch(text)
        if numbered:
            number = numbered.group(1)

            # ── 过滤 3：主章节号必须为小正整数（1~30），子节分量必须为（1~99）──
            parts = [int(p) for p in number.split(".")]
            if not (1 <= parts[0] <= 30 and all(1 <= p <= 99 for p in parts[1:])):
                return None
            if len(parts) == 1 and re.match(r"^\d+\s+", numbered.group(2)):
                return None

            return Heading(text, len(parts), number, "numbered")
        return None

    def is_citation_noise(self, line: str) -> bool:
        clean = self.clean_markdown_markers(line)
        if self.SECTION_START.match(clean):
            return False
        return bool(self.CITATION_NOISE.search(clean) or (
            clean.count(".") >= 3 and not re.match(r"^\d+(?:\.\d+)+", clean)
        ))
