from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from models.claim import AtomicClaimResponse, SourceSpan


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)      # 硬错误：整条 parent 判失败
    warnings: List[str] = field(default_factory=list)    # 软问题：可自动修复或可容忍
    repaired_spans: int = 0


class ClaimValidator:
    """
    确定性事实与契约校验器（纯离线代码，0 LLM）
    逐条严格实施 10 项校验与容错修复规则
    """

    _CITATION_MARKER_RE = re.compile(r"\[\s*\d+(?:[\s,\-–—]+\d+)*\s*\]")
    _NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")
    _WORD_NUMBER_MAP: Dict[str, str] = {
        "zero": "0",
        "one": "1",
        "single": "1",
        "two": "2",
        "pair": "2",
        "both": "2",
        "twice": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
    }

    def validate_and_repair(
        self,
        paragraph: str,
        expected_keys: List[str],
        response: AtomicClaimResponse,
    ) -> ValidationResult:
        errors: List[str] = []
        warnings: List[str] = []
        repaired_spans: int = 0

        expected_set = set(expected_keys)
        all_cited_keys: Set[str] = set()

        # 规则 8: claims 为空而 expected_keys 非空
        if not response.claims and expected_keys:
            errors.append("未产出任何主张")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, repaired_spans=0)

        # 提取原文中合法的数字集合（规则 7 先置准备：抹掉引用标记 + 千分位逗号归一 + 原文英文数词映射）
        clean_paragraph = self._CITATION_MARKER_RE.sub(" ", paragraph)
        raw_para_nums = set(self._NUMBER_RE.findall(clean_paragraph))
        para_numbers: Set[str] = set(raw_para_nums)
        for n in raw_para_nums:
            para_numbers.add(n.replace(",", ""))
        lower_words = set(re.findall(r"[a-z]+", clean_paragraph.lower()))
        for word, digit in self._WORD_NUMBER_MAP.items():
            if word in lower_words:
                para_numbers.add(digit)

        # 遍历每条 claim 进行深度校验
        for idx, claim in enumerate(response.claims, start=1):
            claim_keys_set = set(claim.citation_keys)

            # 规则 1: citation_keys 必须 ⊆ expected_keys
            invalid_keys = claim_keys_set - expected_set
            if invalid_keys:
                errors.append(f"第 {idx} 条主张新增不存在的引用: {sorted(list(invalid_keys))}")

            # 规则 9: 某条 claim 的 citation_keys 为空
            if not claim.citation_keys:
                warnings.append(f"第 {idx} 条主张未绑定任何引用")

            all_cited_keys.update(claim.citation_keys)

            # 规则 4 & 5: source_spans 检查与就地修正（支持同名短语就近吸附与空白折叠修正）
            for span_idx, span in enumerate(claim.source_spans, start=1):
                if not span.text or not span.text.strip():
                    errors.append(f"第 {idx} 条主张的原文片段不存在于段落中: \"{span.text}\"")
                    continue

                if 0 <= span.start < span.end <= len(paragraph) and paragraph[span.start : span.end] == span.text:
                    continue

                nearest_pos = self._find_nearest(paragraph, span.text, span.start, claim.citation_keys)
                if nearest_pos != -1:
                    span.start = nearest_pos
                    span.end = nearest_pos + len(span.text)
                    repaired_spans += 1
                    warnings.append(
                        f"第 {idx} 条主张的第 {span_idx} 个片段下标偏差已就地修正为 [{span.start}:{span.end}]"
                    )
                else:
                    matched_str = self._match_whitespace_collapsed(paragraph, span.text)
                    if matched_str is not None:
                        nearest_pos = self._find_nearest(paragraph, matched_str, span.start, claim.citation_keys)
                        span.text = matched_str
                        span.start = nearest_pos
                        span.end = nearest_pos + len(matched_str)
                        repaired_spans += 1
                        warnings.append(
                            f"第 {idx} 条主张的第 {span_idx} 个片段存在空白/下标偏差，已就地修正为 [{span.start}:{span.end}]"
                        )
                    else:
                        errors.append(f"第 {idx} 条主张的原文片段不存在于段落中: \"{span.text}\"")

            # 规则 7: 数字保真（抹掉两边引用编号后比对数字，兼容千分位逗号）
            clean_norm = self._CITATION_MARKER_RE.sub(" ", claim.normalized_claim)
            spans_text = " ".join(s.text for s in claim.source_spans)
            clean_spans = self._CITATION_MARKER_RE.sub(" ", spans_text)
            claim_numbers = set(self._NUMBER_RE.findall(clean_norm)) | set(self._NUMBER_RE.findall(clean_spans))

            extra_numbers = [
                n for n in claim_numbers
                if n not in para_numbers and n.replace(",", "") not in para_numbers
            ]
            if extra_numbers:
                errors.append(f"第 {idx} 条主张出现原文没有的数字: {sorted(extra_numbers)}")

            # 规则 10: requires_review=True 但 review_reason 为空
            if claim.requires_review and not claim.review_reason.strip():
                claim.review_reason = "unspecified"
                warnings.append(f"第 {idx} 条主张 requires_review 为 True 但原因缺失，已就地修正为 unspecified")

        # 规则 2: expected_keys 中遗漏且未在 unassigned 中的键
        unassigned_set = set(response.unassigned_citation_keys)
        missing_keys = expected_set - all_cited_keys - unassigned_set
        if missing_keys:
            errors.append(f"遗漏引用: {sorted(list(missing_keys))}")

        # 规则 3: expected_keys 中出现在 unassigned_citation_keys 里的键
        valid_unassigned = expected_set & unassigned_set
        if valid_unassigned:
            warnings.append(f"未归属引用: {sorted(list(valid_unassigned))}")

        return ValidationResult(
            valid=(len(errors) == 0),
            errors=errors,
            warnings=warnings,
            repaired_spans=repaired_spans,
        )

    @staticmethod
    def _find_nearest(
        paragraph: str,
        sub: str,
        hint_start: int,
        citation_keys: Optional[List[str]] = None,
    ) -> int:
        """
        在 paragraph 中查找 sub 的所有出现位置，返回最佳起始下标；未找到返回 -1。
        1. 若 hint_start > 0，优先吸附到离 hint_start 最近的位置；
        2. 若 hint_start == 0（模型占位符）且提供了 citation_keys，以该引用在原文中的物理坐标为锚点进行就近吸附（优先选取位于引用标记左侧最近的短语）；
        3. 否则默认选取第一个匹配位置。
        """
        if not sub:
            return -1
        positions: List[int] = []
        start = 0
        while True:
            pos = paragraph.find(sub, start)
            if pos == -1:
                break
            positions.append(pos)
            start = pos + 1
        if not positions:
            return -1

        # 若大模型给出了有意义的非 0 下标，优先使用 hint_start
        if hint_start > 0:
            return min(positions, key=lambda p: abs(p - hint_start))

        # 若 hint_start == 0 且存在 citation_keys，则以引用标记物理坐标为锚点进行近邻匹配
        cite_anchor: Optional[int] = None
        if citation_keys:
            for k in citation_keys:
                kpos = paragraph.find(k)
                if kpos != -1:
                    cite_anchor = kpos
                    break

        if cite_anchor is not None:
            # 学术主张通常位于引用标记前侧（左侧），优先选取 cite_anchor 左侧最近的匹配项
            left_positions = [p for p in positions if p <= cite_anchor]
            if left_positions:
                return max(left_positions)
            # 若全部在右侧，选取绝对距离最近的位置
            return min(positions, key=lambda p: abs(p - cite_anchor))

        return positions[0]

    def _match_whitespace_collapsed(self, paragraph: str, target: str) -> Optional[str]:
        words = target.split()
        if not words:
            return None
        pattern = r"\s+".join(re.escape(w) for w in words)
        m = re.search(pattern, paragraph)
        if m:
            return m.group(0)
        return None
