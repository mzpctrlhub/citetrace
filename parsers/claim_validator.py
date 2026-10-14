from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional

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

        # 提取原文中合法的数字集合（规则 7 先置准备：抹掉引用标记）
        clean_paragraph = self._CITATION_MARKER_RE.sub(" ", paragraph)
        para_numbers = set(self._NUMBER_RE.findall(clean_paragraph))

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

            # 规则 4 & 5: source_spans 检查与就地修正
            for span_idx, span in enumerate(claim.source_spans, start=1):
                text_pos = paragraph.find(span.text)
                if text_pos == -1:
                    errors.append(f"第 {idx} 条主张的原文片段不存在于段落中: \"{span.text}\"")
                else:
                    if paragraph[span.start : span.end] != span.text:
                        span.start = text_pos
                        span.end = text_pos + len(span.text)
                        repaired_spans += 1
                        warnings.append(f"第 {idx} 条主张的第 {span_idx} 个片段下标偏差已就地修正为 [{span.start}:{span.end}]")

            # 规则 6: verbatim_claim 必须是 paragraph 子串
            vb_pos = paragraph.find(claim.verbatim_claim)
            if vb_pos != -1:
                # 定位正常
                pass
            else:
                # 尝试空白折叠修正
                matched_str = self._match_whitespace_collapsed(paragraph, claim.verbatim_claim)
                if matched_str is not None:
                    claim.verbatim_claim = matched_str
                    warnings.append(f"第 {idx} 条主张 verbatim_claim 存在空白差异，已就地修正为原文真实子串")
                else:
                    errors.append(f"第 {idx} 条主张的 verbatim_claim 不存在于段落中: \"{claim.verbatim_claim}\"")

            # 规则 7: 数字保真（抹掉两边引用编号后比对数字）
            clean_norm = self._CITATION_MARKER_RE.sub(" ", claim.normalized_claim)
            clean_vb = self._CITATION_MARKER_RE.sub(" ", claim.verbatim_claim)
            claim_numbers = set(self._NUMBER_RE.findall(clean_norm)) | set(self._NUMBER_RE.findall(clean_vb))
            
            extra_numbers = claim_numbers - para_numbers
            if extra_numbers:
                errors.append(f"第 {idx} 条主张出现原文没有的数字: {sorted(list(extra_numbers))}")

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

    def _match_whitespace_collapsed(self, paragraph: str, target: str) -> Optional[str]:
        words = target.split()
        if not words:
            return None
        pattern = r"\s+".join(re.escape(w) for w in words)
        m = re.search(pattern, paragraph)
        if m:
            return m.group(0)
        return None
