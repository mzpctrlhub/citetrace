"""论文标题质量评估与断裂回退机制 (Broken Title Detection & Robust Fallback)

用于识别 PDF 文本抽取中因 Small-Caps、字形编码或 Type-3 字体缺失导致的字母碎裂乱码
（例如 "N GPU L A"、"N M T J L A T"、"S A N"、"A : A M S O"、"A D R M A S" 等），
并在检测到标题碎裂时，安全、高置信度地回退至检索阶梯已校验的规范标题，同时保留原始抽取字段。
"""
from __future__ import annotations

import difflib
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_BOGUS_TITLE_RE = re.compile(r"(?:\.(?:dvi|tex|ps|pdf)$|^untitled|^arxiv:|^\s*$)", re.I)
_COMMON_SINGLE_CHAR_WORDS = {"a", "i"}


@dataclass
class TitleQualityResult:
    is_broken: bool
    quality_score: float
    single_char_ratio: float
    fragmentation_score: float
    meaningful_word_count: int
    total_tokens: int
    reasons: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_broken": self.is_broken,
            "quality_score": round(self.quality_score, 3),
            "single_char_ratio": round(self.single_char_ratio, 3),
            "fragmentation_score": round(self.fragmentation_score, 3),
            "meaningful_word_count": self.meaningful_word_count,
            "total_tokens": self.total_tokens,
            "reasons": self.reasons,
        }


def evaluate_title_quality(title: Optional[str]) -> TitleQualityResult:
    """
    评估标题字符串的质量与完整性，精确诊断字母碎裂与排版乱码。
    """
    if not title or not title.strip():
        return TitleQualityResult(
            is_broken=True,
            quality_score=0.0,
            single_char_ratio=1.0,
            fragmentation_score=1.0,
            meaningful_word_count=0,
            total_tokens=0,
            reasons=["empty_title"],
        )

    clean = title.strip()
    reasons: List[str] = []

    if _BOGUS_TITLE_RE.search(clean):
        return TitleQualityResult(
            is_broken=True,
            quality_score=0.0,
            single_char_ratio=1.0,
            fragmentation_score=1.0,
            meaningful_word_count=0,
            total_tokens=0,
            reasons=["bogus_pattern_match"],
        )

    # 1. 基础分词分析
    tokens = [t for t in re.split(r"[\s\-_:/]+", clean) if t]
    total_tokens = len(tokens)
    if total_tokens == 0:
        return TitleQualityResult(
            is_broken=True,
            quality_score=0.0,
            single_char_ratio=1.0,
            fragmentation_score=1.0,
            meaningful_word_count=0,
            total_tokens=0,
            reasons=["no_tokens"],
        )

    # 单字母单词统计（排除标点后）
    alpha_tokens = [t for t in tokens if any(c.isalpha() for c in t)]
    single_char_tokens = [t for t in tokens if len(t) == 1 and t.isalpha()]
    non_article_single_chars = [
        t for t in single_char_tokens if t.lower() not in _COMMON_SINGLE_CHAR_WORDS
    ]
    meaningful_words = [t for t in tokens if len(t) >= 3 and any(c.isalpha() for c in t)]

    single_char_ratio = len(single_char_tokens) / total_tokens if total_tokens else 0.0
    meaningful_count = len(meaningful_words)
    avg_token_len = sum(len(t) for t in tokens) / total_tokens if total_tokens else 0.0

    # 2. 连续单字母扫描（Small-caps 乱码的典型指纹：字母被空格隔开，如 "N M T J L A T" 或 "... L A"）
    consecutive_single_seqs = re.findall(
        r"(?:^|[\s\-_])([A-Za-z](?:[\s\-_]+[A-Za-z])+)(?:$|[\s\-_])",
        clean
    )
    max_consecutive_single = 0
    for seq in consecutive_single_seqs:
        seq_chars = re.findall(r"[A-Za-z]", seq)
        if len(seq_chars) > max_consecutive_single:
            max_consecutive_single = len(seq_chars)

    # 3. 判定碎裂度 (Fragmentation Score)
    fragmentation_factors = []
    if single_char_ratio >= 0.40:
        fragmentation_factors.append(single_char_ratio)
    if max_consecutive_single >= 2:
        fragmentation_factors.append(min(1.0, max_consecutive_single / 4.0))
    if avg_token_len < 3.0:
        fragmentation_factors.append(max(0.0, 1.0 - (avg_token_len / 4.0)))

    fragmentation_score = sum(fragmentation_factors) / len(fragmentation_factors) if fragmentation_factors else 0.0

    # 4. 判定是否为碎裂/乱码标题
    is_broken = False

    # 场景 A: 单字符比例极高且词数不少于 3（如 N M T J L A T, N GPU L A, S A N, A D R M A S）
    if total_tokens >= 3 and single_char_ratio >= 0.50:
        is_broken = True
        reasons.append("high_single_char_ratio")

    # 场景 B: 存在连续单字符序列且有效实质词极少（如 A S S - S E, M - S S L）
    if max_consecutive_single >= 2 and meaningful_count <= 2:
        is_broken = True
        reasons.append("consecutive_fragmented_letters")

    # 场景 C: 全词均无至少 3 个字母的实质词，且整体长度有限（如 S A N, A : A M S O）
    if meaningful_count == 0 and len(clean) >= 3:
        is_broken = True
        reasons.append("no_meaningful_words")

    # 场景 D: 平均词长极低且含有非冠词单字母（如 F LSTM）
    if total_tokens <= 3 and non_article_single_chars and avg_token_len < 3.2 and meaningful_count <= 1:
        is_broken = True
        reasons.append("short_fragmented_title")

    # 5. 综合质量评分 (Quality Score, 0.0 ~ 1.0)
    if is_broken:
        quality_score = max(0.05, min(0.40, 0.40 - fragmentation_score * 0.35))
    else:
        quality_score = min(1.0, max(0.70, 0.85 + min(0.15, meaningful_count * 0.03) - (single_char_ratio * 0.2)))

    return TitleQualityResult(
        is_broken=is_broken,
        quality_score=quality_score,
        single_char_ratio=single_char_ratio,
        fragmentation_score=fragmentation_score,
        meaningful_word_count=meaningful_count,
        total_tokens=total_tokens,
        reasons=reasons,
    )


def _is_acronym_match(fragmented: str, target: str) -> bool:
    """检查碎裂的字母串是否与目标标题的首字母缩写相吻合"""
    f_chars = "".join(re.findall(r"[A-Za-z]", fragmented)).upper()
    t_words = [w for w in re.split(r"[\s\-_:]+", target) if w]
    t_initials = "".join(w[0] for w in t_words if w and w[0].isalpha()).upper()
    if len(f_chars) >= 3 and len(t_initials) >= 3:
        if f_chars in t_initials or t_initials in f_chars:
            return True
        ratio = difflib.SequenceMatcher(None, f_chars, t_initials).ratio()
        if ratio >= 0.75:
            return True
    return False


def resolve_best_title(
    extracted_title: str,
    expected_title: Optional[str] = None,
    meta_title: Optional[str] = None,
) -> Dict[str, Any]:
    """
    根据 PDF 提取标题与预期参考标题综合判决最佳标题与来源：
    - 若 PDF 提取标题完整清晰，优先使用
    - 若 PDF 提取标题明显碎裂乱码，自动回退至已校验的 expected_title 或合法 doc_metadata
    - 完整保留 extracted_pdf_title、paper_title、title_source 与 title_confidence
    """
    extracted_clean = (extracted_title or "").strip()
    quality = evaluate_title_quality(extracted_clean)

    expected_clean = (expected_title or "").strip() if expected_title else ""
    expected_quality = evaluate_title_quality(expected_clean) if expected_clean else None

    meta_clean = (meta_title or "").strip() if meta_title else ""
    meta_quality = evaluate_title_quality(meta_clean) if meta_clean else None

    # 1. 提取标题完好且质量高
    if not quality.is_broken and quality.quality_score >= 0.65:
        return {
            "paper_title": extracted_clean,
            "extracted_pdf_title": extracted_clean,
            "title_source": "pdf_text",
            "title_confidence": round(quality.quality_score, 2),
            "title_quality": quality.to_dict(),
        }

    # 2. 提取标题碎裂或低质，尝试回退至 expected_title
    if expected_clean and expected_quality and not expected_quality.is_broken:
        confidence = 0.98 if _is_acronym_match(extracted_clean, expected_clean) else 0.95
        return {
            "paper_title": expected_clean,
            "extracted_pdf_title": extracted_clean,
            "title_source": "reference_metadata",
            "title_confidence": confidence,
            "title_quality": quality.to_dict(),
        }

    # 3. 尝试回退至 PDF 内部 document metadata 标题
    if meta_clean and meta_quality and not meta_quality.is_broken:
        return {
            "paper_title": meta_clean,
            "extracted_pdf_title": extracted_clean,
            "title_source": "pdf_metadata",
            "title_confidence": 0.88,
            "title_quality": quality.to_dict(),
        }

    # 4. 无可靠回退源，保留提取值但记录低置信度与未校验标记
    return {
        "paper_title": extracted_clean,
        "extracted_pdf_title": extracted_clean,
        "title_source": "pdf_text_unverified",
        "title_confidence": round(quality.quality_score, 2),
        "title_quality": quality.to_dict(),
    }
