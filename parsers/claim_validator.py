# ==============================================================================
# 模块名称: parsers/claim_validator.py
# 模块职责: 阶段 1 确定性事实与契约校验器（纯离线 Python 算法，0 LLM 开销）
# 核心功能: 严格执行 10 项不可逾越的事实防伪规则，并提供坐标近邻吸附与空白折叠就地自愈
# ==============================================================================

from __future__ import annotations                     # 启用类型注解延迟求值（支持在方法中使用当前类类型注解）

import re                                              # 导入正则表达式模块，用于文本清洗、符号剥离与模式查找
from dataclasses import dataclass, field               # 导入轻量级数据类装饰器与字段配置器
from typing import Dict, List, Optional, Set           # 导入常用泛型类型注解

from models.claim import AtomicClaimResponse, SourceSpan  # 导入顶级响应包与切片模型


@dataclass
class ValidationResult:
    """
    【校验与修复综合判定报告】
    记录单次校验是否成功，以及在此过程中捕获的硬错误、软告警与自动修补次数。
    """
    valid: bool                                        # 综合判定：True 表示完全合规可入库；False 表示存在硬错误
    errors: List[str] = field(default_factory=list)    # 硬错误列表：哪怕只有 1 条，整批主张判定失败并触发反思重试
    warnings: List[str] = field(default_factory=list)  # 软告警列表：已被自动修复或学术上有歧义但不致命的问题
    repaired_spans: int = 0                            # 本轮校验中被代码在内存中成功就地修复坐标的切片总数


class ClaimValidator:
    """
    确定性事实与契约校验器（纯离线代码，0 LLM）
    逐条严格实施 10 项校验与容错修复规则
    """

    # 正则表达式：匹配各种形式的引用方括号标记，如 [13], [35, 2, 5], [1-3]
    _CITATION_MARKER_RE = re.compile(r"\[\s*\d+(?:[\s,\-–—]+\d+)*\s*\]")

    # 正则表达式：提取纯数字或带千分位/小数点的数值，如 24.6, 100,000, 42
    _NUMBER_RE = re.compile(r"\b\d+(?:[.,]\d+)*\b")

    # 常见英文基数词与数字的等价映射表（防止大模型把原文 two 改写为 2 时被误报为数字篡改）
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
        """
        核心校验流水线：
        :param paragraph: PDF 解析出的原始段落文本
        :param expected_keys: 本段落中出现的期望引用文献列表（全集）
        :param response: 大模型返回的结构化草稿包（将在内存中被就地修补）
        :return: 包含校验状态与错误详情的 ValidationResult 对象
        """
        errors: List[str] = []                         # 初始化本段落的硬错误收集器
        warnings: List[str] = []                       # 初始化本段落的软告警收集器
        repaired_spans: int = 0                        # 初始化坐标修复计数器

        expected_set = set(expected_keys)              # 将期望引用转为哈希集合，用于 O(1) 集合论比对
        all_cited_keys: Set[str] = set()               # 记录本段落所有 claim 实际分配到的引用键并集

        # 【规则 8: 产出为空拦截】原文里明明有引用，大模型却交了白卷（claims 为空）
        if not response.claims and expected_keys:
            errors.append("未产出任何主张")
            return ValidationResult(valid=False, errors=errors, warnings=warnings, repaired_spans=0)

        # ==============================================================================
        # 【规则 7 先置准备：构建原文合法数字集合】
        # 为防止大模型篡改实验数据，必须比对主张中的数字是否全来自原文。
        # 处理流程：抹掉引用编号 -> 提取纯数字 -> 兼容千分位逗号 -> 映射英文数词
        # ==============================================================================
        clean_paragraph = self._CITATION_MARKER_RE.sub(" ", paragraph)  # 抹去引用编号（如把 [13] 换成空格，防止 13 干扰实验数字）
        raw_para_nums = set(self._NUMBER_RE.findall(clean_paragraph))    # 正则提取原文中出现的所有数值字符串
        para_numbers: Set[str] = set(raw_para_nums)                      # 存入哈希集合
        for n in raw_para_nums:
            para_numbers.add(n.replace(",", ""))                         # 兼容千分位：若原文是 100,000，则把 100000 也加入合法集
        lower_words = set(re.findall(r"[a-z]+", clean_paragraph.lower()))# 扫描原文中的小写单词
        for word, digit in self._WORD_NUMBER_MAP.items():
            if word in lower_words:
                para_numbers.add(digit)                                  # 映射数词：若原文含 "two"，则将 "2" 加入合法集

        # ==============================================================================
        # 【主张逐条深潜校验循环】
        # 依次审查大模型交上来的每一条 AtomicClaimDraft
        # ==============================================================================
        for idx, claim in enumerate(response.claims, start=1):
            claim_keys_set = set(claim.citation_keys)                    # 当前主张绑定的引用集合

            # 【规则 1: 引用键超集拦截（防捏造）】
            # claim 中的引用键必须是原文 expected_keys 的子集，严禁编造文献
            invalid_keys = claim_keys_set - expected_set
            if invalid_keys:
                errors.append(f"第 {idx} 条主张新增不存在的引用: {sorted(list(invalid_keys))}")

            # 【规则 9: 孤儿主张告警】
            # 某条主张没有绑定任何文献，虽然不是硬致命错误，但需提示人工复核
            if not claim.citation_keys:
                warnings.append(f"第 {idx} 条主张未绑定任何引用")

            all_cited_keys.update(claim.citation_keys)                   # 将已使用的引用键并入总集合

            # 【规则 4 & 5: 原文片段真实性校验与坐标就地自愈】
            for span_idx, span in enumerate(claim.source_spans, start=1):
                # 判空防呆：严禁空片段
                if not span.text or not span.text.strip():
                    errors.append(f"第 {idx} 条主张的原文片段不存在于段落中: \"{span.text}\"")
                    continue

                # 若模型本身就给出了完美坐标，且 paragraph[start:end] == span.text，直接放行
                if 0 <= span.start < span.end <= len(paragraph) and paragraph[span.start : span.end] == span.text:
                    continue

                # 尝试通过近邻算法（以引用标记为锚点）在原文中就近寻址
                nearest_pos = self._find_nearest(paragraph, span.text, span.start, claim.citation_keys)
                if nearest_pos != -1:
                    # 成功就近寻址：在内存中就地修正 start 和 end 坐标
                    span.start = nearest_pos
                    span.end = nearest_pos + len(span.text)
                    repaired_spans += 1
                    warnings.append(
                        f"第 {idx} 条主张的第 {span_idx} 个片段下标偏差已就地修正为 [{span.start}:{span.end}]"
                    )
                else:
                    # 原文精准匹配失败：尝试空白折叠容错（解决 PDF 断行、多空格排版问题）
                    matched_str = self._match_whitespace_collapsed(paragraph, span.text)
                    if matched_str is not None:
                        # 找到原文真实切片后，重新就近计算绝对坐标并换回原汁原味的字符切片
                        nearest_pos = self._find_nearest(paragraph, matched_str, span.start, claim.citation_keys)
                        span.text = matched_str
                        span.start = nearest_pos
                        span.end = nearest_pos + len(matched_str)
                        repaired_spans += 1
                        warnings.append(
                            f"第 {idx} 条主张的第 {span_idx} 个片段存在空白/下标偏差，已就地修正为 [{span.start}:{span.end}]"
                        )
                    else:
                        # 两种匹配全军覆没：证明大模型私自篡改了原文词句，判死刑！
                        errors.append(f"第 {idx} 条主张的原文片段不存在于段落中: \"{span.text}\"")

            # 【规则 7: 数字保真度硬校验】
            # 抹掉引用标记后，提取 normalized_claim 与各切片中的数字集合
            clean_norm = self._CITATION_MARKER_RE.sub(" ", claim.normalized_claim)
            spans_text = " ".join(s.text for s in claim.source_spans)
            clean_spans = self._CITATION_MARKER_RE.sub(" ", spans_text)
            claim_numbers = set(self._NUMBER_RE.findall(clean_norm)) | set(self._NUMBER_RE.findall(clean_spans))

            # 比对主张里的数字是否全部存在于原文合法数字集中（兼容千分位逗号）
            extra_numbers = [
                n for n in claim_numbers
                if n not in para_numbers and n.replace(",", "") not in para_numbers
            ]
            if extra_numbers:
                errors.append(f"第 {idx} 条主张出现原文没有的数字: {sorted(extra_numbers)}")

            # 【规则 10: 举手求助兜底补齐】
            # 若大模型标记了 requires_review=True 但偷懒留空了原因，自动补全为 "unspecified"
            if claim.requires_review and not claim.review_reason.strip():
                claim.review_reason = "unspecified"
                warnings.append(f"第 {idx} 条主张 requires_review 为 True 但原因缺失，已就地修正为 unspecified")

        # ==============================================================================
        # 【段落级全局引用守恒校验】
        # ==============================================================================
        unassigned_set = set(response.unassigned_citation_keys)          # 未分配引用集合

        # 【规则 2: 引用遗漏拦截】
        # expected_keys 中既没有被任何主张引用、也没写进 unassigned 的，判为丢失
        missing_keys = expected_set - all_cited_keys - unassigned_set
        if missing_keys:
            errors.append(f"遗漏引用: {sorted(list(missing_keys))}")

        # 【规则 3: 合法未归属登记】
        # 记录确实存在于原文中但无法提取主张的合法孤儿引用，计入软告警以供审查
        valid_unassigned = expected_set & unassigned_set
        if valid_unassigned:
            warnings.append(f"未归属引用: {sorted(list(valid_unassigned))}")

        # 返回最终汇总评估对象
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
        【近邻物理寻址算法】
        在 paragraph 中查找 sub 的所有出现位置，返回最佳起始下标；未找到返回 -1。
        寻址策略：
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

        # 若大模型给出了有意义的非 0 下标，优先信任 hint_start
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
        r"""
        【空白折叠自愈算法】
        解决 PDF 提取时的连字符换行、多空格被大模型标准化为单空格的问题。
        通过正则将每个词用 \s+ 重新缝合，在原文中捕获真实的字符切片并返回。
        """
        words = target.split()
        if not words:
            return None
        pattern = r"\s+".join(re.escape(w) for w in words)
        m = re.search(pattern, paragraph)
        if m:
            return m.group(0)
        return None
