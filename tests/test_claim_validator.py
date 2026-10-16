# ==============================================================================
# 模块名称: tests/test_claim_validator.py
# 模块职责: 阶段 1 确定性事实校验器与编排器的离线自动化单元测试套件
# 核心价值: 100% 离线运行（0 Token 消耗、0 网络开销），覆盖 13 种学术防伪与自愈边界场景
# ==============================================================================

import unittest                                        # 导入 Python 标准单元测试框架
from parsers.claim_validator import ClaimValidator     # 导入被测核心组件：事实校验器
from models.claim import AtomicClaimResponse, AtomicClaimDraft, SourceSpan  # 导入测试构造所需的数据模型
from models.verification import ClaimIntent            # 导入引用意图枚举


class TestClaimValidator(unittest.TestCase):
    """
    【校验器核心测试用例集】
    继承 unittest.TestCase，提供 setUp 夹具与断言方法体系
    """

    def setUp(self):
        """
        【测试前置夹具 (Fixture)】
        在每一个 test_* 测试方法执行前都会被自动调用一次，保证测试用例之间状态完全隔离。
        """
        self.validator = ClaimValidator()              # 实例化被测校验器对象
        # 构造一条标准的学术测试段落：包含 2 个引用标记 [13] 和 [7]，以及一个实验数据 24.6
        self.paragraph = (
            "Recurrent models [13] have achieved high scores of 24.6 on translation tasks [7]."
        )
        self.expected_keys = ["[13]", "[7]"]           # 原文包含的期望引用键全集

    def test_1_valid_passing(self):
        """
        【正向黄金用例】：所有引用完全吻合，所有切片与坐标 100% 精确匹配
        预期：valid 为 True，errors 为空列表
        """
        span1_text = "Recurrent models [13]"
        start1 = self.paragraph.find(span1_text)
        end1 = start1 + len(span1_text)

        span2_text = "high scores of 24.6 on translation tasks [7]."
        start2 = self.paragraph.find(span2_text)
        end2 = start2 + len(span2_text)

        # 模拟大模型交出了一份满分作业
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=start1, end=end1, text=span1_text)],
                    normalized_claim="Recurrent models [13] are sequence models.",
                    citation_keys=["[13]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                ),
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=start2, end=end2, text=span2_text)],
                    normalized_claim="The model achieves 24.6 score on translation [7].",
                    citation_keys=["[7]"],
                    relationship=ClaimIntent.METRIC_COMPARISON,
                    requires_review=False,
                    review_reason="",
                ),
            ],
            unassigned_citation_keys=[],
        )

        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertTrue(res.valid)                     # 断言校验必须判定成功
        self.assertEqual(len(res.errors), 0)           # 断言硬错误列表必须为 0

    def test_2_hallucinated_reference(self):
        """编造引用：claim 里出现 expected_keys 之外的键"""
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[99]"],  # 编造 [99]
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                ),
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                    normalized_claim="Models work well.",
                    citation_keys=["[7]", "[13]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                ),
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertFalse(res.valid)
        self.assertTrue(any("新增不存在的引用" in e for e in res.errors))

    def test_3_missing_reference(self):
        """遗漏引用：expected_keys 有一个键既没被引用也没进 unassigned_citation_keys"""
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]"],  # 遗漏了 [7]
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertFalse(res.valid)
        self.assertTrue(any("遗漏引用" in e for e in res.errors))

    def test_4_legitimate_unassigned_reference(self):
        """合法未归属：未被主张引用的键进入了 unassigned_citation_keys"""
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=["[7]"],  # [7] 合法进入未归属列表
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertTrue(res.valid)
        self.assertTrue(any("未归属引用" in w for w in res.warnings))

    def test_5_span_offset_repair(self):
        """下标错但文本对：start/end 故意偏移，就地修正"""
        span_text = "Recurrent models"
        real_start = self.paragraph.find(span_text)
        real_end = real_start + len(span_text)

        # 故意给错误下标 [10:30]
        span = SourceSpan(start=10, end=30, text=span_text)
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[span],
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertTrue(res.valid)
        self.assertGreaterEqual(res.repaired_spans, 1)
        self.assertEqual(span.start, real_start)
        self.assertEqual(span.end, real_end)
        self.assertEqual(self.paragraph[span.start : span.end], span_text)

    def test_6_fabricated_span_text(self):
        """文本捏造：span.text 不在段落中"""
        fake_text = "Completely fabricated text that does not exist"
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=10, text=fake_text)],
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertFalse(res.valid)
        self.assertTrue(any("原文片段不存在于段落中" in e for e in res.errors))

    def test_7_tampered_numbers(self):
        """数字篡改：段落写 24.6，normalized_claim 写 25.6"""
        span_text = "Recurrent models"
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text=span_text)],
                    normalized_claim="The score was 25.6 instead of 24.6.",  # 25.6 未在原文出现
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.METRIC_COMPARISON,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertFalse(res.valid)
        self.assertTrue(any("出现原文没有的数字" in e for e in res.errors))

    def test_8_citation_key_number_not_tampering(self):
        """引用编号里的数字不算篡改：normalized_claim 含 [13]，段落中只有 [13] 没有裸数字 13"""
        span_text = "Recurrent models"
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text=span_text)],
                    normalized_claim="Recurrent models [13] scored 24.6 [7].",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertTrue(res.valid)
        self.assertEqual(len(res.errors), 0)

    def test_9_nearest_span_offset_repair_for_repeated_phrase(self):
        """同名短语在段落中出现多次时，若给定偏移下标，应就近吸附到最近位置"""
        para = "The model [13] improves accuracy. Later, the model [7] reduces training time."
        target_phrase = "the model"
        second_pos = para.rfind(target_phrase)
        span = SourceSpan(start=second_pos + 2, end=second_pos + 2 + len(target_phrase), text=target_phrase)
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[span],
                    normalized_claim="The model [13] and the model [7] are effective.",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(para, ["[13]", "[7]"], resp)
        self.assertTrue(res.valid)
        self.assertEqual(span.start, second_pos)
        self.assertEqual(span.end, second_pos + len(target_phrase))

    def test_10_span_whitespace_collapsed_repair(self):
        """原文含换行或多空格而 span.text 折叠为单空格时，应自动修补为原文真实子串与下标"""
        para = "Recurrent   models\n[13] achieve 24.6 BLEU [7]."
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=20, text="Recurrent models [13]")],
                    normalized_claim="Recurrent models [13] achieve 24.6 BLEU [7].",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.METRIC_COMPARISON,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(para, ["[13]", "[7]"], resp)
        self.assertTrue(res.valid)
        self.assertGreaterEqual(res.repaired_spans, 1)
        span = resp.claims[0].source_spans[0]
        self.assertEqual(para[span.start : span.end], span.text)
        self.assertIn("Recurrent   models\n[13]", span.text)

    def test_11_thousands_comma_and_word_number_equivalence(self):
        """原文含千分位 100,000 或英文数词 two 时，normalized_claim 写 100000 或 2 不得误报数字篡改"""
        para = "We trained for 100,000 steps across two GPUs [13]."
        span_text = "trained for 100,000 steps across two GPUs [13]"
        start = para.find(span_text)
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=start, end=start + len(span_text), text=span_text)],
                    normalized_claim="The system trained for 100000 steps across 2 GPUs [13].",
                    citation_keys=["[13]"],
                    relationship=ClaimIntent.METHOD_USAGE,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=[],
        )
        res = self.validator.validate_and_repair(para, ["[13]"], resp)
        self.assertTrue(res.valid, f"不应误报数字篡改，但报错: {res.errors}")

    def test_12_extractor_preserves_raw_response_and_all_round_errors(self):
        """当多轮校验均失败时，AtomicClaimExtractor 必须保留 raw_response 并合并所有轮次的 errors"""
        from parsers.atomic_claim_extractor import AtomicClaimExtractor

        class _FakeBadLLM:
            def __init__(self):
                self.last_raw_response = '{"claims": [], "unassigned_citation_keys": []}'
                self.calls = 0

            def generate_json(self, **kwargs):
                self.calls += 1
                self.last_raw_response = f'{{"round": {self.calls}, "bad": true}}'
                # 返回遗漏 [7] 的响应触发校验失败
                return AtomicClaimResponse(
                    claims=[
                        AtomicClaimDraft(
                            source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                            normalized_claim="Recurrent models are used.",
                            citation_keys=["[13]"],
                            relationship=ClaimIntent.BACKGROUND,
                            requires_review=False,
                            review_reason="",
                        )
                    ],
                    unassigned_citation_keys=[],
                )

        fake_llm = _FakeBadLLM()
        extractor = AtomicClaimExtractor(
            llm=fake_llm,
            model_name="fake-model",
            validator=self.validator,
            max_attempts=2,
        )
        outcome = extractor.extract({
            "claim_id": "claim_001",
            "context_id": "ctx_001",
            "paragraph": self.paragraph,
            "citation_keys": ["[13]", "[7]"],
        })
        self.assertEqual(outcome.status, "validation_failed")
        self.assertEqual(outcome.attempts, 2)
        self.assertIn("第1轮", outcome.error)
        self.assertIn("第2轮", outcome.error)
        self.assertIn('"round": 2', outcome.raw_response)

    def test_13_repeated_phrase_anchored_by_citation_key_when_placeholder_zero(self):
        """当模型给出占位符下标 start=0 且短语在段落中出现多次时，应通过 citation_key 就近吸附到正确的第二个位置"""
        para = "We evaluate the baseline model [3] on WMT 14. Later, our system beats the baseline model [12] easily."
        target_phrase = "the baseline model"
        first_pos = para.find(target_phrase)
        second_pos = para.rfind(target_phrase)
        self.assertNotEqual(first_pos, second_pos)

        # 大模型抽取 [12] 对应的 claim，但按指令给了占位符 start=0, end=0
        span = SourceSpan(start=0, end=0, text=target_phrase)
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[span],
                    normalized_claim="Our system beats the baseline model [12].",
                    citation_keys=["[12]"],
                    relationship=ClaimIntent.METRIC_COMPARISON,
                    requires_review=False,
                    review_reason="",
                )
            ],
            unassigned_citation_keys=["[3]"],
        )

        res = self.validator.validate_and_repair(para, ["[3]", "[12]"], resp)
        self.assertTrue(res.valid)
        # 必须命中 [12] 附近的 second_pos，绝不能误吸附到 [3] 附近的 first_pos！
        self.assertEqual(span.start, second_pos)
        self.assertEqual(span.end, second_pos + len(target_phrase))
        self.assertEqual(para[span.start : span.end], target_phrase)


if __name__ == "__main__":
    unittest.main()
