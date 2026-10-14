import unittest
from parsers.claim_validator import ClaimValidator
from models.claim import AtomicClaimResponse, AtomicClaimDraft, SourceSpan
from models.verification import ClaimIntent


class TestClaimValidator(unittest.TestCase):
    def setUp(self):
        self.validator = ClaimValidator()
        self.paragraph = (
            "Recurrent models [13] have achieved high scores of 24.6 on translation tasks [7]."
        )
        self.expected_keys = ["[13]", "[7]"]

    def test_1_valid_passing(self):
        """正常通过：一段含 2 个引用标记的段落，键与 span 全部正确"""
        span1_text = "Recurrent models [13]"
        start1 = self.paragraph.find(span1_text)
        end1 = start1 + len(span1_text)

        span2_text = "high scores of 24.6 on translation tasks [7]."
        start2 = self.paragraph.find(span2_text)
        end2 = start2 + len(span2_text)

        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=start1, end=end1, text=span1_text)],
                    verbatim_claim=span1_text,
                    normalized_claim="Recurrent models [13] are sequence models.",
                    citation_keys=["[13]"],
                    relationship=ClaimIntent.BACKGROUND,
                ),
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=start2, end=end2, text=span2_text)],
                    verbatim_claim=span2_text,
                    normalized_claim="The model achieves 24.6 score on translation [7].",
                    citation_keys=["[7]"],
                    relationship=ClaimIntent.METRIC_COMPARISON,
                ),
            ],
            unassigned_citation_keys=[],
        )

        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertTrue(res.valid)
        self.assertEqual(len(res.errors), 0)

    def test_2_hallucinated_reference(self):
        """编造引用：claim 里出现 expected_keys 之外的键"""
        resp = AtomicClaimResponse(
            claims=[
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                    verbatim_claim="Recurrent models",
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[99]"],  # 编造 [99]
                    relationship=ClaimIntent.BACKGROUND,
                ),
                AtomicClaimDraft(
                    source_spans=[SourceSpan(start=0, end=16, text="Recurrent models")],
                    verbatim_claim="Recurrent models",
                    normalized_claim="Models work well.",
                    citation_keys=["[7]", "[13]"],
                    relationship=ClaimIntent.BACKGROUND,
                ),
            ]
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
                    verbatim_claim="Recurrent models",
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]"],  # 遗漏了 [7]
                    relationship=ClaimIntent.BACKGROUND,
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
                    verbatim_claim="Recurrent models",
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]"],
                    relationship=ClaimIntent.BACKGROUND,
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
                    verbatim_claim=span_text,
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                )
            ]
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
                    verbatim_claim="Recurrent models",
                    normalized_claim="Recurrent models are used.",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                )
            ]
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
                    verbatim_claim=span_text,
                    normalized_claim="The score was 25.6 instead of 24.6.",  # 25.6 未在原文出现
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.METRIC_COMPARISON,
                )
            ]
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
                    verbatim_claim=span_text,
                    normalized_claim="Recurrent models [13] scored 24.6 [7].",
                    citation_keys=["[13]", "[7]"],
                    relationship=ClaimIntent.BACKGROUND,
                )
            ]
        )
        res = self.validator.validate_and_repair(self.paragraph, self.expected_keys, resp)
        self.assertTrue(res.valid)
        self.assertEqual(len(res.errors), 0)


if __name__ == "__main__":
    unittest.main()
