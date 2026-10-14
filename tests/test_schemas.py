import unittest
from models import (
    CitationRef,
    CitationStatus,
    CitationVerdict,
    FailureReason,
)


class TestCiteTraceSchemas(unittest.TestCase):
    """验证底层领域数据契约与枚举定义"""

    def test_citation_ref_creation(self):
        ref = CitationRef(
            key="[1]",
            raw_text="Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.",
            title="Attention is all you need",
            year=2017,
        )
        self.assertEqual(ref.key, "[1]")
        self.assertEqual(ref.status, CitationStatus.UNRESOLVED)
        self.assertEqual(ref.year, 2017)
        self.assertEqual(ref.authors, [])
        self.assertIsNone(ref.doi)

    def test_citation_verdict_values(self):
        expected = {
            "correct",
            "data_error",
            "statement_error",
            "insufficient",
            "invalid_citation",
            "search_incomplete",
        }
        self.assertEqual({v.value for v in CitationVerdict}, expected)

    def test_failure_reason_values(self):
        self.assertEqual(FailureReason.QUOTE_NOT_LOCATED.value, "quote_not_located")
        self.assertEqual(FailureReason.DOCUMENT_NOT_FOUND.value, "document_not_found")
        self.assertEqual(FailureReason.PAYWALL_LOCKED.value, "paywall_locked")


if __name__ == "__main__":
    unittest.main()
