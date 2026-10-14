"""标题质量检测与可靠标题回退机制测试 (Step 10)

核心断言：
1. evaluate_title_quality 必须准确检出所有已知字母碎裂与排版乱码标题：
   [2]  N M T J L A T
   [17] N GPU L A
   [19] S A N
   [20] A : A M S O
   [21] F LSTM
   [22] A S S - S E
   [23] M - S S L
   [28] A D R M A S
   [32] O L N N : T S -G M - -E L
2. 正常学术论文标题绝不得被误判为碎裂 (False Positive 防护)
3. 当 PDF 抽取标题碎裂时，resolve_best_title 自动高置信度回退至已校验的 expected_title
4. OutlineAuditor 严密拦截未经修复的碎裂标题，防止带病过审
5. PDFProcessor 落地产物中完整保留 paper_title、extracted_pdf_title、title_source 与 title_confidence
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from citetrace_pdf.pdf.title_evaluator import (
    evaluate_title_quality,
    resolve_best_title,
)
from citetrace_pdf.parser.auditor import OutlineAuditor
from citetrace_pdf.model.document import Section
from citetrace_pdf import PDFProcessor


class TestTitleQualityAndFallback(unittest.TestCase):
    """测试标题质量评估、碎裂检测与元数据回退机制"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache_dir = self.root / "papers"
        self.cache_dir.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_detects_all_known_broken_titles(self):
        """精准检出 9 个典型 Small-caps / 字母碎裂与控制字符乱码标题"""
        broken_titles = [
            "N M T J L A T",
            "N GPU L A",
            "S A N",
            "A : A M S O",
            "F LSTM",
            "A S S - S E",
            "M - S S L",
            "A D R M A S",
            "O L N N : T S -G M - -E L",
        ]
        for title in broken_titles:
            res = evaluate_title_quality(title)
            self.assertTrue(
                res.is_broken,
                f"碎裂标题未能被检出: '{title}', score={res.quality_score}, reasons={res.reasons}"
            )
            self.assertLessEqual(res.quality_score, 0.45)

    def test_clean_titles_are_not_flagged_broken(self):
        """正常论文标题绝不得被误报为碎裂 (False Positive 保护)"""
        clean_titles = [
            "Attention Is All You Need",
            "Deep Residual Learning for Image Recognition",
            "A Fast Learning Algorithm for Deep Belief Nets",
            "Sequence to Sequence Learning with Neural Networks",
            "Q-learning",
            "Adam: A Method for Stochastic Optimization",
            "Mastering the game of Go with deep neural networks and tree search",
        ]
        for title in clean_titles:
            res = evaluate_title_quality(title)
            self.assertFalse(
                res.is_broken,
                f"正常标题被误判为碎裂: '{title}', score={res.quality_score}, reasons={res.reasons}"
            )
            self.assertGreaterEqual(res.quality_score, 0.65)

    def test_resolve_best_title_fallback_on_broken_extracted_title(self):
        """PDF 抽取为碎裂标题时，自动回退至已校验的 expected_title 并保留原始抽取值"""
        res = resolve_best_title(
            extracted_title="N GPU L A",
            expected_title="Neural GPUs Learn Algorithms",
            meta_title=None,
        )
        self.assertEqual(res["paper_title"], "Neural GPUs Learn Algorithms")
        self.assertEqual(res["extracted_pdf_title"], "N GPU L A")
        self.assertEqual(res["title_source"], "reference_metadata")
        self.assertGreaterEqual(res["title_confidence"], 0.95)
        self.assertTrue(res["title_quality"]["is_broken"])

    def test_outline_auditor_flags_broken_title_if_not_resolved(self):
        """OutlineAuditor 必须拦截未经回退修复的碎裂主标题"""
        dummy_sec = Section(
            title="1. Introduction",
            number="1",
            start_page=1,
            end_page=2,
            content="Text content.",
        )
        report_broken = OutlineAuditor.audit(
            sections=[dummy_sec],
            total_pages=2,
            paper_title="N GPU L A",
        )
        self.assertFalse(report_broken.passed)
        issue_msgs = [i.message for i in report_broken.issues]
        self.assertTrue(any("异常、碎裂或缺失" in msg for msg in issue_msgs))

        # 若已修复为规范标题，则主标题审计通过
        report_clean = OutlineAuditor.audit(
            sections=[dummy_sec],
            total_pages=2,
            paper_title="Neural GPUs Learn Algorithms",
        )
        self.assertTrue(report_clean.passed)

    @patch("citetrace_pdf.processor.PDFLoader.load")
    @patch("citetrace_pdf.processor.PDFExtractor.extract")
    def test_pdf_processor_persists_title_provenance_metadata(self, mock_extract, mock_load):
        """PDFProcessor 写入 metadata.json 时必须包含完整的标题源与置信度追溯字段"""
        target_dir = self.cache_dir / "[17]"
        target_dir.mkdir(parents=True, exist_ok=True)
        fake_pdf = target_dir / "raw.pdf"
        fake_pdf.write_bytes(
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
            + b"A" * 1024
            + b"\nxref\n0 4\n0000000000 65535 f\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n100\n%%EOF\n"
        )
        mock_load.return_value = fake_pdf

        dummy_sec = Section(
            title="1. Introduction",
            number="1",
            start_page=1,
            end_page=1,
            content="Text content.",
        )
        mock_extract.return_value = ("N GPU L A", 1, [])

        processor = PDFProcessor(cache_dir=self.cache_dir)
        with patch.object(processor.section_parser, "parse", return_value=[dummy_sec]):
            meta = processor.process_paper(
                paper_id="[17]",
                pdf_url=str(fake_pdf),
                force=True,
                expected_title="Neural GPUs Learn Algorithms",
            )

        self.assertEqual(meta["paper_title"], "Neural GPUs Learn Algorithms")
        self.assertEqual(meta["extracted_pdf_title"], "N GPU L A")
        self.assertEqual(meta["title_source"], "reference_metadata")
        self.assertGreaterEqual(meta["title_confidence"], 0.95)
        self.assertTrue(meta["title_quality"]["is_broken"])

        # 验证落地 metadata.json 包含该字段
        saved_meta = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
        self.assertEqual(saved_meta["paper_title"], "Neural GPUs Learn Algorithms")
        self.assertEqual(saved_meta["extracted_pdf_title"], "N GPU L A")
        self.assertEqual(saved_meta["title_source"], "reference_metadata")


if __name__ == "__main__":
    unittest.main()
