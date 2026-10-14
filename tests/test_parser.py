import unittest

from citetrace_pdf.parser.heading import HeadingDetector
from citetrace_pdf.parser.reference import ReferenceFormatter
from models.citation import CitationStatus
from parsers import expand_citation_keys, parse_references_section
from tools.academic_fetcher import is_title_match


class TestCiteTraceParser(unittest.TestCase):
    """
    第二阶段核心测试：验证论文正文解析、参考文献抽样与论断-引用对齐
    """

    def test_expand_citation_keys(self):
        """测试复杂引用展开逻辑"""
        self.assertEqual(expand_citation_keys("1"), ["[1]"])
        self.assertEqual(expand_citation_keys("1, 2"), ["[1]", "[2]"])
        self.assertEqual(expand_citation_keys("1-3"), ["[1]", "[2]", "[3]"])
        self.assertEqual(expand_citation_keys("1, 3-5"), ["[1]", "[3]", "[4]", "[5]"])

    def test_parse_references_section(self):
        """测试参考文献列表的结构化抽取"""
        ref_text = """
        [1] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS. arXiv:1706.03762.
        [2] Wadden, D., et al. (2020). Fact or fiction. EMNLP. DOI: 10.18653/v1/2020.emnlp-main.701.
        """
        citations = parse_references_section(ref_text)
        self.assertIn("[1]", citations)
        self.assertIn("[2]", citations)

        ref1 = citations["[1]"]
        self.assertEqual(ref1.year, 2017)
        self.assertEqual(ref1.arxiv_id, "1706.03762")
        self.assertEqual(ref1.status, CitationStatus.RESOLVED)

        ref2 = citations["[2]"]
        self.assertEqual(ref2.year, 2020)
        self.assertEqual(ref2.doi, "10.18653/v1/2020.emnlp-main.701")

    def test_is_title_match_sequence_awareness(self):
        """测试标题比对具备词序感知，防止类似 LSTM 与老年学百科词条颠倒词序误匹配"""
        # 真实 LSTM vs 颠倒的老年医学百科词条
        self.assertFalse(
            is_title_match("Long Short-Term Memory", "Short-Term/Long-Term Memory"),
            "应当拦截词序颠倒的非相关论文"
        )
        # 真实 LSTM vs 真实 LSTM
        self.assertTrue(
            is_title_match("Long Short-Term Memory", "Long Short-Term Memory")
        )
        self.assertTrue(
            is_title_match("Attention Is All You Need", "Attention is all you need")
        )

    def test_heading_detector_filters_citation_noise_and_years(self):
        """测试章节标题检测器坚决拦截文献出处噪声、4位年份行及非法主章节号"""
        detector = HeadingDetector()
        
        # 1. AlexNet 中跨列被切分的文献出处行，绝对不能被当成章节标题
        self.assertIsNone(
            detector.detect("2004 IEEE Computer Society Conference on, volume 2, pages II–97. IEEE, 2004")
        )
        self.assertIsNone(
            detector.detect("2010 IEEE International Symposium on, pages 253–256. IEEE, 2010")
        )
        
        # 2. 超出正常章节范围的编号（如 2004, 35）
        self.assertIsNone(detector.detect("2004 Some Title"))
        self.assertIsNone(detector.detect("55 Some Title"))
        
        # 3. 合法的正常章节标题应被正确识别
        h1 = detector.detect("1 Introduction")
        self.assertIsNotNone(h1)
        self.assertEqual(h1.number, "1")
        self.assertEqual(h1.kind, "numbered")

        h2 = detector.detect("3.1 ReLU Nonlinearity")
        self.assertIsNotNone(h2)
        self.assertEqual(h2.number, "3.1")
        self.assertEqual(h2.kind, "numbered")

    def test_heading_detector_detects_unnumbered_appendices(self):
        """测试标题检测器能准确识别 Attention Visualizations, Checklist 等无编号附录标题"""
        detector = HeadingDetector()

        for title in [
            "Attention Visualizations",
            "Attention Visualization",
            "Broader Impact",
            "Broader Impacts Statement",
            "Limitations",
            "Ethical Considerations",
            "Checklist",
            "NeurIPS Checklist",
            "Reproducibility Checklist",
            "Appendix",
            "Appendices",
            "Appendix A",
        ]:
            head = detector.detect(title)
            self.assertIsNotNone(head, f"未能识别附录标题: {title}")
            self.assertEqual(head.kind, "appendix")

    def test_reference_formatter_strips_trailing_appendix_garbage(self):
        """测试参考文献格式化器能剔除误粘连在最后一条文献末尾的附录/图表题注/表格"""
        raw_ref = (
            "[39] Jie Zhou. Deep recurrent models. CoRR, 2016.\n\n"
            "[40] Muhua Zhu. Fast parsing. ACL, August 2013. "
            "Attention Visualizations governments registration American majority passed making voting process difficult "
            "Figure 3: An example of the attention mechanism. | --- | --- |"
        )
        formatted = ReferenceFormatter.format(raw_ref)
        self.assertIn("[39] Jie Zhou. Deep recurrent models. CoRR, 2016.", formatted)
        self.assertIn("[40] Muhua Zhu. Fast parsing. ACL, August 2013.", formatted)
        self.assertNotIn("Figure 3:", formatted)
        self.assertNotIn("Attention Visualizations", formatted)
        self.assertNotIn("| --- |", formatted)



    def test_heading_detector_detects_markdown_decorated_headings(self):
        """测试标题检测器能识别带有 Markdown 标签（如 #, ##, **）的深层版面标题"""
        detector = HeadingDetector()

        h1 = detector.detect("## **1 Introduction**")
        self.assertIsNotNone(h1)
        self.assertEqual(h1.text, "1 Introduction")
        self.assertEqual(h1.number, "1")

        h2 = detector.detect("# **7 Conclusion**")
        self.assertIsNotNone(h2)
        self.assertEqual(h2.text, "7 Conclusion")
        self.assertEqual(h2.number, "7")

        h3 = detector.detect("### 3.2.1 Scaled Dot-Product Attention")
        self.assertIsNotNone(h3)
        self.assertEqual(h3.text, "3.2.1 Scaled Dot-Product Attention")
        self.assertEqual(h3.level, 3)

        h_app = detector.detect("# **Attention Visualizations**")
        self.assertIsNotNone(h_app)
        self.assertEqual(h_app.kind, "appendix")

    def test_reference_formatter_strips_markdown_list_dashes(self):
        """测试参考文献格式化器能剔除 Markdown 列表前缀或尾部残留连字符"""
        raw_ref = (
            "- [1] Jimmy Lei Ba. Layer normalization. arXiv, 2016. -\n\n"
            "- [2] Dzmitry Bahdanau. Neural machine translation. CoRR, 2014. -"
        )
        formatted = ReferenceFormatter.format(raw_ref)
        self.assertIn("[1] Jimmy Lei Ba. Layer normalization. arXiv, 2016.", formatted)
        self.assertIn("[2] Dzmitry Bahdanau. Neural machine translation. CoRR, 2014.", formatted)
        self.assertNotIn("- [1]", formatted)
        for line in formatted.splitlines():
            if line.strip():
                self.assertFalse(line.strip().endswith("-"), f"条目尾部残留破折号: {line}")


if __name__ == "__main__":
    unittest.main()
