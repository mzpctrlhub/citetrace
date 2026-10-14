import tempfile
import unittest
from pathlib import Path

from citetrace_pdf.parser.reference import ReferenceFormatter
from citetrace_pdf.processor import sanitize_paper_folder_name
from parsers.citation_parser import parse_reference_entry, parse_references_section
from parsers.folder_extractor import (
    _write_citations_directory,
    expand_citation_keys,
    parse_reference_catalog_from_md,
    parse_references_from_md,
)
from parsers.link_extractor import PDFLinkCitationExtractor, normalize_citation_anchor


class TestLinkExtractorAndAuthorYear(unittest.TestCase):
    """
    测试 PDF 链接级高精度引用提取与作者-年份排版支持
    """

    def test_normalize_citation_anchor(self):
        self.assertEqual(normalize_citation_anchor("[35,"), "[35]")
        self.assertEqual(normalize_citation_anchor("2,"), "[2]")
        self.assertEqual(normalize_citation_anchor("5]"), "[5]")
        self.assertEqual(normalize_citation_anchor("[Krizhevsky et al., 2012]"), "[Krizhevsky et al., 2012]")
        self.assertEqual(normalize_citation_anchor("(Hinton et al., 2012)"), "[Hinton et al., 2012]")

    def test_expand_author_year_citation_keys(self):
        # 纯数字
        self.assertEqual(expand_citation_keys("1, 2"), ["[1]", "[2]"])
        self.assertEqual(expand_citation_keys("1-3"), ["[1]", "[2]", "[3]"])

        # 作者-年份复合引用
        compound = "Laurent et al., 2015, Amodei et al., 2015, Cooijmans et al., 2016"
        expanded = expand_citation_keys(compound)
        self.assertEqual(len(expanded), 3)
        self.assertIn("[Laurent et al., 2015]", expanded)
        self.assertIn("[Amodei et al., 2015]", expanded)
        self.assertIn("[Cooijmans et al., 2016]", expanded)

    def test_parse_author_year_reference_entry(self):
        raw = "Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012."
        ref = parse_reference_entry(raw)
        self.assertIsNotNone(ref)
        self.assertEqual(ref.year, 2012)
        self.assertIn("Krizhevsky", ref.key)
        self.assertIn("2012", ref.key)
        self.assertTrue(len(ref.authors) >= 1)

    def test_parse_references_section_author_year(self):
        text = """
        Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

        Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in speech recognition. IEEE, 2012.
        """
        refs = parse_references_section(text)
        self.assertEqual(len(refs), 2)
        keys = list(refs.keys())
        self.assertTrue(any("Krizhevsky" in k for k in keys))
        self.assertTrue(any("Hinton" in k for k in keys))

    def test_pdf_link_extractor_on_layernorm_if_exists(self):
        candidate_paths = [
            Path("/root/citetrace/cache/1607.06450.pdf"),
            Path("cache/1607.06450.pdf"),
            Path("d:/langgraph/cache/1607.06450.pdf")
        ]
        pdf_path = next((p for p in candidate_paths if p.exists()), None)
        if pdf_path:
            with PDFLinkCitationExtractor(pdf_path) as extractor:
                claims, all_refs = extractor.extract_claims_and_references()
                self.assertGreater(len(claims), 10)
                self.assertGreater(len(all_refs), 10)
                # 验证第 1 页是否有 Hinton 或 Krizhevsky 的引用
                first_page_claims = [c for c in claims if c.get("source_page") == 1]
                self.assertTrue(len(first_page_claims) > 0)


    def test_auto_numbering_and_catalog(self):
        raw_unformatted = """
        Alex Krizhevsky, Ilya Sutskever, and Geoffrey E Hinton. Imagenet classification with deep convolutional neural networks. In NIPS, 2012.

        Geoffrey Hinton, Li Deng, Dong Yu, George E Dahl, Abdel-rahman Mohamed, Navdeep Jaitly, Andrew Senior, Vincent Vanhoucke, Patrick Nguyen, Tara N Sainath, et al. Deep neural networks for acoustic modeling in speech recognition. IEEE, 2012.
        """
        formatted = ReferenceFormatter.format(raw_unformatted)
        self.assertIn("[1]", formatted)
        self.assertIn("[2]", formatted)

        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".md", delete=False) as tf:
            tf.write(formatted)
            tf.flush()
            temp_path = Path(tf.name)

        try:
            catalog = parse_reference_catalog_from_md(temp_path)
            self.assertEqual(len(catalog), 2)
            self.assertEqual(catalog[0]["canonical_id"], "[1]")
            self.assertEqual(catalog[0]["index"], 1)
            self.assertIn("Krizhevsky", catalog[0]["author_year_key"])
            self.assertEqual(catalog[1]["canonical_id"], "[2]")
            self.assertIn("Hinton", catalog[1]["author_year_key"])

            refs_dict = parse_references_from_md(temp_path)
            self.assertIn("[1]", refs_dict)
            self.assertIn("[2]", refs_dict)
            self.assertIn(catalog[0]["author_year_key"], refs_dict)
        finally:
            temp_path.unlink(missing_ok=True)

    def test_citations_directory_and_paper_folder_structure(self):
        """测试 1. 论文名字文件夹规范化 2. citations/ 文件夹及以 1 开始排序的文件生成"""
        # 1. 验证论文文件夹命名规范
        self.assertEqual(sanitize_paper_folder_name("Layer Normalization"), "Layer_Normalization")
        self.assertEqual(sanitize_paper_folder_name("Attention Is All You Need"), "Attention_Is_All_You_Need")
        self.assertEqual(sanitize_paper_folder_name("attention as"), "attention_as")
        self.assertEqual(sanitize_paper_folder_name("[1]"), "[1]")

        # 2. 验证 citations/ 文件夹与 1.md, 2.md 顺序生成
        with tempfile.TemporaryDirectory() as td:
            temp_dir = Path(td)
            mock_result = {
                "paper_id": "test_paper",
                "paper_title": "Attention Is All You Need",
                "extraction_method": "pdf_vector_links",
                "total_claims": 2,
                "total_unique_refs_cited": 2,
                "total_refs_in_paper": 2,
                "extracted_at": "2026-09-30 18:00:00",
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "paragraph": "The Transformer is the first transduction model relying entirely on self-attention.",
                        "citation_keys": ["[1]"],
                        "citation_key_map": {"[1]": "[Vaswani et al., 2017]"},
                        "matched_references": {
                            "[1]": "Ashish Vaswani et al. Attention is all you need. NeurIPS 2017."
                        },
                        "source_section": "1 Introduction",
                        "source_file": "01_Introduction.md"
                    },
                    {
                        "claim_id": "claim_002",
                        "paragraph": "Recurrent models generate a sequence of hidden states.",
                        "citation_keys": ["[2]"],
                        "citation_key_map": {"[2]": "[Hochreiter and Schmidhuber, 1997]"},
                        "matched_references": {
                            "[2]": "Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. 1997."
                        },
                        "source_section": "2 Background",
                        "source_file": "02_Background.md"
                    }
                ]
            }

            mock_catalog = [
                {"index": 1, "canonical_id": "[1]", "author_year_key": "[Vaswani et al., 2017]", "title": "Attention is all you need", "raw_text": "Ashish Vaswani et al. Attention is all you need. NeurIPS 2017."},
                {"index": 2, "canonical_id": "[2]", "author_year_key": "[Hochreiter and Schmidhuber, 1997]", "title": "Long short-term memory", "raw_text": "Sepp Hochreiter and Jurgen Schmidhuber. Long short-term memory. 1997."}
            ]

            citations_dir = _write_citations_directory(temp_dir, mock_result, mock_catalog, {})

            self.assertTrue(citations_dir.is_dir())
            f1 = citations_dir / "1.md"
            f1_json = citations_dir / "1.json"
            f2 = citations_dir / "2.md"
            f2_json = citations_dir / "2.json"
            idx_md = citations_dir / "index.md"
            idx_json = citations_dir / "index.json"

            self.assertTrue(f1.is_file(), "1.md 必须存在并从 1 开始排序")
            self.assertTrue(f1_json.is_file(), "1.json 必须存在")
            self.assertTrue(f2.is_file(), "2.md 必须存在并按序递增")
            self.assertTrue(f2_json.is_file(), "2.json 必须存在")
            self.assertTrue(idx_md.is_file(), "index.md 索引必须存在")
            self.assertTrue(idx_json.is_file(), "index.json 索引必须存在")

            f1_content = f1.read_text(encoding="utf-8")
            self.assertIn("# 引用 #1 (Citation 1)", f1_content)
            self.assertIn("01_Introduction.md", f1_content)
            self.assertIn("The Transformer is the first", f1_content)
            self.assertIn("[1]", f1_content)
            self.assertIn("../cited_papers/[1]/", f1_content)


if __name__ == "__main__":
    unittest.main()
