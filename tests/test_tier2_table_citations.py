"""Tier 2 Markdown 表格引用提取契约测试 (Step 13 TDD)

验证目标：
1. 识别 Markdown 表格，坚决杜绝跳过 table 行
2. 提取表格上/下方的表题 (table_caption)
3. 提取表头列表 (table_headers)
4. 逐行识别数据行中的引用锚点，无引用的行忽略，有引用的行独立生成 claim
5. 输出包含 content_type="table", table_caption, table_headers, paragraph, citation_keys
"""
import json
import tempfile
import unittest
from pathlib import Path

from parsers.folder_extractor import extract_cited_paragraphs, extract_from_paper_folder


class TestTier2TableCitations(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_extract_cited_paragraphs_finds_table_citations(self):
        """测试从包含表格的 Markdown 章节中精准提取表格行引用及列头元数据"""
        sec_content = (
            "# 5 Results\n\n"
            "We evaluate on WMT 2014 English-to-German task as described in [1].\n\n"
            "Table 1: Translation results across recurrent and convolutional models.\n\n"
            "| Model | EN-DE BLEU |\n"
            "| --- | --- |\n"
            "| GNMT [38] | 24.6 |\n"
            "| ConvS2S [9] | 25.16 |\n"
            "| Transformer (ours) | 28.4 |\n\n"
            "Our model establishes a new state of the art [2]."
        )
        sec_path = self.root / "05_Results.md"
        sec_path.write_text(sec_content, encoding="utf-8")

        claims = extract_cited_paragraphs(sec_path, "5 Results")

        # 应当提取出 4 处引用：
        # 1. 表前散文 -> [1]
        # 2. 表格行 1 -> [38]
        # 3. 表格行 2 -> [9]
        # (Transformer 行无引用，不生成 claim)
        # 4. 表后散文 -> [2]
        self.assertEqual(len(claims), 4, f"预期提取 4 处引用，实际提取 {len(claims)}: {claims}")

        # 验证 Claim 0 (表前散文)
        c0 = claims[0]
        self.assertEqual(c0.get("content_type"), "prose")
        self.assertIn("[1]", c0["citation_keys"])

        # 验证 Claim 1 (表格行 1)
        c1 = claims[1]
        self.assertEqual(c1.get("content_type"), "table")
        self.assertIn("Table 1", c1.get("table_caption", ""))
        self.assertEqual(c1.get("table_headers"), ["Model", "EN-DE BLEU"])
        self.assertIn("GNMT [38]", c1["paragraph"])
        self.assertIn("24.6", c1["paragraph"])
        self.assertEqual(c1["citation_keys"], ["[38]"])

        # 验证 Claim 2 (表格行 2)
        c2 = claims[2]
        self.assertEqual(c2.get("content_type"), "table")
        self.assertEqual(c2.get("table_headers"), ["Model", "EN-DE BLEU"])
        self.assertIn("ConvS2S [9]", c2["paragraph"])
        self.assertIn("25.16", c2["paragraph"])
        self.assertEqual(c2["citation_keys"], ["[9]"])

        # 验证 Claim 3 (表后散文)
        c3 = claims[3]
        self.assertEqual(c3.get("content_type"), "prose")
        self.assertIn("[2]", c3["citation_keys"])

    def test_extract_from_paper_folder_tier2_includes_table_claims(self):
        """测试纯 Markdown 模式下 extract_from_paper_folder 生成包含 table 结构化元数据的 extracted_claims.json"""
        paper_dir = self.root / "paper_test"
        paper_dir.mkdir(parents=True)

        metadata = {
            "paper_id": "test_t2",
            "paper_title": "Test Tier 2 Tables",
            "outline": [
                {"index": 0, "title": "5 Results", "file": "05_Results.md", "start_page": 5, "end_page": 5}
            ]
        }
        (paper_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        sec_content = (
            "# 5 Results\n\n"
            "Prior baselines are summarized below.\n\n"
            "| Architecture | BLEU |\n"
            "| --- | --- |\n"
            "| ByteNet [18] | 23.75 |\n"
        )
        (paper_dir / "05_Results.md").write_text(sec_content, encoding="utf-8")

        ref_content = (
            "# References\n\n"
            "[18] Nal Kalchbrenner et al. Neural machine translation in linear time. 2016."
        )
        (paper_dir / "References.md").write_text(ref_content, encoding="utf-8")

        result = extract_from_paper_folder(str(paper_dir))
        self.assertEqual(result["extraction_method"], "markdown_syntax")
        self.assertEqual(result["total_claims"], 1)

        claim = result["claims"][0]
        self.assertEqual(claim.get("content_type"), "table")
        self.assertEqual(claim.get("table_headers"), ["Architecture", "BLEU"])
        self.assertIn("ByteNet [18]", claim["paragraph"])
        self.assertIn("[18]", claim["citation_keys"])
        self.assertIn("[18]", claim["matched_references"])


if __name__ == "__main__":
    unittest.main()
