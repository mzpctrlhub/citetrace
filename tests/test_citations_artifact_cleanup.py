"""引用提取产物生命周期与清理契约测试 (Step 15 TDD)

验证目标：
1. _write_citations_directory 在重新生成时，能够彻底清理旧残留文件（如上次 40 个 claim 本次 35 个 claim 时的 36.md ~ 40.md）
2. extract_from_paper_folder 全流程在重复运行时，citations 目录自动净化
3. 表格引用的 content_type、table_caption、table_headers 数据流在 citations/{idx}.json 与 {idx}.md 中完整透传
"""
import json
import tempfile
import unittest
from pathlib import Path

from parsers.folder_extractor import (
    _clean_citations_directory,
    _write_citations_directory,
    extract_from_paper_folder,
)


class TestCitationsArtifactCleanup(unittest.TestCase):

    def test_write_citations_directory_cleans_obsolete_files(self):
        """测试重新生成 citations 时，旧的超出范围的 md 与 json 文件必须被彻底清理"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            citations_dir = folder / "citations"
            citations_dir.mkdir(parents=True, exist_ok=True)

            # 模拟上一次生成遗留了 40 个 claim 文件与历史脏文件
            for i in range(1, 41):
                (citations_dir / f"{i}.md").write_text(f"old claim {i}", encoding="utf-8")
                (citations_dir / f"{i}.json").write_text(json.dumps({"old": i}), encoding="utf-8")
            (citations_dir / "stale_legacy.md").write_text("stale", encoding="utf-8")
            (citations_dir / "stale_legacy.json").write_text("{}", encoding="utf-8")

            # 验证写入前有 40*2 + 2 = 82 个文件
            all_before = list(citations_dir.glob("*"))
            self.assertEqual(len(all_before), 82)

            # 构造本次仅有 2 个 claim 的新结果
            new_result = {
                "paper_id": "test_paper",
                "paper_title": "Test Paper",
                "total_claims": 2,
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "paragraph": "Prose claim citing [1]",
                        "citation_keys": ["[1]"],
                        "citation_key_map": {"[1]": "[1]"},
                        "matched_references": {"[1]": "Reference 1 text"},
                        "source_file": "01_Intro.md",
                        "source_section": "Introduction",
                        "content_type": "prose",
                    },
                    {
                        "claim_id": "claim_002",
                        "paragraph": "Table row | [2]",
                        "citation_keys": ["[2]"],
                        "citation_key_map": {"[2]": "[2]"},
                        "matched_references": {"[2]": "Reference 2 text"},
                        "source_file": "02_Results.md",
                        "source_section": "Results",
                        "content_type": "table",
                        "table_caption": "Table 1: Main Results",
                        "table_headers": ["Model", "Score"],
                    },
                ],
            }
            ref_catalog = [
                {"index": 1, "canonical_id": "[1]", "author_year_key": "[Vaswani et al., 2017]", "title": "Attention", "raw_text": "Reference 1 text"},
                {"index": 2, "canonical_id": "[2]", "author_year_key": "[Devlin et al., 2018]", "title": "BERT", "raw_text": "Reference 2 text"},
            ]
            ref_entries = {"[1]": "Reference 1 text", "[2]": "Reference 2 text"}

            # 执行写入
            out_dir = _write_citations_directory(folder, new_result, ref_catalog, ref_entries)
            self.assertEqual(out_dir, citations_dir)

            # 验证新产物存在
            self.assertTrue((citations_dir / "1.md").is_file())
            self.assertTrue((citations_dir / "1.json").is_file())
            self.assertTrue((citations_dir / "2.md").is_file())
            self.assertTrue((citations_dir / "2.json").is_file())
            self.assertTrue((citations_dir / "index.md").is_file())
            self.assertTrue((citations_dir / "index.json").is_file())

            # 验证旧的 3.md ~ 40.md 以及 stale 文件已彻底消失
            for i in range(3, 41):
                self.assertFalse((citations_dir / f"{i}.md").exists(), f"{i}.md 应当被清理")
                self.assertFalse((citations_dir / f"{i}.json").exists(), f"{i}.json 应当被清理")
            self.assertFalse((citations_dir / "stale_legacy.md").exists())
            self.assertFalse((citations_dir / "stale_legacy.json").exists())

            # 验证 citations 目录下恰好只有 6 个受管理文件 (1.md, 1.json, 2.md, 2.json, index.md, index.json)
            remaining_files = sorted([f.name for f in citations_dir.glob("*") if f.is_file()])
            expected_files = sorted(["1.md", "1.json", "2.md", "2.json", "index.md", "index.json"])
            self.assertEqual(remaining_files, expected_files)

    def test_table_claim_metadata_propagation_in_citation_files(self):
        """测试表格引用元数据（content_type, table_caption, table_headers）在独立 JSON/MD 中完整保留"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            folder = Path(tmp_dir)
            citations_dir = folder / "citations"

            result = {
                "paper_id": "test_paper",
                "paper_title": "Test Paper",
                "total_claims": 1,
                "claims": [
                    {
                        "claim_id": "claim_001",
                        "paragraph": "| Model A | [1] |",
                        "citation_keys": ["[1]"],
                        "citation_key_map": {"[1]": "[1]"},
                        "matched_references": {"[1]": "Vaswani et al."},
                        "source_file": "02_Results.md",
                        "source_section": "Results",
                        "content_type": "table",
                        "table_caption": "Table 1: Benchmark Comparison",
                        "table_headers": ["Model", "Citations"],
                    }
                ],
            }
            ref_catalog = [{"index": 1, "canonical_id": "[1]", "author_year_key": "[Vaswani, 2017]", "title": "Attention", "raw_text": "Vaswani et al."}]
            _write_citations_directory(folder, result, ref_catalog, {"[1]": "Vaswani et al."})

            # 读取生成的 1.json
            item_json = json.loads((citations_dir / "1.json").read_text(encoding="utf-8"))
            self.assertEqual(item_json.get("content_type"), "table")
            self.assertEqual(item_json.get("table_caption"), "Table 1: Benchmark Comparison")
            self.assertEqual(item_json.get("table_headers"), ["Model", "Citations"])

            # 读取生成的 1.md
            item_md = (citations_dir / "1.md").read_text(encoding="utf-8")
            self.assertIn("Table 1: Benchmark Comparison", item_md)

    def test_extract_from_paper_folder_cleans_stale_citations(self):
        """测试 extract_from_paper_folder 运行时自动净化 citations/ 目录下的历史残留文件"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paper_dir = Path(tmp_dir) / "test_paper"
            sections_dir = paper_dir / "sections"
            citations_dir = paper_dir / "citations"
            sections_dir.mkdir(parents=True, exist_ok=True)
            citations_dir.mkdir(parents=True, exist_ok=True)

            # 写入基础 metadata
            meta = {
                "paper_id": "test_paper",
                "paper_title": "Attention Paper",
                "outline": [
                    {"title": "Introduction", "file": "01_Intro.md", "start_page": 1, "end_page": 1},
                    {"title": "References", "file": "References.md", "start_page": 2, "end_page": 2},
                ],
            }
            (paper_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

            # 写入章节正文与参考文献
            (sections_dir / "01_Intro.md").write_text("We propose the Transformer [1].", encoding="utf-8")
            (sections_dir / "References.md").write_text(
                "[1] Vaswani, A., et al. (2017). Attention is all you need. NeurIPS.",
                encoding="utf-8"
            )

            # 预先放置历史脏文件
            (citations_dir / "stale_99.md").write_text("old", encoding="utf-8")
            (citations_dir / "stale_99.json").write_text("{}", encoding="utf-8")
            (citations_dir / "old_2.md").write_text("old", encoding="utf-8")

            # 运行 extract_from_paper_folder
            result = extract_from_paper_folder(str(paper_dir))
            self.assertEqual(result["total_claims"], 1)

            # 验证历史残留文件已彻底清除
            self.assertFalse((citations_dir / "stale_99.md").exists())
            self.assertFalse((citations_dir / "stale_99.json").exists())
            self.assertFalse((citations_dir / "old_2.md").exists())

            # 验证仅有当前的 1.md, 1.json, index.md, index.json
            active_files = sorted([f.name for f in citations_dir.glob("*") if f.is_file()])
            self.assertEqual(active_files, ["1.json", "1.md", "index.json", "index.md"])


if __name__ == "__main__":
    unittest.main()

