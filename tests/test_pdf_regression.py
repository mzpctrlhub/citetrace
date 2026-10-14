"""PDF 处理回归基准：对 manifest 中登记的论文夹具做可重复断言。

夹具需先通过 demo_verifier.py 生成缓存 (cache/papers/<name>)。
本测试完全离线、确定性，用于防止解析/对齐逻辑改动造成回归。
"""
import json
import unittest
from pathlib import Path

from parsers.folder_extractor import extract_from_paper_folder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANIFEST = PROJECT_ROOT / "tests" / "fixtures" / "papers" / "manifest.json"


def _load_manifest() -> dict:
    if not MANIFEST.exists():
        return {"papers": []}
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


class TestPDFRegression(unittest.TestCase):
    """基于固定夹具的解析-对齐回归测试（离线，不联网）"""

    def test_manifested_paper_fixtures(self):
        manifest = _load_manifest()
        self.assertTrue(manifest["papers"], "manifest.json 至少应登记一个夹具")
        checked = 0

        for spec in manifest["papers"]:
            folder = PROJECT_ROOT / spec["folder"]
            if not (folder / "metadata.json").is_file():
                self.skipTest(
                    f"夹具未缓存: {spec['folder']} — 先运行 demo_verifier.py 生成后再跑回归"
                )

            # 1. 章节切分数量下限
            metadata = json.loads(
                (folder / "metadata.json").read_text(encoding="utf-8")
            )
            if "expect_sections_min" in spec:
                self.assertGreaterEqual(
                    len(metadata.get("outline", [])),
                    spec["expect_sections_min"],
                    f"{spec['folder']} 章节数低于预期",
                )

            # 2. 重跑抽取管线（离线、确定性）
            result = extract_from_paper_folder(str(folder))

            # 3. 数据契约字段：pdf 指纹 / parser 版本 / 上下文 ID
            self.assertIsNotNone(result.get("pdf_sha256"), "缺少 pdf_sha256 指纹")
            self.assertIsNotNone(result.get("parser_version"), "缺少 parser_version")
            for c in result["claims"]:
                self.assertTrue(
                    c.get("context_id", "").startswith("ctx_"),
                    "每条引用上下文必须带 context_id",
                )

            # 4. 数量区间
            if "expect_refs" in spec:
                self.assertEqual(result["total_refs_in_paper"], spec["expect_refs"])
            if "expect_contexts_min" in spec:
                self.assertGreaterEqual(result["total_claims"], spec["expect_contexts_min"])
            if "expect_contexts_max" in spec:
                self.assertLessEqual(result["total_claims"], spec["expect_contexts_max"])

            # 5. 引用-文献对齐完整性（回归保护：对齐率不得显著回退）
            total_keys = 0
            matched_keys = 0
            for c in result["claims"]:
                self.assertTrue(c["citation_keys"], f"{c['claim_id']} 无引用键")
                for k in c["citation_keys"]:
                    total_keys += 1
                    if c["matched_references"].get(k):
                        matched_keys += 1
            self.assertGreater(total_keys, 0)
            ratio = matched_keys / total_keys
            self.assertGreaterEqual(
                ratio, 0.9,
                f"引用-文献对齐率回退: {ratio:.2f} ({matched_keys}/{total_keys})",
            )
            checked += 1

        self.assertGreater(checked, 0)

    def test_text_cleaner_latex_and_font_artifacts(self):
        from citetrace_pdf.parser.cleaner import TextCleaner
        cleaner = TextCleaner()
        sample = (
            "1 _._ 0% and 90 _._ 53% with p = 0 . 5 and 17 . 2 BLEU.\n"
            "Set _{_ _x_ _, y_ _}_ with _1_ and _100_ items.\n"
            "Formula \x00a + b\x01 and \x10c + d\x11 and \uf8ee3x3\uf8f9.\n"
            "European Commis-\n\nsion and transduc- tion vs sequence-\naligned and position-\nwise.\n"
            "Layers conv3 ~~x~~ and fast ~~a~~ lign with _~~√~~_ <u>1</u> _dk_ and 2<sup>5</sup><sup>.6</sup>.\n"
            "Scaling factor of\n1\n√dk .\n"
            "We call this \\Long \\short term memory.\n"
            "**\n\n** and **** here."
        )
        cleaned = cleaner.clean(sample)
        self.assertIn("1.0%", cleaned)
        self.assertIn("90.53%", cleaned)
        self.assertIn("0.5", cleaned)
        self.assertIn("17.2", cleaned)
        self.assertNotIn("_{_", cleaned)
        self.assertNotIn("_1_", cleaned)
        self.assertIn("(a + b)", cleaned)
        self.assertIn("(c + d)", cleaned)
        self.assertIn("[3x3]", cleaned)
        self.assertIn("Commission", cleaned)
        self.assertIn("transduction", cleaned)
        self.assertIn("sequence-aligned", cleaned)
        self.assertIn("position-wise", cleaned)
        self.assertIn("conv3_x", cleaned)
        self.assertIn("fast_align", cleaned)
        self.assertIn("2<sup>5.6</sup>", cleaned)
        self.assertIn("1/√dk", cleaned)
        self.assertIn('"Long "short', cleaned)
        self.assertNotIn("****", cleaned)


if __name__ == "__main__":
    unittest.main()


