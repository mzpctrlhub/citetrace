"""引用提取去重与序号规整契约测试 (Step 12 TDD)

验证目标：
1. 规范化各类数字编号排版: [1], 1., - [1], * [2], [03], - 4. 统一规整为标准 [n]
2. 剥离条目中的 Markdown 列表前缀及残留前置序号
3. 参考文献去重：重复条目（相同编号或相同文献内容）严格去重保序
4. 作者-年份后缀与碰撞消歧：支持 2020a / 2020b，同名作者同年不同论文自动消歧
5. 附录/Checklist/致谢等尾部噪声彻底剥离
6. References.md catalog 解析保证有序无重复
"""
import tempfile
import unittest
from pathlib import Path

from citetrace_pdf.parser.reference import ReferenceFormatter
from parsers.citation_parser import parse_reference_entry, parse_references_section
from parsers.folder_extractor import parse_reference_catalog_from_md


class TestReferenceNormalization(unittest.TestCase):

    def test_numbering_normalization_various_formats(self):
        """测试多种编号格式 (- [1], * [2], 3., [04]) 统一归一化为标准 [n]"""
        raw_text = (
            "- [1] Vaswani, A. et al. Attention is all you need. NeurIPS, 2017.\n\n"
            "* [2] Ba, J. L. et al. Layer normalization. arXiv, 2016.\n\n"
            "3. Wadden, D. et al. Fact or fiction. EMNLP, 2020.\n\n"
            "[04] Devlin, J. et al. BERT. NAACL, 2019."
        )
        formatted = ReferenceFormatter.format(raw_text)
        self.assertIn("[1] Vaswani, A. et al. Attention is all you need. NeurIPS, 2017.", formatted)
        self.assertIn("[2] Ba, J. L. et al. Layer normalization. arXiv, 2016.", formatted)
        self.assertIn("[3] Wadden, D. et al. Fact or fiction. EMNLP, 2020.", formatted)
        self.assertIn("[4] Devlin, J. et al. BERT. NAACL, 2019.", formatted)
        # 绝不能出现 - [1]、* [2] 或 [3] 3. 形式的冗余前缀
        self.assertNotIn("- [1]", formatted)
        self.assertNotIn("* [2]", formatted)
        self.assertNotIn("[3] 3.", formatted)
        self.assertNotIn("[04]", formatted)

    def test_reference_deduplication(self):
        """测试文末参考文献列表中重复出现的文献条目自动去重"""
        raw_with_dups = (
            "[1] Vaswani, A. et al. Attention is all you need. NeurIPS, 2017.\n\n"
            "[2] Ba, J. L. et al. Layer normalization. arXiv, 2016.\n\n"
            "[1] Vaswani, A. et al. Attention is all you need. NeurIPS, 2017.\n\n"
            "[3] Devlin, J. et al. BERT. NAACL, 2019."
        )
        formatted = ReferenceFormatter.format(raw_with_dups)
        # [1] 只能出现一次
        self.assertEqual(formatted.count("[1] Vaswani"), 1)

        refs = parse_references_section(formatted)
        self.assertEqual(len(refs), 3)
        self.assertIn("[1]", refs)
        self.assertIn("[2]", refs)
        self.assertIn("[3]", refs)

    def test_author_year_suffix_preservation(self):
        """测试作者-年份引用保留 a/b 后缀（如 2020a, 2020b）且不混淆"""
        entry_a = "Smith, J. et al. (2020a). Deep learning for NLP. ACL 2020."
        entry_b = "Smith, J. et al. (2020b). Reinforcement learning with transformers. ICML 2020."

        ref_a = parse_reference_entry(entry_a)
        ref_b = parse_reference_entry(entry_b)

        self.assertIsNotNone(ref_a)
        self.assertIsNotNone(ref_b)
        self.assertIn("2020a", ref_a.key)
        self.assertIn("2020b", ref_b.key)
        self.assertNotEqual(ref_a.key, ref_b.key)

    def test_author_year_collision_disambiguation_in_section(self):
        """测试无后缀但同作者同年的不同文献在解析章节时自动分配 a/b 序号消歧"""
        raw_section = (
            "Smith, J. and Doe, A. (2020). Deep learning for NLP. ACL 2020.\n\n"
            "Smith, J. and Doe, A. (2020). Reinforcement learning with transformers. ICML 2020."
        )
        refs = parse_references_section(raw_section)
        # 必须解析出 2 条独立文献，绝不能互相覆盖导致只剩 1 条
        self.assertEqual(len(refs), 2)
        keys = list(refs.keys())
        self.assertEqual(len(set(keys)), 2)

    def test_trailing_noise_and_checklist_stripping(self):
        """测试尾部混入的 Checklist、致谢、附录等噪声行被彻底剥离"""
        raw_text = (
            "[39] Jie Zhou. Deep recurrent models. CoRR, 2016.\n\n"
            "[40] Muhua Zhu. Fast parsing. ACL, 2013. "
            "Acknowledgments This work was supported by NSF grant 123456. "
            "NeurIPS Checklist 1. For all authors: (a) Do the main claims match? Yes.\n\n"
            "[41] Appendix A: Proofs of Lemmas"
        )
        formatted = ReferenceFormatter.format(raw_text)
        self.assertIn("[39] Jie Zhou", formatted)
        self.assertIn("[40] Muhua Zhu", formatted)
        self.assertNotIn("Acknowledgments", formatted)
        self.assertNotIn("NeurIPS Checklist", formatted)
        # 纯附录伪条目 [41] 必须被过滤掉
        self.assertNotIn("Appendix A", formatted)

    def test_catalog_parsing_deduplication_and_clean_keys(self):
        """测试 parse_reference_catalog_from_md 生成有序无冗余的清单"""
        md_content = (
            "# References\n\n"
            "- [1] Vaswani, A. et al. Attention is all you need. NeurIPS, 2017.\n\n"
            "- [2] Ba, J. L. et al. Layer normalization. arXiv, 2016.\n\n"
            "- [1] Vaswani, A. et al. Attention is all you need. NeurIPS, 2017."
        )
        with tempfile.NamedTemporaryFile("w+", encoding="utf-8", suffix=".md", delete=False) as tf:
            tf.write(md_content)
            tf.flush()
            temp_path = Path(tf.name)

        try:
            catalog = parse_reference_catalog_from_md(temp_path)
            self.assertEqual(len(catalog), 2)
            self.assertEqual(catalog[0]["canonical_id"], "[1]")
            self.assertEqual(catalog[1]["canonical_id"], "[2]")
            # 保证 raw_text 中不含前置 - 
            self.assertFalse(catalog[0]["raw_text"].startswith("-"))
        finally:
            temp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
