"""作者-年份与参考文献高级匹配契约测试 (Step 14 TDD)

验证目标：
1. 优先级 1: PDF 物理链接目标 (raw_matched) 最高优先匹配
2. 优先级 2: DOI / arXiv ID 强特征精准对齐
3. 优先级 3: 标准引用键与带有 a/b 后缀的键 (Smith et al., 2020a vs 2020b) 严格隔离不混淆
4. 优先级 4: 同姓氏不同合作者 (Smith & Doe vs Smith & Brown) 能够精准消歧，杜绝贪婪误匹配
5. 括号、逗号、et al. 等标点变体自适应归一化匹配
"""
import unittest

from parsers.folder_extractor import (
    _align_citation_keys,
    _build_advanced_catalog_lookup,
    _make_catalog_item,
)


class TestAuthorYearAlignment(unittest.TestCase):

    def setUp(self):
        # 构造基准文献目录
        self.raw_catalog = [
            _make_catalog_item(
                1, "[1]",
                "Vaswani, A., Shazeer, N., Parmar, N., et al. Attention is all you need. NeurIPS, 2017. arXiv:1706.03762.",
            ),
            _make_catalog_item(
                2, "[2]",
                "Wadden, D. et al. Fact or fiction. EMNLP, 2020. DOI: 10.18653/v1/2020.emnlp-main.701.",
            ),
            _make_catalog_item(
                3, "[Smith et al., 2020a]",
                "Smith, J., Doe, A. (2020a). Deep learning for natural language processing. In ACL 2020.",
                custom_key="[Smith et al., 2020a]",
            ),
            _make_catalog_item(
                4, "[Smith et al., 2020b]",
                "Smith, J., Doe, A. (2020b). Reinforcement learning with transformers. In ICML 2020.",
                custom_key="[Smith et al., 2020b]",
            ),
            _make_catalog_item(
                5, "[Smith & Brown, 2020]",
                "Smith, B., Brown, C. (2020). Quantum computing architectures. In Nature 2020.",
                custom_key="[Smith & Brown, 2020]",
            ),
        ]
        self.catalog_lookup = _build_advanced_catalog_lookup(self.raw_catalog)

    def test_align_doi_and_arxiv_keys(self):
        """测试 DOI / arXiv ID 强标识直接命中对应文献"""
        # 测试 arXiv ID 命中
        canon_keys, c_map, matched = _align_citation_keys(
            ["arXiv:1706.03762"], self.raw_catalog, self.catalog_lookup
        )
        self.assertEqual(canon_keys, ["[1]"])
        self.assertIn("Attention is all you need", matched["[1]"])

        # 测试 DOI 命中
        canon_keys, c_map, matched = _align_citation_keys(
            ["10.18653/v1/2020.emnlp-main.701"], self.raw_catalog, self.catalog_lookup
        )
        self.assertEqual(canon_keys, ["[2]"])
        self.assertIn("Fact or fiction", matched["[2]"])

    def test_align_author_year_suffix_a_and_b(self):
        """测试 2020a 与 2020b 精准对齐到对应文献，杜绝相互覆盖或误归并"""
        # 匹配 2020a
        canon_keys_a, _, matched_a = _align_citation_keys(
            ["[Smith et al., 2020a]"], self.raw_catalog, self.catalog_lookup
        )
        self.assertEqual(canon_keys_a, ["[Smith et al., 2020a]"])
        self.assertIn("Deep learning for natural language", matched_a["[Smith et al., 2020a]"])

        # 匹配 2020b
        canon_keys_b, _, matched_b = _align_citation_keys(
            ["[Smith et al., 2020b]"], self.raw_catalog, self.catalog_lookup
        )
        self.assertEqual(canon_keys_b, ["[Smith et al., 2020b]"])
        self.assertIn("Reinforcement learning with transformers", matched_b["[Smith et al., 2020b]"])

    def test_align_author_year_various_punctuation_and_et_al(self):
        """测试不同标点变体（如圆括号、去逗号、去 et al）能自适应归一化匹配"""
        variants = [
            "(Smith et al., 2020a)",
            "[Smith et al. 2020a]",
            "[Smith, 2020a]",
        ]
        for var in variants:
            canon_keys, _, matched = _align_citation_keys(
                [var], self.raw_catalog, self.catalog_lookup
            )
            self.assertEqual(
                canon_keys, ["[Smith et al., 2020a]"],
                f"变体 {var} 应当正确对齐至 [Smith et al., 2020a]"
            )

    def test_disambiguate_same_surname_different_coauthors(self):
        """测试同姓氏但在同一年份有不同合作者的文献（Smith & Doe vs Smith & Brown）准确消歧"""
        canon_keys, _, matched = _align_citation_keys(
            ["[Smith and Brown, 2020]"], self.raw_catalog, self.catalog_lookup
        )
        self.assertEqual(canon_keys, ["[Smith & Brown, 2020]"])
        self.assertIn("Quantum computing architectures", matched["[Smith & Brown, 2020]"])

    def test_pdf_link_physical_destination_priority(self):
        """测试 PDF 物理链接目标文本 (raw_matched) 优先级最高"""
        # 正文锚点可能模糊 (如仅显示 [Smith, 2020])，但底层 PDF 物理链接命中了 2020b 的条目
        canon_keys, _, matched = _align_citation_keys(
            ["[Smith, 2020]"],
            self.raw_catalog,
            self.catalog_lookup,
            raw_matched={"[Smith, 2020]": "Smith, J., Doe, A. (2020b). Reinforcement learning with transformers."}
        )
        # 必须根据物理目标精准锚定到 2020b，而不能盲目取第一个 2020a
        self.assertEqual(canon_keys, ["[Smith et al., 2020b]"])
        self.assertIn("Reinforcement learning with transformers", matched["[Smith et al., 2020b]"])


if __name__ == "__main__":
    unittest.main()
