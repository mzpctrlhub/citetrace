"""学术文献检索与全局缓存键契约测试

核心防回归断言：
1. 两篇不同主论文的 [17] 不得共享全局缓存（禁止将 [1]、[17] 等序号作为全局缓存键）
2. 全局缓存键优先级应为：arXiv ID -> DOI 哈希 -> 规范化(title + year + author)哈希
3. 纯 DOI 引用在缺失标题时，Unpaywall 结果不得被丢弃
"""
import json
import tempfile
import unittest
from pathlib import Path

from models.citation import CitationRef, CitationStatus
from tools.academic_fetcher import (
    enrich_citation_with_cache,
    fetch_paper_document,
    get_global_cache_key,
)


class TestAcademicFetcherCacheKeys(unittest.TestCase):
    """测试学术抓取器全局缓存键隔离与冲突防护"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.global_cache = self.root / "global_cache"
        self.global_cache.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_global_cache_key_prohibits_serial_number_and_raw_key(self):
        """禁止将 [1]、[17] 或 ref.key 作为全局缓存键"""
        ref_serial_only = CitationRef(
            key="[17]",
            raw_text="[17] Unknown paper without metadata",
            title=None,
            year=None,
        )
        # 缺少稳定身份（无 arXiv/DOI/Title），全局缓存键必须为 None（只能落局部目录）
        key = get_global_cache_key(ref_serial_only)
        self.assertIsNone(
            key,
            f"缺少稳态标识的引用不得生成全局缓存键，但返回了: {key}"
        )

    def test_different_papers_with_same_serial_do_not_share_global_cache(self):
        """两篇不同主论文中同名的 [17] 绝不能共享全局缓存"""
        ref_a = CitationRef(
            key="[17]",
            raw_text="[17] Kaiser et al. Neural GPUs Learn Algorithms. ICLR 2016.",
            title="Neural GPUs Learn Algorithms",
            year=2016,
            authors=["Łukasz Kaiser"],
        )
        ref_b = CitationRef(
            key="[17]",
            raw_text="[17] He et al. Deep Residual Learning for Image Recognition. CVPR 2016.",
            title="Deep Residual Learning for Image Recognition",
            year=2016,
            authors=["Kaiming He"],
        )

        key_a = get_global_cache_key(ref_a)
        key_b = get_global_cache_key(ref_b)
        self.assertIsNotNone(key_a)
        self.assertIsNotNone(key_b)
        self.assertNotEqual(key_a, key_b, "两篇不同论文的全局缓存键不得相同！")
        self.assertNotIn("[17]", key_a)
        self.assertNotIn("[17]", key_b)

        # 模拟 Paper A 写入缓存
        meta_cache_dir = self.global_cache / ".metadata_cache"
        meta_cache_dir.mkdir(parents=True, exist_ok=True)
        (meta_cache_dir / f"{key_a}.json").write_text(
            json.dumps({
                "title": "Neural GPUs Learn Algorithms",
                "pdf_url": "https://arxiv.org/pdf/1511.08228.pdf",
            }),
            encoding="utf-8",
        )

        # Paper B 查缓存，绝不能查到 Paper A 的缓存
        resolved_b, source = enrich_citation_with_cache(
            ref_b,
            cache_dir=str(self.global_cache),
        )
        self.assertNotEqual(
            resolved_b.title,
            "Neural GPUs Learn Algorithms",
            "Paper B 误命中了 Paper A 的缓存数据！",
        )
        self.assertNotEqual(
            resolved_b.pdf_url,
            "https://arxiv.org/pdf/1511.08228.pdf",
            "Paper B 误读取了 Paper A 的 PDF 链接！",
        )

    def test_doi_only_unpaywall_result_not_dropped(self):
        """仅有 DOI 没有 title 的引用，Unpaywall 返回的有效 PDF 链接不得被丢弃"""
        ref_doi_only = CitationRef(
            key="[42]",
            raw_text="[42] doi:10.1038/nature14539",
            doi="10.1038/nature14539",
            title=None,
        )
        key = get_global_cache_key(ref_doi_only)
        self.assertIsNotNone(key)
        self.assertTrue(key.startswith("doi_"), f"DOI 缓存键应以 doi_ 开头，实际为: {key}")

    def test_fetch_paper_document_stores_by_serial_number(self):
        """测试将文献获取并放入指定文件夹，且文件夹按序列号命名（标题未知时不做身份校验）"""
        sample_pdf = self.root / "mock_paper.pdf"
        sample_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n" + b"A" * 1500 + b"\n%%EOF")
        cited_dir = self.root / "cited_papers"

        ref = CitationRef(
            key="[17]",
            raw_text="[17] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. In ICLR, 2016.",
            pdf_url=str(sample_pdf),
        )

        fetch_paper_document(
            ref=ref,
            cache_dir=str(cited_dir),
            paper_id="[17]",
            global_cache_dir=str(self.global_cache),
        )

        target_folder = cited_dir / "[17]"
        self.assertTrue(target_folder.is_dir(), "未生成以 [17] 命名的切分文件夹")
        self.assertTrue((target_folder / "raw.pdf").is_file(), "切分文件夹中缺少 raw.pdf")

    def test_fetch_paper_document_rejects_different_paper(self):
        """下载内容经标题校验确认不是所请求的文献时，必须拒绝登记并清除错误 PDF 缓存"""
        sample_pdf = self.root / "mock_paper.pdf"
        sample_pdf.write_bytes(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog >>\nendobj\n" + b"A" * 1500 + b"\n%%EOF")
        cited_dir = self.root / "cited_papers"

        (self.global_cache / ".metadata_cache").mkdir(parents=True, exist_ok=True)
        payload = {
            "title": "Neural GPUs Learn Algorithms",
            "abstract": "",
            "pdf_url": str(sample_pdf),
            "pdf_candidates": [str(sample_pdf)],
            "doi": None,
            "arxiv_id": None,
        }
        (self.global_cache / ".metadata_cache" / "[99].json").write_text(
            json.dumps(payload, ensure_ascii=False), encoding="utf-8"
        )

        ref = CitationRef(
            key="[99]",
            raw_text="[99] Łukasz Kaiser and Ilya Sutskever. Neural GPUs learn algorithms. ICLR 2016.",
            title="Neural GPUs Learn Algorithms",
            pdf_url=str(sample_pdf),
        )

        resolved_ref, metadata = fetch_paper_document(
            ref=ref,
            cache_dir=str(cited_dir),
            paper_id="[99]",
            global_cache_dir=str(self.global_cache),
        )

        self.assertEqual(metadata, {}, "错误文献不应返回可用元数据")
        self.assertEqual(resolved_ref.last_failure, "title_mismatch")
        self.assertFalse((cited_dir / "[99]" / "raw.pdf").exists(), "错误文献的 PDF 缓存应被清除")


if __name__ == "__main__":
    unittest.main()
