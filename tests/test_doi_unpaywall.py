"""纯 DOI 文献检索与 Unpaywall 下载链路测试 (Step 9)

核心断言：
1. fetch_unpaywall_record_by_doi 能够完整提取文献标题、年份、主 PDF 直链以及多候选列表 (oa_locations)
2. 仅有 DOI（无 title、无 arXiv ID）的 CitationRef，通过 enrich_citation_with_cache 能够反向恢复标题、主 PDF 与 candidates
3. 纯 DOI 检索结果正确写入全局稳态缓存 (.metadata_cache/doi_<hash_16>.json)
4. 若第一候选链接失效或标题不符，系统自动平滑回退至 Unpaywall 提供的后续候选 URL
5. 当 Unpaywall 无 OA 链接时，能够通过 Crossref by DOI 补全文献标题，标记为受付费墙保护
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from models.citation import CitationRef, CitationStatus
from tools.academic_fetcher import (
    enrich_citation_with_cache,
    fetch_paper_document,
    fetch_unpaywall_record_by_doi,
    fetch_unpaywall_by_doi,
    fetch_crossref_by_doi,
    get_global_cache_key,
)


class TestDOIUnpaywall(unittest.TestCase):
    """测试纯 DOI 检索与 Unpaywall 候选链完备性"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache_dir = self.root / "papers"
        self.cache_dir.mkdir(parents=True)

        self.mock_unpaywall_data = {
            "doi": "10.1038/nature14539",
            "title": "Mastering the game of Go with deep neural networks and tree search",
            "year": 2016,
            "is_oa": True,
            "best_oa_location": {
                "url_for_pdf": "https://storage.googleapis.com/deepmind-media/alphago/nature14539.pdf",
                "url": "https://nature.com/articles/nature14539",
                "version": "publishedVersion",
            },
            "oa_locations": [
                {
                    "url_for_pdf": "https://storage.googleapis.com/deepmind-media/alphago/nature14539.pdf",
                    "url": "https://nature.com/articles/nature14539",
                },
                {
                    "url_for_pdf": "https://openaccess.city.ac.uk/id/eprint/13861/1/nature14539.pdf",
                    "url": "https://openaccess.city.ac.uk/id/eprint/13861/",
                },
            ],
        }

    def tearDown(self):
        self.tmp.cleanup()

    @patch("tools.academic_fetcher._http_get_json")
    def test_unpaywall_record_extracts_title_year_and_multiple_pdf_candidates(self, mock_get):
        mock_get.return_value = self.mock_unpaywall_data

        rec = fetch_unpaywall_record_by_doi("10.1038/nature14539")
        self.assertIsNotNone(rec)
        self.assertEqual(
            rec["title"],
            "Mastering the game of Go with deep neural networks and tree search"
        )
        self.assertEqual(rec["year"], 2016)
        self.assertEqual(
            rec["pdf_url"],
            "https://storage.googleapis.com/deepmind-media/alphago/nature14539.pdf"
        )
        self.assertEqual(len(rec["pdf_candidates"]), 2)
        self.assertIn("https://openaccess.city.ac.uk/id/eprint/13861/1/nature14539.pdf", rec["pdf_candidates"])

    @patch("tools.academic_fetcher._http_get_json")
    def test_enrich_citation_with_doi_only_recovers_metadata_and_candidates(self, mock_get):
        """纯 DOI（无标题）引用能够自动反向补全标题与全部 PDF 候选，并正确写入缓存"""
        mock_get.return_value = self.mock_unpaywall_data

        ref = CitationRef(
            key="[42]",
            raw_text="[42] Silver et al. doi:10.1038/nature14539",
            doi="10.1038/nature14539",
            title=None,
            year=None,
        )

        resolved_ref, step_source = enrich_citation_with_cache(ref, cache_dir=str(self.cache_dir))

        self.assertEqual(
            resolved_ref.title,
            "Mastering the game of Go with deep neural networks and tree search"
        )
        self.assertEqual(resolved_ref.year, 2016)
        self.assertEqual(resolved_ref.status, CitationStatus.FETCHED)
        self.assertEqual(
            resolved_ref.pdf_url,
            "https://storage.googleapis.com/deepmind-media/alphago/nature14539.pdf"
        )
        self.assertEqual(len(resolved_ref.pdf_candidates), 2)

        # 验证全局缓存正确写入
        key = get_global_cache_key(resolved_ref)
        self.assertTrue(key.startswith("doi_"))
        cache_file = self.cache_dir / ".metadata_cache" / f"{key}.json"
        self.assertTrue(cache_file.is_file(), "全局 metadata 缓存未成功写入")
        cached = json.loads(cache_file.read_text(encoding="utf-8"))
        self.assertEqual(cached["title"], resolved_ref.title)
        self.assertEqual(len(cached["pdf_candidates"]), 2)

    @patch("tools.academic_fetcher._http_get_json")
    def test_doi_only_crossref_metadata_fallback_when_paywalled(self, mock_get):
        """当 Unpaywall 无 OA 链接时，通过 Crossref by DOI 补充论文标题与年份"""
        def _mock_dispatch(url, timeout=10):
            if "unpaywall.org" in url:
                return {
                    "doi": "10.1016/j.cell.2020.01.001",
                    "is_oa": False,
                    "best_oa_location": None,
                    "oa_locations": [],
                }
            if "crossref.org" in url:
                return {
                    "message": {
                        "title": ["A Molecular Atlas of Cell Types in the Brain"],
                        "issued": {"date-parts": [[2020, 2, 1]]},
                        "link": [],
                    }
                }
            return {}

        mock_get.side_effect = _mock_dispatch

        ref = CitationRef(
            key="[99]",
            raw_text="[99] doi:10.1016/j.cell.2020.01.001",
            doi="10.1016/j.cell.2020.01.001",
            title=None,
        )

        resolved_ref, step_source = enrich_citation_with_cache(ref, cache_dir=str(self.cache_dir))

        self.assertEqual(resolved_ref.title, "A Molecular Atlas of Cell Types in the Brain")
        self.assertEqual(resolved_ref.year, 2020)
        self.assertEqual(resolved_ref.status, CitationStatus.PAYWALL_LOCKED)

    @patch("tools.academic_fetcher._http_get_json")
    @patch("tools.academic_fetcher.PDFProcessor")
    def test_doi_only_fetch_paper_document_with_fallback(self, mock_processor_cls, mock_get):
        """纯 DOI 论文通过 fetch_paper_document 获取时，支持首选链接失效后平滑回退至后续候选并恢复标题"""
        mock_get.return_value = self.mock_unpaywall_data

        target_dir = self.cache_dir / "[42]"
        target_dir.mkdir(parents=True, exist_ok=True)

        valid_pdf_bytes = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
            + b"A" * 1024
            + b"\nxref\n0 4\n0000000000 65535 f\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n100\n%%EOF\n"
        )
        import hashlib
        pdf_sha = hashlib.sha256(valid_pdf_bytes).hexdigest()

        def _mock_process(paper_id, pdf_url, force=False, **kwargs):
            if "deepmind-media" in pdf_url:
                raise RuntimeError("HTTP 403 Forbidden")
            # 候选 2 成功
            (target_dir / "raw.pdf").write_bytes(valid_pdf_bytes)
            (target_dir / "metadata.json").write_text(
                json.dumps({
                    "paper_id": "[42]",
                    "paper_title": "Mastering the game of Go with deep neural networks and tree search",
                    "pdf_sha256": pdf_sha,
                    "artifact_schema_version": 2,
                    "parser_version": "2.1.0",
                    "chunker_version": "1.0.0",
                    "title_extractor_version": "1.1.0",
                    "outline_audit": {"passed": True, "issues": []},
                    "outline": [{"index": 0, "title": "Abstract", "file": "00_Abstract.md"}],
                }),
                encoding="utf-8",
            )
            (target_dir / "00_Abstract.md").write_text("# Abstract\nText", encoding="utf-8")
            (target_dir / "chunks.json").write_text(
                json.dumps([{"chunk_id": "c1", "text": "Text"}]),
                encoding="utf-8",
            )
            return {
                "paper_title": "Mastering the game of Go with deep neural networks and tree search",
                "folder": str(target_dir),
            }

        mock_proc_instance = mock_processor_cls.return_value
        mock_proc_instance.process_paper.side_effect = _mock_process
        mock_proc_instance.loader.is_valid_pdf.return_value = False

        ref = CitationRef(
            key="[42]",
            raw_text="[42] doi:10.1038/nature14539",
            doi="10.1038/nature14539",
            title=None,
        )

        updated_ref, meta = fetch_paper_document(
            ref,
            cache_dir=str(self.cache_dir),
            global_cache_dir=str(self.cache_dir),
        )

        self.assertEqual(updated_ref.status, CitationStatus.FETCHED)
        self.assertEqual(
            updated_ref.title,
            "Mastering the game of Go with deep neural networks and tree search"
        )
        self.assertEqual(
            updated_ref.pdf_url,
            "https://openaccess.city.ac.uk/id/eprint/13861/1/nature14539.pdf"
        )
        self.assertEqual(
            meta.get("paper_title"),
            "Mastering the game of Go with deep neural networks and tree search"
        )


if __name__ == "__main__":
    unittest.main()
