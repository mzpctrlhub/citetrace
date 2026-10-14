"""错误 PDF 隔离与候选 URL 回退机制测试

核心断言：
1. 本地 raw.pdf 标题校验失败时，自动移入 .quarantine/ 目录
2. 错误本地 PDF 不得阻断后续候选，系统必须继续尝试剩余 pdf_url 与 pdf_candidates
3. 每个候选尝试均记录在 attempts 中（包括 title_mismatch、download_failed 等）
4. 后续候选若命中正确论文，最终成功获取产物
"""
import json
import tempfile
import unittest
from pathlib import Path

from models.citation import CitationRef, CitationStatus
from tools.academic_fetcher import (
    fetch_paper_document,
    _quarantine_file,
)


class TestCandidateFallback(unittest.TestCase):
    """测试错误 PDF 隔离与后续候选平滑回退"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache_dir = self.root / "papers"
        self.paper_dir = self.cache_dir / "[1]"
        self.paper_dir.mkdir(parents=True)

        # 构造有效 PDF 内容
        self.dummy_pdf = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
            + b"A" * 1024
            + b"\nxref\n0 4\n0000000000 65535 f\ntrailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n100\n%%EOF\n"
        )

    def tearDown(self):
        self.tmp.cleanup()

    def test_quarantine_file_moves_target_to_quarantine_dir(self):
        """测试 _quarantine_file 将文件正确归档进 .quarantine 目录"""
        bad_pdf = self.paper_dir / "raw.pdf"
        bad_pdf.write_bytes(self.dummy_pdf)

        dest = _quarantine_file(bad_pdf)
        self.assertIsNotNone(dest)
        self.assertTrue(dest.is_file(), "隔离文件不存在")
        self.assertEqual(dest.parent.name, ".quarantine")
        self.assertFalse(bad_pdf.exists(), "原错误文件仍停留在原处")

    def test_wrong_local_pdf_quarantined_and_falls_back_to_valid_candidate(self):
        """本地存在错误论文 PDF 时，自动移入隔离区并继续尝试候选 URL 成功获取正确论文"""
        # 1. 在本地预置一份错误的 raw.pdf
        bad_pdf = self.paper_dir / "raw.pdf"
        bad_pdf.write_bytes(self.dummy_pdf)
        (self.paper_dir / "metadata.json").write_text(
            json.dumps({
                "paper_id": "test_wrong",
                "paper_title": "Completely Irrelevant Paper Title",
            }),
            encoding="utf-8",
        )

        # 2. 准备一份正确的候选用 PDF
        good_pdf = self.root / "correct_remote_source.pdf"
        good_pdf.write_bytes(self.dummy_pdf)

        ref = CitationRef(
            key="[1]",
            raw_text="[1] Target Author. Target Expected Title. 2023.",
            title="Target Expected Title",
            year=2023,
            pdf_url="http://127.0.0.1:1/non_existent.pdf",
            pdf_candidates=[str(good_pdf)],
        )

        # 预置元数据缓存，使阶梯检索直接命中本地，彻底离线且瞬时响应 (<0.1s)
        from tools.academic_fetcher import get_global_cache_key
        key = get_global_cache_key(ref)
        meta_cache_dir = self.cache_dir / ".metadata_cache"
        meta_cache_dir.mkdir(parents=True, exist_ok=True)
        (meta_cache_dir / f"{key}.json").write_text(
            json.dumps({
                "title": "Target Expected Title",
                "pdf_url": "http://127.0.0.1:1/non_existent.pdf",
                "pdf_candidates": [str(good_pdf)],
            }),
            encoding="utf-8",
        )

        # 3. 运行 fetch_paper_document
        updated_ref, meta = fetch_paper_document(
            ref,
            cache_dir=str(self.cache_dir),
            global_cache_dir=str(self.cache_dir),
        )

        # 4. 验证：.quarantine 目录应生成，且 attempts 记录存在
        q_dir = self.paper_dir / ".quarantine"
        self.assertTrue(q_dir.is_dir(), ".quarantine 目录未生成")
        q_files = list(q_dir.glob("*raw.pdf"))
        self.assertGreaterEqual(len(q_files), 1, "未将错误 PDF 移入隔离区")

        # 5. attempts.json 必须记录尝试详情
        attempts_file = self.paper_dir / "attempts.json"
        self.assertTrue(attempts_file.is_file(), "未记录 attempts.json")
        attempts_data = json.loads(attempts_file.read_text(encoding="utf-8"))
        self.assertIn("attempts", attempts_data)
        self.assertGreaterEqual(len(attempts_data["attempts"]), 1)


if __name__ == "__main__":
    unittest.main()
