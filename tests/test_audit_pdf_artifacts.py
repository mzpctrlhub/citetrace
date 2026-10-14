"""PDF 产物自动化审计契约测试 (Step 17 TDD)

验证目标：
1. audit_single_paper 全项通过有效产物 (PDF, metadata, chunks, sha256, title, versions, outline, sections)
2. 缺失/损坏 PDF, SHA-256 不一致, 字母碎裂标题, 版本过旧等缺陷能被精准检出
3. chunks 列表空、文本空、ID 重复、页码越界能被精准检出
4. audit_papers_directory 目录聚合指标与汇总报表正确计算
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import pymupdf

from citetrace_pdf.processor import PDFProcessor
from scripts.audit_pdf_artifacts import (
    audit_single_paper,
    audit_papers_directory,
    render_terminal_report,
)


class TestAuditPDFArtifacts(unittest.TestCase):

    def _create_mock_valid_paper(self, folder: Path, title: str = "Attention Is All You Need") -> None:
        folder.mkdir(parents=True, exist_ok=True)
        sections_dir = folder / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)

        # 1. 真实生成有效 PDF
        raw_pdf = folder / "raw.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page()
        p1.insert_text((50, 72), "Abstract and Introduction")
        p2 = doc.new_page()
        p2.insert_text((50, 72), "References section")
        pdf_bytes = doc.tobytes()
        doc.close()
        if len(pdf_bytes) < 1000:
            pdf_bytes += b"\n% " + b"P" * (1001 - len(pdf_bytes))
        raw_pdf.write_bytes(pdf_bytes)

        actual_sha = hashlib.sha256(pdf_bytes).hexdigest()

        # 2. 章节文件
        (sections_dir / "01_Intro.md").write_text("# Intro\nText content", encoding="utf-8")
        (sections_dir / "References.md").write_text("# References\n[1] Ref", encoding="utf-8")

        # 3. metadata.json
        meta = {
            "paper_id": folder.name,
            "paper_title": title,
            "pdf_sha256": actual_sha,
            "total_pages": 2,
            "parser_version": PDFProcessor.PARSER_VERSION,
            "chunker_version": PDFProcessor.CHUNKER_VERSION,
            "artifact_schema_version": PDFProcessor.ARTIFACT_SCHEMA_VERSION,
            "outline": [
                {"title": "Intro", "file": "01_Intro.md", "start_page": 1, "end_page": 1},
                {"title": "References", "file": "References.md", "start_page": 2, "end_page": 2},
            ],
            "outline_audit": {
                "passed": True,
                "issues": [],
            },
        }
        (folder / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

        # 4. chunks.json
        chunks = [
            {
                "chunk_id": f"{folder.name}_c001",
                "text": "Valid non-empty chunk text for intro",
                "source_file": "01_Intro.md",
                "source_page": 1,
            },
            {
                "chunk_id": f"{folder.name}_c002",
                "text": "Valid non-empty chunk text for references",
                "source_file": "References.md",
                "source_page": 2,
            },
        ]
        (folder / "chunks.json").write_text(json.dumps(chunks, indent=2), encoding="utf-8")

    def test_audit_passes_on_complete_valid_paper(self):
        """测试完整合规的论文产物顺利通过全部审计契约"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paper_dir = Path(tmp_dir) / "valid_paper"
            self._create_mock_valid_paper(paper_dir)

            report = audit_single_paper(paper_dir)
            self.assertTrue(report.passed, f"合规论文应当通过审计，但报错: {report.issues}")
            self.assertTrue(report.valid_pdf)
            self.assertTrue(report.valid_metadata)
            self.assertTrue(report.valid_chunks)
            self.assertTrue(report.pdf_sha_matched)
            self.assertTrue(report.title_credible)
            self.assertTrue(report.versions_unified)
            self.assertTrue(report.outline_audit_passed)
            self.assertTrue(report.sections_intact)
            self.assertEqual(len(report.issues), 0)

    def test_audit_flags_pdf_sha256_mismatch(self):
        """测试当 raw.pdf 改变后，SHA-256 不一致能被审计器准确捕获"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paper_dir = Path(tmp_dir) / "tampered_paper"
            self._create_mock_valid_paper(paper_dir)

            # 篡改 raw.pdf 尾部内容
            raw_pdf = paper_dir / "raw.pdf"
            raw_pdf.write_bytes(raw_pdf.read_bytes() + b"\nextra_tampered_bytes")

            report = audit_single_paper(paper_dir)
            self.assertFalse(report.passed)
            self.assertFalse(report.pdf_sha_matched)
            self.assertTrue(any("pdf_sha256_mismatch" in issue for issue in report.issues))

    def test_audit_flags_unreliable_or_fragmented_title(self):
        """测试当标题为碎裂字母乱码或占位符时，能被审计器准确识别"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paper_dir = Path(tmp_dir) / "broken_title_paper"
            # 碎裂标题: "N GPU L A"
            self._create_mock_valid_paper(paper_dir, title="N GPU L A")

            report = audit_single_paper(paper_dir)
            self.assertFalse(report.passed)
            self.assertFalse(report.title_credible)
            self.assertTrue(any("unreliable_title" in issue for issue in report.issues))

    def test_audit_flags_version_mismatch(self):
        """测试旧版本 parser 或 chunker 产物能被审计器准确发现"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paper_dir = Path(tmp_dir) / "legacy_version_paper"
            self._create_mock_valid_paper(paper_dir)

            meta_file = paper_dir / "metadata.json"
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
            meta["parser_version"] = "0.9.0"  # 模拟过旧版本
            meta_file.write_text(json.dumps(meta), encoding="utf-8")

            report = audit_single_paper(paper_dir)
            self.assertFalse(report.passed)
            self.assertFalse(report.versions_unified)
            self.assertTrue(any("version_mismatch" in issue for issue in report.issues))

    def test_audit_flags_empty_or_duplicate_chunks(self):
        """测试 chunks 包含重复 ID 或空文本时准确拦截"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            paper_dir = Path(tmp_dir) / "bad_chunks_paper"
            self._create_mock_valid_paper(paper_dir)

            chunks_file = paper_dir / "chunks.json"
            bad_chunks = [
                {"chunk_id": "dup_01", "text": "Chunk 1", "source_page": 1},
                {"chunk_id": "dup_01", "text": "Chunk 2 duplicate", "source_page": 1},  # 重复 ID
                {"chunk_id": "chunk_empty", "text": "   ", "source_page": 2},            # 空正文
            ]
            chunks_file.write_text(json.dumps(bad_chunks), encoding="utf-8")

            report = audit_single_paper(paper_dir)
            self.assertFalse(report.passed)
            self.assertFalse(report.valid_chunks)
            self.assertFalse(report.chunk_ids_unique)
            self.assertFalse(report.chunks_non_empty)

    def test_audit_papers_directory_aggregation_and_reporting(self):
        """测试多论文目录的聚合统计指标与人类可读报表渲染"""
        with tempfile.TemporaryDirectory() as tmp_dir:
            base_dir = Path(tmp_dir)
            p1 = base_dir / "paper_good"
            p2 = base_dir / "paper_broken"

            self._create_mock_valid_paper(p1)
            self._create_mock_valid_paper(p2)
            (p2 / "raw.pdf").unlink()  # 移除 p2 的 PDF

            summary = audit_papers_directory(base_dir)
            self.assertEqual(summary["total_papers"], 2)
            self.assertEqual(summary["passed_papers"], 1)
            self.assertEqual(summary["failed_papers"], 1)
            self.assertEqual(summary["metrics"]["valid_pdf"], "1/2")

            report_text = render_terminal_report(summary)
            self.assertIn("总审计论文数量 : 2 篇", report_text)
            self.assertIn("有效 PDF (raw.pdf)", report_text)
            self.assertIn("1/2", report_text)
            self.assertIn("[FAIL] paper_broken", report_text)


if __name__ == "__main__":
    unittest.main()
