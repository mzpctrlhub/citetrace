"""论文产物契约审计与完整性校验测试 (Artifact Validation)

验证规则：
1. metadata.json 存在但 raw.pdf 缺失/损坏时不得判定成功
2. chunks.json 缺失或空内容时不得判定为有效产物
3. metadata 中记录的 pdf_sha256 与真实 raw.pdf 不一致时判定为缓存失效
4. outline_audit 缺失或未通过 (passed=False) 时判定为无效产物
5. 缓存命中后必须正确更新 ref.status = CitationStatus.FETCHED，更新 ref.pdf_url，清空 ref.last_failure
"""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from models.citation import CitationRef, CitationStatus
from tools.academic_fetcher import (
    validate_paper_artifacts,
    fetch_paper_document,
)


class TestArtifactValidation(unittest.TestCase):
    """测试文献产物完整性与版本校验"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paper_dir = self.root / "papers" / "[1]"
        self.paper_dir.mkdir(parents=True)

        # 构造有效基准 PDF
        self.valid_pdf_content = (
            b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n"
            b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n"
            b"3 0 obj\n<< /Type /Page /Parent 2 0 R >>\nendobj\n"
            + b"A" * 1024
            + b"\nxref\n0 4\n0000000000 65535 f\n"
            b"trailer\n<< /Size 4 /Root 1 0 R >>\nstartxref\n100\n%%EOF\n"
        )
        self.pdf_sha = hashlib.sha256(self.valid_pdf_content).hexdigest()

    def tearDown(self):
        self.tmp.cleanup()

    def _create_minimal_valid_artifacts(self):
        """写入一套完整的有效产物"""
        (self.paper_dir / "raw.pdf").write_bytes(self.valid_pdf_content)
        (self.paper_dir / "metadata.json").write_text(
            json.dumps({
                "paper_id": "test_001",
                "paper_title": "Attention Is All You Need",
                "pdf_sha256": self.pdf_sha,
                "artifact_schema_version": 2,
                "parser_version": "2.1.0",
                "chunker_version": "1.0.0",
                "title_extractor_version": "1.1.0",
                "outline_audit": {
                    "passed": True,
                    "issues": [],
                },
                "outline": [
                    {"index": 0, "title": "Abstract", "file": "00_Abstract.md", "start_page": 1, "end_page": 1}
                ],
            }, ensure_ascii=False),
            encoding="utf-8",
        )
        para_text = "This is a valid section paragraph containing sufficient text content to form chunks properly. " * 5
        (self.paper_dir / "00_Abstract.md").write_text(f"# Abstract\n{para_text}", encoding="utf-8")
        (self.paper_dir / "chunks.json").write_text(
            json.dumps([
                {
                    "chunk_id": "test_001::00_Abstract.md::001",
                    "sequence_index": 0,
                    "content_type": "prose",
                    "text": para_text,
                    "char_count": len(para_text),
                }
            ], ensure_ascii=False),
            encoding="utf-8",
        )

    def test_complete_valid_artifacts_pass_validation(self):
        """完整、一致的产物必须通过校验"""
        self._create_minimal_valid_artifacts()
        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertTrue(audit_result.is_valid, f"有效产物被误判为失败: {audit_result.errors}")

    def test_metadata_only_without_raw_pdf_fails(self):
        """只有 metadata.json 缺少 raw.pdf 时严禁判定为有效"""
        self._create_minimal_valid_artifacts()
        (self.paper_dir / "raw.pdf").unlink()

        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertFalse(audit_result.is_valid, "缺少 raw.pdf 却通过了校验！")
        self.assertIn("missing_raw_pdf", audit_result.errors)

    def test_missing_or_empty_chunks_fails(self):
        """缺少 chunks.json 或 chunks 为空列表时严禁判定为有效"""
        self._create_minimal_valid_artifacts()
        (self.paper_dir / "chunks.json").write_text("[]", encoding="utf-8")

        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertFalse(audit_result.is_valid, "空 chunks 却通过了校验！")
        self.assertIn("empty_chunks", audit_result.errors)

    def test_pdf_sha256_mismatch_fails(self):
        """PDF SHA-256 不一致（说明 PDF 被篡改或不对应）必须判定为失效"""
        self._create_minimal_valid_artifacts()
        # 篡改 PDF
        (self.paper_dir / "raw.pdf").write_bytes(self.valid_pdf_content + b"corrupted")

        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertFalse(audit_result.is_valid, "PDF SHA-256 冲突却通过了校验！")
        self.assertIn("sha256_mismatch", audit_result.errors)

    def test_outline_audit_missing_or_failed_fails(self):
        """缺少 outline_audit 或 outline_audit 判定未通过时严禁判定为有效"""
        self._create_minimal_valid_artifacts()
        meta = json.loads((self.paper_dir / "metadata.json").read_text(encoding="utf-8"))
        meta["outline_audit"] = {"passed": False, "issues": [{"kind": "missing_section"}]}
        (self.paper_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertFalse(audit_result.is_valid, "未通过 outline_audit 却通过了产物校验！")
        self.assertIn("outline_audit_failed", audit_result.errors)

    def test_fetch_paper_document_updates_ref_status_on_cache_hit(self):
        """缓存命中后必须正确更新 ref.status 和 ref.pdf_url，且清除 failure"""
        self._create_minimal_valid_artifacts()
        ref = CitationRef(
            key="[1]",
            raw_text="[1] Vaswani et al. Attention Is All You Need. 2017.",
            title="Attention Is All You Need",
            year=2017,
        )
        updated_ref, meta = fetch_paper_document(
            ref,
            cache_dir=str(self.root / "papers"),
            global_cache_dir=str(self.root / "papers"),
        )
        self.assertEqual(updated_ref.status, CitationStatus.FETCHED)
        self.assertEqual(updated_ref.pdf_url, str(self.paper_dir / "raw.pdf"))
        self.assertIsNone(updated_ref.last_failure)


    def test_outdated_parser_version_invalidates_cache(self):
        """parser_version 旧版本（如 2.0.0）必须判定为版本不匹配失效"""
        self._create_minimal_valid_artifacts()
        meta = json.loads((self.paper_dir / "metadata.json").read_text(encoding="utf-8"))
        meta["parser_version"] = "2.0.0"
        (self.paper_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertFalse(audit_result.is_valid)
        self.assertIn("parser_version_mismatch", audit_result.errors)

    def test_outdated_artifact_schema_version_invalidates_cache(self):
        """artifact_schema_version 缺失或旧版必须判定为模式不匹配失效"""
        self._create_minimal_valid_artifacts()
        meta = json.loads((self.paper_dir / "metadata.json").read_text(encoding="utf-8"))
        meta.pop("artifact_schema_version", None)
        (self.paper_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        audit_result = validate_paper_artifacts(self.paper_dir)
        self.assertFalse(audit_result.is_valid)
        self.assertIn("artifact_schema_version_mismatch", audit_result.errors)

    def test_outdated_chunker_version_triggers_auto_rebuild(self):
        """chunker_version 变动时，fetch_paper_document 必须自动补齐/重建 chunks 并更新版本"""
        self._create_minimal_valid_artifacts()
        meta = json.loads((self.paper_dir / "metadata.json").read_text(encoding="utf-8"))
        meta["chunker_version"] = "0.9.0"
        (self.paper_dir / "metadata.json").write_text(json.dumps(meta), encoding="utf-8")

        ref = CitationRef(
            key="[1]",
            raw_text="[1] Vaswani et al. Attention Is All You Need. 2017.",
            title="Attention Is All You Need",
            year=2017,
        )
        updated_ref, meta_out = fetch_paper_document(
            ref,
            cache_dir=str(self.root / "papers"),
            global_cache_dir=str(self.root / "papers"),
        )
        self.assertEqual(meta_out.get("chunker_version"), "1.0.0")


if __name__ == "__main__":
    unittest.main()
