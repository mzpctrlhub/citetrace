from __future__ import annotations

import re
import shutil
import logging
from pathlib import Path

import hashlib
import pymupdf
from typing import Any, Optional

from .chunker import chunk_paper_folder
from .model.document import Document
from .parser.auditor import OutlineAuditor
from .parser.heading import HeadingDetector
from .parser.inspector import PageInspector
from .parser.referee import LLMSkeletonReferee
from .parser.section import SectionParser
from .pdf.extractor import PDFExtractor
from .pdf.loader import PDFLoader
from .storage.cache import MetadataCache
from .storage.writer import MarkdownWriter

logger = logging.getLogger(__name__)


def sanitize_paper_folder_name(name: str) -> str:
    """清洗论文名称/标题为安全易读的文件夹名称，例如 'Layer Normalization' -> 'Layer_Normalization'，保留 [1] 等编号"""
    m_bracket = re.match(r"^\[\d+\]$", name.strip())
    if m_bracket:
        return m_bracket.group(0)
    clean = re.sub(r'[/\\:*?"<>|]', '_', name)
    clean = re.sub(r'\s+', '_', clean)
    clean = re.sub(r'_+', '_', clean).strip("._")
    return clean or "unnamed_paper"


class PDFProcessor:
    """Public orchestration API, retaining the original ``process_paper`` entry point."""

    ARTIFACT_SCHEMA_VERSION = 2
    PARSER_VERSION = "2.1.0"
    CHUNKER_VERSION = "1.0.0"
    TITLE_EXTRACTOR_VERSION = "1.1.0"
    REFERENCE_EXTRACTOR_VERSION = "1.0.0"

    def __init__(self, cache_dir: str | Path = "cache/papers") -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.loader = PDFLoader(self.cache_dir)
        self.extractor = PDFExtractor()
        self.section_parser = SectionParser(HeadingDetector())
        self.cache = MetadataCache(
            parser_version=self.PARSER_VERSION,
            chunker_version=self.CHUNKER_VERSION,
            artifact_schema_version=self.ARTIFACT_SCHEMA_VERSION,
            title_extractor_version=self.TITLE_EXTRACTOR_VERSION,
            reference_extractor_version=self.REFERENCE_EXTRACTOR_VERSION,
        )
        self.writer = MarkdownWriter()

    def process_paper(
        self,
        paper_id: str,
        pdf_url: str,
        force: bool = False,
        paper_name: Optional[str] = None,
        expected_title: Optional[str] = None,
    ) -> dict[str, Any]:
        folder_name = sanitize_paper_folder_name(paper_name or paper_id)
        paper_dir = self.cache_dir / folder_name
        paper_dir.mkdir(parents=True, exist_ok=True)
        metadata_path = paper_dir / "metadata.json"
        if not force:
            cached = self.cache.read_valid(metadata_path)
            if cached is not None:
                if not (paper_dir / "chunks.json").is_file():
                    chunk_paper_folder(paper_dir)
                logger.info("Using parser-versioned cache for %s", folder_name)
                return cached

        pdf_path = self.loader.load(folder_name, pdf_url)
        paper_title, total_pages, pages = self.extractor.extract(pdf_path)

        doc_meta_title = ""
        try:
            with pymupdf.open(pdf_path) as doc:
                doc_meta_title = (doc.metadata or {}).get("title", "")
        except Exception:
            pass

        from citetrace_pdf.pdf.title_evaluator import resolve_best_title
        title_res = resolve_best_title(
            extracted_title=paper_title,
            expected_title=expected_title,
            meta_title=doc_meta_title,
        )
        resolved_title = title_res["paper_title"]

        # Step 1: 常规快速切分
        sections = self.section_parser.parse(pages)

        # Step 2: OutlineAuditor 大纲结构自检
        audit_report = OutlineAuditor.audit(sections, total_pages, resolved_title)
        initial_issue_count = len(audit_report.issues)
        repair_stage = "none"

        # Step 3 & 4: 若自检未通过，依次触发 PageInspector 本地底层透视自愈 -> LLMSkeletonReferee 骨架终审
        if not audit_report.passed:
            try:
                with pymupdf.open(pdf_path) as doc:
                    sections, audit_report = PageInspector.auto_repair(doc, sections, audit_report)
                    repair_stage = "page_inspector"
                    if not audit_report.passed:
                        sections, audit_report = LLMSkeletonReferee.adjudicate_and_split(
                            doc, sections, audit_report
                        )
                        repair_stage = "llm_skeleton_referee"
            except Exception as exc:
                logger.warning("Agentic PDF repair skipped on %s: %s", folder_name, exc)

        document = Document(
            paper_id=paper_id,
            paper_title=resolved_title or paper_id,
            total_pages=total_pages,
            sections=sections,
        )
        pdf_sha = None
        if pdf_path.is_file():
            try:
                h = hashlib.sha256()
                with open(pdf_path, "rb") as f:
                    for block in iter(lambda: f.read(1 << 20), b""):
                        h.update(block)
                pdf_sha = h.hexdigest()
            except Exception:
                pass

        extra_fields = {
            **self.cache.version_fields,
            "pdf_sha256": pdf_sha,
            "paper_title": resolved_title,
            "extracted_pdf_title": title_res["extracted_pdf_title"],
            "title_source": title_res["title_source"],
            "title_confidence": title_res["title_confidence"],
            "title_quality": title_res["title_quality"],
            "outline_audit": {
                "passed": audit_report.passed,
                "initial_issues": initial_issue_count,
                "repair_stage": repair_stage,
                "remaining_issues": [iss.message for iss in audit_report.issues],
            },
        }
        metadata = self.writer.write(document, paper_dir, extra_fields)
        metadata["paper_name"] = folder_name
        chunk_paper_folder(paper_dir)
        logger.info("Wrote %d section files for %s (audit_passed=%s)", len(sections), folder_name, audit_report.passed)
        return metadata

    def process_local_pdf(
        self,
        pdf_path: str | Path,
        paper_id: Optional[str] = None,
        paper_name: Optional[str] = None,
        force: bool = False,
        expected_title: Optional[str] = None,
    ) -> dict[str, Any]:
        """直接解析本地 PDF 文件并落地为专属文献夹（以论文标题/名称命名）"""
        local_file = Path(pdf_path).resolve()
        if not local_file.is_file():
            raise FileNotFoundError(f"Local PDF file not found: {local_file}")

        extracted_title = ""
        if not paper_name:
            try:
                with pymupdf.open(local_file) as doc:
                    is_t3 = PDFExtractor._is_type3_dvips_doc(doc)
                    extracted_title = PDFExtractor.extract_title(doc, is_type3=is_t3)
            except Exception:
                pass

        stem = local_file.stem
        pid = paper_id or (local_file.parent.name if stem.lower() in ("raw", "input", "paper") and local_file.parent.name else stem)
        if paper_name:
            chosen_name = paper_name
        elif paper_id:
            chosen_name = paper_id
        elif extracted_title:
            chosen_name = extracted_title
        else:
            chosen_name = pid
        return self.process_paper(
            paper_id=pid,
            pdf_url=str(local_file),
            force=force,
            paper_name=chosen_name,
            expected_title=expected_title,
        )
