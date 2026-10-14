#!/usr/bin/env python3
"""CiteTrace 全量 PDF 产物审计脚本 (PDF Artifacts Quality Audit Script)

对落地论文目录自动化全面核查：
1. raw.pdf 是否有效（大小、魔数、PyMuPDF打开）
2. metadata.json 是否有效（JSON合法、核心键值完备）
3. chunks.json 是否有效（非空列表、ID唯一、内容非空、页码未越界）
4. PDF SHA-256 是否强一致
5. 标题是否可信（防占位符、防字母碎裂乱码）
6. 产物契约版本是否统一（parser_version, chunker_version, artifact_schema_version）
7. outline_audit 是否存在且 passed
8. section 文件是否齐全
9. chunk ID 是否全局唯一
10. chunk 文字是否非空
11. 页码是否在有效文档区间
12. 错误论文与空正文论文统计
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import pymupdf

from citetrace_pdf.pdf.loader import PDFLoader
from citetrace_pdf.pdf.title_evaluator import evaluate_title_quality
from citetrace_pdf.processor import PDFProcessor
from citetrace_pdf.storage.cache import MetadataCache

logger = logging.getLogger("artifact_auditor")


@dataclass
class PaperAuditReport:
    paper_id: str
    folder: str
    passed: bool
    valid_pdf: bool = False
    valid_metadata: bool = False
    valid_chunks: bool = False
    pdf_sha_matched: bool = False
    title_credible: bool = False
    versions_unified: bool = False
    outline_audit_passed: bool = False
    sections_intact: bool = False
    chunk_ids_unique: bool = False
    chunks_non_empty: bool = False
    pages_in_bounds: bool = False
    issues: List[str] = field(default_factory=list)
    chunks_count: int = 0
    total_pages: int = 0
    paper_title: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def audit_single_paper(paper_dir: Path | str) -> PaperAuditReport:
    """对单篇论文专属产物目录进行全面契约审计"""
    folder = Path(paper_dir)
    paper_id = folder.name
    report = PaperAuditReport(paper_id=paper_id, folder=str(folder), passed=False)

    if not folder.is_dir():
        report.issues.append("folder_not_found")
        return report

    # 1. raw.pdf 校验
    raw_pdf = folder / "raw.pdf"
    total_pages = 0
    if not raw_pdf.is_file():
        report.issues.append("missing_raw_pdf")
    else:
        if raw_pdf.stat().st_size < PDFLoader.DEFAULT_MIN_SIZE:
            report.issues.append("undersized_pdf")
        else:
            try:
                with open(raw_pdf, "rb") as f:
                    magic = f.read(5)
                if magic != b"%PDF-":
                    report.issues.append("invalid_pdf_magic")
                else:
                    with pymupdf.open(raw_pdf) as doc:
                        total_pages = doc.page_count
                        if total_pages < 1:
                            report.issues.append("empty_pdf_pages")
                        else:
                            report.valid_pdf = True
                            report.total_pages = total_pages
            except Exception as e:
                report.issues.append(f"corrupt_pdf: {e}")

    # 2. metadata.json 校验
    meta_path = folder / "metadata.json"
    metadata: Dict[str, Any] = {}
    if not meta_path.is_file():
        report.issues.append("missing_metadata")
    else:
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict):
                report.issues.append("invalid_metadata_format")
            else:
                report.paper_title = metadata.get("paper_title", "")
                req_keys = ["paper_id", "paper_title", "outline"]
                if all(k in metadata for k in req_keys) and report.paper_title:
                    report.valid_metadata = True
                else:
                    report.issues.append("incomplete_metadata_keys")
        except Exception as e:
            report.issues.append(f"corrupt_metadata_json: {e}")

    # 3. 产物契约版本统一校验
    if report.valid_metadata:
        p_ver = metadata.get("parser_version")
        c_ver = metadata.get("chunker_version")
        s_ver = metadata.get("artifact_schema_version")
        if (
            p_ver == PDFProcessor.PARSER_VERSION
            and c_ver == PDFProcessor.CHUNKER_VERSION
            and s_ver == PDFProcessor.ARTIFACT_SCHEMA_VERSION
        ):
            report.versions_unified = True
        else:
            report.issues.append(
                f"version_mismatch: parser={p_ver}(exp {PDFProcessor.PARSER_VERSION}), "
                f"chunker={c_ver}(exp {PDFProcessor.CHUNKER_VERSION}), "
                f"schema={s_ver}(exp {PDFProcessor.ARTIFACT_SCHEMA_VERSION})"
            )

    # 4. PDF SHA-256 强一致校验
    if report.valid_pdf and report.valid_metadata:
        try:
            h = hashlib.sha256()
            with open(raw_pdf, "rb") as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b""):
                    h.update(chunk)
            actual_sha = h.hexdigest().lower()
            meta_sha = (metadata.get("pdf_sha256") or "").strip().lower()
            if meta_sha and meta_sha == actual_sha:
                report.pdf_sha_matched = True
            else:
                report.issues.append(f"pdf_sha256_mismatch: meta={meta_sha} vs disk={actual_sha}")
        except Exception as e:
            report.issues.append(f"sha256_computation_failed: {e}")

    # 5. 标题可信度校验
    if report.valid_metadata and report.paper_title:
        title_eval = evaluate_title_quality(report.paper_title)
        if not title_eval.is_broken and len(report.paper_title.strip()) >= 5:
            report.title_credible = True
        else:
            report.issues.append(
                f"unreliable_title: '{report.paper_title}' (reasons={title_eval.reasons})"
            )

    # 6. outline_audit 存在且通过
    if report.valid_metadata:
        oa = metadata.get("outline_audit")
        if isinstance(oa, dict) and oa.get("passed") is True:
            report.outline_audit_passed = True
        else:
            report.issues.append("outline_audit_missing_or_failed")

    # 7. section 文件齐全性
    if report.valid_metadata:
        outline = metadata.get("outline", [])
        missing_sections = []
        if isinstance(outline, list):
            for sec in outline:
                fname = sec.get("file")
                if fname:
                    p1 = folder / fname
                    p2 = folder / "sections" / fname
                    if not p1.is_file() and not p2.is_file():
                        missing_sections.append(fname)
        if not missing_sections:
            report.sections_intact = True
        else:
            report.issues.append(f"missing_sections: {missing_sections}")

    # 8. chunks.json 校验 (非空、唯一ID、非空文本、页码不越界)
    chunks_path = folder / "chunks.json"
    if not chunks_path.is_file():
        report.issues.append("missing_chunks_json")
    else:
        try:
            chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
            if not isinstance(chunks, list) or len(chunks) == 0:
                report.issues.append("empty_chunks_list")
            else:
                report.chunks_count = len(chunks)
                seen_chunk_ids = set()
                duplicate_ids = set()
                empty_chunks_found = False
                out_of_bounds_pages = []

                effective_max_page = (
                    total_pages if total_pages > 0 else metadata.get("total_pages", 9999)
                )

                for c in chunks:
                    cid = c.get("chunk_id")
                    if cid in seen_chunk_ids:
                        duplicate_ids.add(cid)
                    seen_chunk_ids.add(cid)

                    txt = (c.get("text") or "").strip()
                    if not txt:
                        empty_chunks_found = True

                    sp = c.get("source_page", 1)
                    if sp < 1 or (effective_max_page > 0 and sp > effective_max_page):
                        out_of_bounds_pages.append(sp)

                if not duplicate_ids:
                    report.chunk_ids_unique = True
                else:
                    report.issues.append(f"duplicate_chunk_ids: {list(duplicate_ids)[:5]}")

                if not empty_chunks_found:
                    report.chunks_non_empty = True
                else:
                    report.issues.append("contains_empty_chunk_text")

                if not out_of_bounds_pages:
                    report.pages_in_bounds = True
                else:
                    report.issues.append(f"pages_out_of_bounds: {out_of_bounds_pages[:5]}")

                if (
                    report.chunk_ids_unique
                    and report.chunks_non_empty
                    and report.pages_in_bounds
                ):
                    report.valid_chunks = True
        except Exception as e:
            report.issues.append(f"corrupt_chunks_json: {e}")

    # 判定整体是否通过全量质量契约
    report.passed = (
        report.valid_pdf
        and report.valid_metadata
        and report.valid_chunks
        and report.pdf_sha_matched
        and report.title_credible
        and report.versions_unified
        and report.outline_audit_passed
        and report.sections_intact
    )

    return report


def audit_papers_directory(
    directory: Path | str, pattern: str = "*"
) -> Dict[str, Any]:
    """审计指定目录下的所有论文专属文件夹"""
    base_dir = Path(directory)
    if not base_dir.is_dir():
        return {
            "total_papers": 0,
            "passed_papers": 0,
            "failed_papers": 0,
            "error_papers": 0,
            "empty_text_papers": 0,
            "metrics": {},
            "reports": [],
            "error": f"Directory not found: {base_dir}",
        }

    subdirs = sorted([p for p in base_dir.glob(pattern) if p.is_dir() and not p.name.startswith(".")])

    reports: List[PaperAuditReport] = []
    for d in subdirs:
        # 跳过非论文文件夹 (如 cited_papers 顶层目录等)
        if (d / "metadata.json").is_file() or (d / "raw.pdf").is_file():
            reports.append(audit_single_paper(d))

    total = len(reports)
    passed_count = sum(1 for r in reports if r.passed)
    failed_count = total - passed_count

    valid_pdf_count = sum(1 for r in reports if r.valid_pdf)
    valid_metadata_count = sum(1 for r in reports if r.valid_metadata)
    valid_chunks_count = sum(1 for r in reports if r.valid_chunks)
    pdf_sha_count = sum(1 for r in reports if r.pdf_sha_matched)
    title_credible_count = sum(1 for r in reports if r.title_credible)
    outline_audit_count = sum(1 for r in reports if r.outline_audit_passed)
    unified_parser_count = sum(1 for r in reports if r.versions_unified)
    sections_intact_count = sum(1 for r in reports if r.sections_intact)
    unique_chunk_ids_count = sum(1 for r in reports if r.chunk_ids_unique)
    non_empty_chunks_count = sum(1 for r in reports if r.chunks_non_empty)
    pages_in_bounds_count = sum(1 for r in reports if r.pages_in_bounds)

    error_papers_count = sum(1 for r in reports if not r.valid_pdf or not r.valid_metadata)
    empty_text_papers_count = sum(1 for r in reports if r.chunks_count == 0)

    summary = {
        "total_papers": total,
        "passed_papers": passed_count,
        "failed_papers": failed_count,
        "error_papers": error_papers_count,
        "empty_text_papers": empty_text_papers_count,
        "metrics": {
            "valid_pdf": f"{valid_pdf_count}/{total}",
            "valid_metadata": f"{valid_metadata_count}/{total}",
            "valid_chunks": f"{valid_chunks_count}/{total}",
            "pdf_sha_matched": f"{pdf_sha_count}/{total}",
            "title_credible": f"{title_credible_count}/{total}",
            "outline_audit_passed": f"{outline_audit_count}/{total}",
            "unified_parser_version": f"{unified_parser_count}/{total}",
            "sections_intact": f"{sections_intact_count}/{total}",
            "unique_chunk_ids": f"{unique_chunk_ids_count}/{total}",
            "non_empty_chunks": f"{non_empty_chunks_count}/{total}",
            "pages_in_bounds": f"{pages_in_bounds_count}/{total}",
        },
        "reports": [r.to_dict() for r in reports],
    }
    return summary


def render_terminal_report(summary: Dict[str, Any]) -> str:
    """渲染人类可读的终端审计汇总看板"""
    lines = [
        "=" * 60,
        "CiteTrace 全量 PDF 产物质量审计报告 (Artifact Audit Report)",
        "=" * 60,
        f"总审计论文数量 : {summary['total_papers']} 篇",
        f"全部通过契约   : {summary['passed_papers']} 篇",
        f"存在缺陷论文   : {summary['failed_papers']} 篇",
        "-" * 60,
        "关键指标达成情况：",
    ]
    m = summary.get("metrics", {})
    metrics_map = [
        ("有效 PDF (raw.pdf)", m.get("valid_pdf", "-")),
        ("有效 metadata (metadata.json)", m.get("valid_metadata", "-")),
        ("有效 chunks (chunks.json)", m.get("valid_chunks", "-")),
        ("PDF SHA-256 强一致", m.get("pdf_sha_matched", "-")),
        ("标题质量可信 (防碎裂/占位)", m.get("title_credible", "-")),
        ("outline_audit 审计存在且通过", m.get("outline_audit_passed", "-")),
        ("统一 parser/chunker/schema 版本", m.get("unified_parser_version", "-")),
        ("Section 章节文件齐全", m.get("sections_intact", "-")),
        ("Chunk ID 全局唯一", m.get("unique_chunk_ids", "-")),
        ("Chunk 文字非空", m.get("non_empty_chunks", "-")),
        ("页码在文档区间内", m.get("pages_in_bounds", "-")),
    ]
    for label, val in metrics_map:
        lines.append(f"  * {label:<35}: {val}")

    lines.extend([
        "-" * 60,
        f"错误论文数     : {summary.get('error_papers', 0)}",
        f"空正文论文数   : {summary.get('empty_text_papers', 0)}",
        "=" * 60,
    ])

    failed_reports = [r for r in summary.get("reports", []) if not r["passed"]]
    if failed_reports:
        lines.append("\n缺陷论文详情列表:")
        for r in failed_reports[:20]:
            lines.append(f"  [FAIL] {r['paper_id']}: {', '.join(r['issues'])}")
        if len(failed_reports) > 20:
            lines.append(f"  ... 另有 {len(failed_reports) - 20} 篇未全部展示")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="CiteTrace PDF 全量产物自动化审计工具")
    parser.add_argument("--dir", default="cache/papers", help="论文产物根目录 (默认: cache/papers)")
    parser.add_argument("--paper", help="单篇论文专属目录路径或 ID")
    parser.add_argument("--json", action="store_true", help="输出 JSON 格式报告")
    parser.add_argument("--strict", action="store_true", help="若存在未通过的论文，退出码设为 1")
    args = parser.parse_args()

    if args.paper:
        p_path = Path(args.paper)
        if not p_path.is_dir() and (Path(args.dir) / args.paper).is_dir():
            p_path = Path(args.dir) / args.paper
        report = audit_single_paper(p_path)
        if args.json:
            print(json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            status = "PASS" if report.passed else "FAIL"
            print(f"[{status}] 论文: {report.paper_id} ({report.paper_title})")
            if not report.passed:
                print(f"  缺陷列表: {', '.join(report.issues)}")
        return 0 if report.passed or not args.strict else 1

    summary = audit_papers_directory(args.dir)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(render_terminal_report(summary))

    if args.strict and summary["failed_papers"] > 0:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
