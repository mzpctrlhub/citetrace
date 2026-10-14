from __future__ import annotations

import json
import re
from pathlib import Path

from ..model.document import Document, Section


class MarkdownWriter:
    KEEP_ROOT_MDS = {"verification_report.md", "extracted_claims.md", "atomic_claims_review.md", "readme.md"}

    def write(self, document: Document, paper_dir: Path, version_fields: dict[str, object]) -> dict:
        paper_dir.mkdir(parents=True, exist_ok=True)
        sections_dir = paper_dir / "sections"
        sections_dir.mkdir(parents=True, exist_ok=True)
        for old_markdown in sections_dir.glob("*.md"):
            old_markdown.unlink(missing_ok=True)
        for legacy_md in paper_dir.glob("*.md"):
            if legacy_md.name.lower() not in self.KEEP_ROOT_MDS:
                legacy_md.unlink(missing_ok=True)

        section_meta = []
        for index, section in enumerate(document.sections):
            section_meta.append({
                "index": index,
                "title": section.title,
                "file": self.section_filename(section.title, index),
                "start_page": section.start_page,
                "end_page": section.end_page,
                "section_id": section.number or f"section-{index:03d}",
            })

        outline = []
        for index, (section, entry) in enumerate(zip(document.sections, section_meta)):
            body = self._with_child_guide(section, index, section_meta)
            path = sections_dir / entry["file"]
            with path.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(f"# {section.title}\n\n")
                stream.write(f"> **所属论文**: {document.paper_title} ({document.paper_id})\n")
                stream.write(f"> **对应页码**: 第 {section.start_page} ~ {section.end_page} 页\n\n")
                if not body.startswith("> **章节类型"):
                    stream.write("---\n\n")
                stream.write(body + "\n")
            outline.append({**entry, "char_count": len(body)})

        metadata = {
            "paper_id": document.paper_id,
            "paper_title": document.paper_title,
            "total_pages": document.total_pages,
            "folder": str(paper_dir.resolve()),
            "outline": outline,
            **version_fields,
        }
        meta_path = paper_dir / "metadata.json"
        temporary = meta_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(meta_path)
        return metadata

    @staticmethod
    def _with_child_guide(section: Section, index: int, entries: list[dict]) -> str:
        body = section.content.strip()
        children = []
        num_match = re.match(r"^(?:Appendix\s+)?((?:\d+|[A-G])(?:\.\d+)*)\.?\s+", section.title, re.I)
        if num_match:
            prefix = re.escape(num_match.group(1))
            children = [
                entry for idx, entry in enumerate(entries)
                if idx != index and re.match(rf"^(?:Appendix\s+)?{prefix}\.\d+(?:\.|\s|$)", entry["title"], re.I)
            ]
        elif re.match(r"^Appendi(?:x|ces)$", section.title.strip(), re.I):
            children = [
                entry for idx, entry in enumerate(entries)
                if idx > index and re.match(r"^(?:Appendix\s+[A-G0-9]+|[A-G](?:\.\d+)*)(?:[.:]\s*|\s+)", entry["title"], re.I)
            ]
        if not children:
            return body
        links = "\n".join(f"- [{child['title']}]({child['file']})" for child in children)
        if len(body) < 80:
            return ("> **章节类型**: 导引章节（本章节核心内容由以下子章节具体展开）\n\n"
                    f"---\n\n本章节具体内容请查阅各子章节：\n\n{links}")
        return f"{body}\n\n---\n\n### 本章子章节导引\n\n{links}"

    @staticmethod
    def section_filename(title: str, fallback_index: int) -> str:
        clean = re.sub(r'[\\/:*?"<>|]', "", title).strip()
        if re.match(r"^(Abstract|Title & Abstract|Overview)\b", clean, re.I):
            return "00_Abstract.md"
        if re.match(r"^References\b", clean, re.I):
            return "References.md"
        if clean.lower() == "other":
            return "other.md"
        appendix = re.match(r"^(?:Appendix\s+([A-G0-9]+(?:\.\d+)*)[.:]?\s*|([A-G](?:\.\d+)*)(?:[.:]\s*|\s+))(.+)", clean, re.I)
        if appendix:
            label = (appendix.group(1) or appendix.group(2)).upper()
            name = re.sub(r"\s+", "_", appendix.group(3).strip())[:40].strip("_")
            return f"Appendix_{label}_{name}.md"
        subsection = re.match(r"^(\d+\.(?:\d+(?:\.\d+)*))\.?\s*(.+)", clean)
        if subsection:
            number, name = subsection.groups()
            padded = f"{int(number.split('.')[0]):02d}." + ".".join(number.split('.')[1:])
            slug = re.sub(r"\s+", "_", name.strip())[:40].strip("_")
            return f"{padded}_{slug}.md"
        major = re.match(r"^(\d+)\.?\s*(.+)", clean)
        if major:
            slug = re.sub(r"\s+", "_", major.group(2).strip())[:40].strip("_")
            return f"{int(major.group(1)):02d}_{slug}.md"
        slug = re.sub(r"\s+", "_", clean)[:40].strip("_") or "section"
        return f"{fallback_index:02d}_{slug}.md"
