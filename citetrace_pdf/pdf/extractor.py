from __future__ import annotations

import logging
import re
from pathlib import Path

import pymupdf
try:
    import pymupdf4llm
except ImportError:
    pymupdf4llm = None

from ..model.layout import LayoutBlock, PageLayout
from ..parser.cleaner import clean_dvips_text, normalize_text
from .table_reconstructor import reconstruct_table, reconstruct_tables_in_markdown

logger = logging.getLogger(__name__)


class PDFExtractor:
    """Extract text/table blocks and title while retaining page geometry, with layout-aware deep learning acceleration."""

    FIGURE_LABEL = re.compile(r"^(?:Figure|Fig\.|Table)\s*\d+", re.I)
    _BOGUS_TITLE = re.compile(r"(?:\.(?:dvi|tex|ps|pdf)$|^untitled|^arxiv:)", re.I)

    def extract(self, pdf_path: str | Path) -> tuple[str, int, list[PageLayout]]:
        with pymupdf.open(pdf_path) as doc:
            is_type3 = self._is_type3_dvips_doc(doc)
            title = self.extract_title(doc, is_type3=is_type3)
            if is_type3:
                pages = [self._extract_type3_page(page, index + 1) for index, page in enumerate(doc)]
                return title, len(doc), pages
            try:
                pages = self._extract_pages_with_layout(doc)
            except Exception as exc:
                logger.warning("Deep-learning layout extraction failed (%s), falling back to heuristic", exc)
                pages = [self.extract_page(page, index + 1) for index, page in enumerate(doc)]
            return title, len(doc), pages

    @staticmethod
    def _is_type3_dvips_doc(doc: pymupdf.Document) -> bool:
        """Detect legacy dvips Type-3 bitmap PDFs (e.g. 1997 Neural Computation / NIPS) where font size is < 1.0 and digits are 0x01-0x09."""
        if not len(doc):
            return False
        d = doc[0].get_text("dict")
        type3_small = 0
        total_spans = 0
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for l in b.get("lines", []):
                for s in l.get("spans", []):
                    txt = s.get("text", "").strip()
                    if not txt:
                        continue
                    total_spans += 1
                    font = str(s.get("font", ""))
                    size = float(s.get("size", 10.0))
                    if (font.startswith("Type3") and size < 1.0) or any(1 <= ord(c) <= 8 for c in txt):
                        type3_small += 1
        return total_spans > 0 and (type3_small / total_spans) >= 0.3

    @classmethod
    def _extract_type3_page(cls, page: pymupdf.Page, page_no: int) -> PageLayout:
        """Reconstruct visual lines for Type-3 dvips PDFs by grouping spans at the same y0 and joining intra-word kerning splits."""
        d = page.get_text("dict")
        raw_spans: list[tuple[float, float, float, float, str]] = []
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for l in b.get("lines", []):
                x0, y0, x1, y1 = map(float, l["bbox"])
                txt = clean_dvips_text("".join(s.get("text", "") for s in l.get("spans", [])))
                if txt:
                    raw_spans.append((x0, y0, x1, min(y1, y0 + 12.0), txt))
        raw_spans.sort(key=lambda s: (round(s[1] / 4.0) * 4.0, s[0]))
        visual_lines: list[tuple[float, float, float, float, str]] = []
        for x0, y0, x1, y1, txt in raw_spans:
            if visual_lines and abs(y0 - visual_lines[-1][1]) < 4.5:
                px0, py0, px1, py1, ptxt = visual_lines[-1]
                sep = " " if (x0 - px1) >= 1.5 else ""
                visual_lines[-1] = (min(px0, x0), py0, max(px1, x1), max(py1, y1), ptxt + sep + txt)
            else:
                visual_lines.append((x0, y0, x1, y1, txt))
        blocks: list[LayoutBlock] = []
        for x0, y0, x1, y1, txt in visual_lines:
            txt = txt.strip()
            if not txt or re.fullmatch(r"\d{1,3}", txt):
                continue
            blocks.append(LayoutBlock(txt, (x0, y0, x1, y1), page_no))
        return PageLayout(page_no, float(page.rect.width), float(page.rect.height), blocks)

    def _extract_pages_with_layout(self, doc: pymupdf.Document) -> list[PageLayout]:
        if pymupdf4llm is None:
            raise ImportError("pymupdf4llm is not installed")

        chunks = pymupdf4llm.to_markdown(doc, page_chunks=True)
        if not chunks or len(chunks) != len(doc):
            raise ValueError(f"pymupdf4llm produced {len(chunks)} chunks for {len(doc)} pages")

        pages: list[PageLayout] = []
        for pno, chunk in enumerate(chunks, 1):
            mpage = doc[pno - 1]
            page_width = float(mpage.rect.width)
            page_height = float(mpage.rect.height)
            page_boxes = chunk.get("page_boxes", [])
            ptext = chunk.get("text", "")

            if not page_boxes:
                pages.append(self.extract_page(mpage, pno))
                continue

            blocks: list[LayoutBlock] = []
            for b in page_boxes:
                cls = b.get("class", "text")
                if cls in ("page-header", "page-footer"):
                    continue

                bbox = tuple(map(float, b.get("bbox", (0, 0, 0, 0))))
                start, end = b.get("pos", (0, 0))
                raw = ptext[start:end].strip()

                # 公式补救：如果公式位置未渲染出文本，直接从 PDF 页面选区提取
                if cls == "formula" and not raw:
                    raw = mpage.get_text("text", clip=pymupdf.Rect(bbox)).strip()

                if not raw:
                    continue

                if "|" in raw and "---" in raw:
                    raw = reconstruct_tables_in_markdown(raw, pages=mpage, bbox=bbox)
                is_table = cls == "table" or (raw.startswith("|") and "---" in raw)
                kind = "table" if is_table else "text"

                if kind == "text":
                    # 行内 References / 致谢切分保护
                    subparts = re.split(r"(?<=\.\s)(?=(?:\*\*)?(?:References|Bibliography)\b)", raw)
                    if len(subparts) > 1:
                        for subpart in subparts:
                            subpart = subpart.strip()
                            m = re.match(r"^(\*{0,2}(?:References|Bibliography)\*{0,2})\s*(.*)$", subpart, re.DOTALL)
                            if m:
                                blocks.append(LayoutBlock(text="References", bbox=bbox, page=pno, kind="text"))
                                if m.group(2).strip():
                                    blocks.append(LayoutBlock(text=m.group(2).strip(), bbox=bbox, page=pno, kind="text"))
                            else:
                                blocks.append(LayoutBlock(text=subpart, bbox=bbox, page=pno, kind="text"))
                        continue

                    m_lead = re.match(r"^(\*{0,2}(?:References|Bibliography)\*{0,2})\s*(\[\d+\].*)$", raw, re.DOTALL)
                    if m_lead:
                        blocks.append(LayoutBlock(text="References", bbox=bbox, page=pno, kind="text"))
                        blocks.append(LayoutBlock(text=m_lead.group(2).strip(), bbox=bbox, page=pno, kind="text"))
                        continue

                blocks.append(LayoutBlock(text=raw, bbox=bbox, page=pno, kind=kind))

            pages.append(PageLayout(pno, page_width, page_height, blocks))

        return pages

    @classmethod
    def extract_title(cls, doc: pymupdf.Document, *, is_type3: bool = False) -> str:
        if not len(doc):
            return (doc.metadata or {}).get("title", "").strip()
        page = doc[0]
        if is_type3:
            layout = cls._extract_type3_page(page, 1)
            for b in layout.blocks:
                if b.y0 < page.rect.height * 0.25 and len(b.text) >= 4:
                    if not re.search(r"arXiv:|http|doi|Computation|Proceedings|Journal|Transactions", b.text, re.I):
                        return b.text
        max_size = 0.0
        candidates: list[str] = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    bbox = span.get("bbox", [0, 0, 0, 0])
                    size = span.get("size", 0.0)
                    if not text or bbox[1] >= page.rect.height * 0.50:
                        continue
                    if re.search(r"arXiv:|http|doi|permission|published as|proceedings", text, re.I):
                        continue
                    if size > max_size + 1.0:
                        max_size, candidates = size, [text]
                    elif abs(size - max_size) < 1.0 and max_size > 13.0:
                        candidates.append(text)
        title = normalize_text(" ".join(candidates))
        if not title or cls._BOGUS_TITLE.search(title):
            meta_title = normalize_text((doc.metadata or {}).get("title", ""))
            if meta_title and not cls._BOGUS_TITLE.search(meta_title):
                title = meta_title
        return title

    def extract_page(self, page: pymupdf.Page, page_no: int) -> PageLayout:
        tables, table_boxes = self._extract_tables(page)
        raw_blocks = page.get_text("blocks")
        figure_boxes = [tuple(block[:4]) for block in raw_blocks if block[6] == 1]
        blocks = list(tables)
        for raw in raw_blocks:
            if raw[6] != 0:
                continue
            text = normalize_text(raw[4])
            if not text or re.fullmatch(r"\d{1,3}", text) or len(text) < 3:
                continue
            x0, y0, x1, y1 = map(float, raw[:4])
            center_x, center_y = (x0 + x1) / 2.0, (y0 + y1) / 2.0
            in_figure = any(fx0 - 10 < center_x < fx1 + 10 and fy0 - 10 < center_y < fy1 + 10
                            for fx0, fy0, fx1, fy1 in figure_boxes)
            if in_figure and not self.FIGURE_LABEL.match(text):
                continue
            overlaps_table = any(max(x0, tx0) < min(x1, tx1) and max(y0, ty0) < min(y1, ty1)
                                 for tx0, ty0, tx1, ty1 in table_boxes)
            if not overlaps_table:
                blocks.append(LayoutBlock(text, (x0, y0, x1, y1), page_no))
        return PageLayout(page_no, float(page.rect.width), float(page.rect.height), blocks)

    @staticmethod
    def _extract_tables(page: pymupdf.Page) -> tuple[list[LayoutBlock], list[tuple[float, float, float, float]]]:
        tables: list[LayoutBlock] = []
        boxes: list[tuple[float, float, float, float]] = []
        try:
            found = page.find_tables()
            for table in found.tables:
                rows = table.extract()
                if not rows or len(rows) < 2 or len(rows[0]) < 2:
                    continue
                cells = [str(cell).strip() for row in rows for cell in row if cell and str(cell).strip()]
                if not cells or sum(any(char.isalpha() for char in cell) for cell in cells) / len(cells) < 0.30:
                    continue
                if sum(map(len, cells)) / len(cells) < 4:
                    continue
                bbox = tuple(map(float, table.bbox))
                boxes.append(bbox)
                newline = chr(10)
                md_rows = ["| " + " | ".join(str(cell or "").strip().replace(newline, " ") for cell in row) + " |"
                           for row in rows]
                md_rows.insert(1, "| " + " | ".join("---" for _ in rows[0]) + " |")
                tables.append(LayoutBlock(reconstruct_table(newline.join(md_rows), pages=page, bbox=bbox), bbox, page.number + 1, "table"))
        except Exception as exc:
            logger.warning("Table extraction failed on page %s: %s", page.number + 1, exc)
        return tables, boxes
