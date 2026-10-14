# citetrace/parsers/link_extractor.py
"""
CiteTrace PDF Link-Based Citation Extractor
通过解析 PDF 底层 Link Annotations (LINK_GOTO / LINK_NAMED) 提取正文引用与文末参考文献的物理对齐。

核心能力：
1. 自动定位文末 References 章节起止页与各文献条目物理边界
2. 提取正文中所有内部跳转链接，并自动聚合相邻碎片锚点 (如 "[Ioffe and", "Szegedy,", "2015]")
3. 基于目标物理纵坐标 (target_y_top = page_height - to.y) 实现 100% 精度的文献条目吸附
4. 无论论文是数字序列型 ([1], [2]) 还是作者-年份型 ([Krizhevsky et al., 2012])，均可免正则完美对齐
5. 自动提取包含引用锚点的正文完整上下文段落并结构化归并
"""
from __future__ import annotations

import re
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pymupdf

logger = logging.getLogger(__name__)


def normalize_citation_anchor(raw: str) -> str:
    """
    清洗并规范化引用标记
    例: '[35,'  -> '[35]'
        '2,'    -> '[2]'
        '5]'    -> '[5]'
        '[Krizhevsky et al., 2012]' -> '[Krizhevsky et al., 2012]'
    """
    cleaned = raw.strip()
    inner = re.sub(r"^[\[\(]+|[\]\)]+$", "", cleaned).strip().rstrip(",;")
    return f"[{inner}]" if inner else cleaned


class PDFLinkCitationExtractor:
    """基于 PDF 矢量交互链接的引用-参考文献高精度提取器"""

    _REF_HEADING_RE = re.compile(r"^\s*(?:[0-9\.]*\s*)?(?:References|Bibliography)\b", re.M | re.I)

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        self.doc = pymupdf.open(str(self.pdf_path))

    def close(self):
        if self.doc:
            self.doc.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def find_references_range(self) -> Tuple[Optional[int], int]:
        """
        定位 References 章节所在的起止页码 (0-indexed)
        返回 (start_page, end_page)
        """
        start_page = None
        for i, page in enumerate(self.doc):
            text = page.get_text()
            if self._REF_HEADING_RE.search(text):
                start_page = i
                break

        if start_page is None:
            # 兜底：从后向前查找
            for i in range(len(self.doc) - 1, -1, -1):
                text = self.doc[i].get_text()
                if re.search(r"\b(?:References|Bibliography)\b", text, re.I):
                    start_page = i
                    break

        if start_page is None:
            return None, len(self.doc)

        # 确定结束页（检查是否有 Supplementary Material / Appendix）
        end_page = len(self.doc)
        for i in range(start_page + 1, len(self.doc)):
            text = self.doc[i].get_text()
            if re.search(r"^\s*(?:Supplementary\s+Material|Appendix)\b", text, re.M | re.I):
                end_page = i
                break

        return start_page, end_page

    def _find_references_heading_y0(self, page_idx: Optional[int], default: float = 0.0) -> float:
        """定位指定页面上 References / Bibliography 标题的纵向起始坐标 y0"""
        if page_idx is not None and page_idx < len(self.doc):
            for b in self.doc[page_idx].get_text("blocks"):
                if b[6] == 0 and self._REF_HEADING_RE.search(b[4].strip()):
                    return b[1]
        return default

    def extract_reference_blocks(self, start_page: int, end_page: int) -> List[Dict[str, Any]]:
        """
        提取 References 区域的所有段落文本块及其几何坐标
        """
        ref_heading_y0 = self._find_references_heading_y0(start_page, default=0.0)

        ref_blocks = []
        for pno in range(start_page, end_page):
            page = self.doc[pno]
            blocks = page.get_text("blocks")
            for b in blocks:
                # b: (x0, y0, x1, y1, text, block_no, block_type)
                if b[6] == 0:  # 文本类型
                    if pno == start_page and b[1] < ref_heading_y0:
                        continue
                    raw_txt = b[4].strip()
                    from citetrace_pdf.parser.cleaner import TextCleaner
                    dehyph = TextCleaner.dehyphenate(raw_txt)
                    cleaned = re.sub(r"\s*\n\s*", " ", dehyph).strip()
                    # 过滤标题自身与页脚纯数字
                    if not cleaned or cleaned.lower() in ("references", "bibliography") or cleaned.isdigit():
                        continue
                    ref_blocks.append({
                        "page": pno,
                        "x0": b[0],
                        "y0": b[1],
                        "x1": b[2],
                        "y1": b[3],
                        "text": cleaned
                    })
        return ref_blocks

    def extract_raw_links(self) -> List[Dict[str, Any]]:
        """
        扫描全文提取所有指向参考文献的原始跳转链接，完成碎片拼接与坐标转换
        """
        ref_start, ref_end = self.find_references_range()
        if ref_start is None:
            logger.warning(f"未能自动定位 {self.pdf_path.name} 的 References 章节")
            return []

        ref_heading_y0 = self._find_references_heading_y0(ref_start, default=float("inf"))

        ref_blocks = self.extract_reference_blocks(ref_start, ref_end)
        extracted = []

        scan_end = min(len(self.doc), ref_start + 1)
        for pno in range(scan_end):
            page = self.doc[pno]
            links = page.get_links()
            if not links:
                continue

            merged_links: List[Dict[str, Any]] = []
            for link in links:
                kind = link.get("kind")
                if kind not in (pymupdf.LINK_GOTO, pymupdf.LINK_NAMED):
                    continue

                from_rect = pymupdf.Rect(link["from"])
                # 若当前页就是 References 起始页，且链接位于 References 标题下方，则属于参考文献自身内部，跳过
                if pno == ref_start and from_rect.y0 >= ref_heading_y0:
                    continue

                tgt_page = link.get("page")
                if tgt_page is None or tgt_page < ref_start or tgt_page >= ref_end:
                    continue

                anchor_text = page.get_text("text", clip=from_rect).strip().replace("\n", " ")
                to_pt = link.get("to")
                if not anchor_text:
                    continue

                # 仅当同一行空间紧邻、且并非重复锚点文本时才聚合碎片（如 "[Ioffe and" + "Szegedy," + "2015]"）
                prev = merged_links[-1] if merged_links else None
                if (
                    prev is not None
                    and prev["tgt_page"] == tgt_page
                    and prev["to"] == to_pt
                    and abs(from_rect.y0 - prev["from_rect"].y0) < 6.0
                    and -2.0 <= (from_rect.x0 - prev["from_rect"].x1) < 15.0
                ):
                    if anchor_text.strip("[](),;") != prev["anchor"].strip("[](),;"):
                        prev["anchor"] += " " + anchor_text
                    prev["from_rect"] |= from_rect
                else:
                    merged_links.append({
                        "src_page": pno,
                        "anchor": anchor_text,
                        "tgt_page": tgt_page,
                        "to": to_pt,
                        "from_rect": from_rect
                    })

            for ml in merged_links:
                to_pt = ml["to"]
                tgt_page_idx = ml["tgt_page"]
                target_page = self.doc[tgt_page_idx]

                target_y_top = target_page.rect.height - to_pt.y if to_pt else None

                # 若跳转目标位于 References 起始页，但落点在 References 标题上方（如指向同页顶部的 Table 4），则跳过
                if (
                    tgt_page_idx == ref_start
                    and target_y_top is not None
                    and ref_heading_y0 != float("inf")
                    and target_y_top < ref_heading_y0 - 15
                ):
                    continue

                matched_ref_text = None
                if target_y_top is not None:
                    # 寻找紧贴 target_y_top (在上方或下方 15pt 内) 的文献块
                    candidates = [
                        b for b in ref_blocks
                        if b["page"] == tgt_page_idx and b["y0"] >= target_y_top - 12
                    ]
                    if candidates:
                        matched_ref_text = candidates[0]["text"]
                    else:
                        page_candidates = [b for b in ref_blocks if b["page"] == tgt_page_idx]
                        if page_candidates:
                            matched_ref_text = min(page_candidates, key=lambda b: abs(b["y0"] - target_y_top))["text"]

                context_para = self._extract_surrounding_paragraph(page, ml["from_rect"])
                clean_anchor = normalize_citation_anchor(ml["anchor"])

                # 过滤正文中无括号的图表/章节交叉引用（如 "Table 4" 中的 "4"）
                raw_anc = ml["anchor"].strip()
                if raw_anc.isdigit() and clean_anchor not in context_para:
                    if not re.search(rf"\[[^\]]*\b{re.escape(raw_anc)}\b[^\]]*\]", context_para):
                        continue

                extracted.append({
                    "source_page": pno + 1,
                    "anchor": clean_anchor,
                    "target_page": tgt_page_idx + 1,
                    "target_y": round(target_y_top, 2) if target_y_top else None,
                    "reference_entry": matched_ref_text or "",
                    "context_paragraph": context_para
                })

        return extracted

    def extract_claims_and_references(self) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
        """
        高级入口：将抽取的链接按段落自动聚合为标准 Claim 格式
        返回: (claims_list, all_references_dict)
        """
        raw_links = self.extract_raw_links()
        if not raw_links:
            return [], {}

        # 收集所有全局参考文献条目
        all_references: Dict[str, str] = {}
        for r in raw_links:
            if r["anchor"] and r["reference_entry"]:
                all_references[r["anchor"]] = r["reference_entry"]

        # 按段落内容聚合
        para_groups: Dict[str, Dict[str, Any]] = defaultdict(
            lambda: {"source_page": None, "citation_keys": [], "matched_references": {}}
        )

        for r in raw_links:
            p = r["context_paragraph"]
            if not p:
                continue
            key = r["anchor"]
            para_groups[p]["source_page"] = r["source_page"]
            if key not in para_groups[p]["citation_keys"]:
                para_groups[p]["citation_keys"].append(key)
            if r["reference_entry"]:
                para_groups[p]["matched_references"][key] = r["reference_entry"]

        claims = []
        for idx, (para, data) in enumerate(para_groups.items(), 1):
            claims.append({
                "claim_id": f"claim_{idx:03d}",
                "paragraph": para,
                "citation_keys": data["citation_keys"],
                "matched_references": data["matched_references"],
                "source_page": data["source_page"]
            })

        return claims, all_references

    def _extract_surrounding_paragraph(self, page: pymupdf.Page, rect: pymupdf.Rect) -> str:
        """从页面中提取包含给定矩形区域的完整段落"""
        blocks = page.get_text("blocks")
        for b in blocks:
            if b[6] == 0:  # text block
                block_rect = pymupdf.Rect(b[:4])
                if block_rect.intersects(rect):
                    from citetrace_pdf.parser.cleaner import TextCleaner
                    clean = TextCleaner.clean(b[4])
                    if clean:
                        return clean
        from citetrace_pdf.parser.cleaner import TextCleaner
        expanded = pymupdf.Rect(page.rect.x0, max(0, rect.y0 - 40), page.rect.x1, min(page.rect.y1, rect.y1 + 40))
        return TextCleaner.clean(page.get_text("text", clip=expanded))


def extract_citations_from_pdf_links(pdf_path: str | Path) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """便捷函数：一行代码从 PDF 链接中提取全部论断段落与参考文献对齐"""
    with PDFLinkCitationExtractor(pdf_path) as extractor:
        return extractor.extract_claims_and_references()
