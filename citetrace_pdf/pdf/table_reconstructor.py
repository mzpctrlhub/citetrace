from __future__ import annotations

import re
from typing import Any, Sequence

import pymupdf


_NUMERIC_CELL_RE = re.compile(
    r"^(?:\*\*|_)?[+\-\u2212]?\d+(?:\.\d+)?(?:\s*(?:%|\u00b1\s*\d+(?:\.\d+)?|\u00b7\s*10\^\d+|M|B|K|s|days?))?(?:\*\*|_)?$"
)

_MD_TABLE_BLOCK_RE = re.compile(r"(?:^[ \t]*\|.*\|[ \t]*\r?\n?)+", re.M)


def clean_cell_text(cell: str, is_header: bool = False) -> str:
    """Clean rich-text/math fragmentation artifacts inside a Markdown table cell."""
    s = cell.strip()
    s = re.sub(r"^(?:<br\s*/?>\s*)+", "", s, flags=re.I)
    s = re.sub(r"(?:\s*<br\s*/?>)+$", "", s, flags=re.I)
    if is_header:
        s = re.sub(r"\s*<br\s*/?>\s*", " ", s, flags=re.I)
    # Remove empty adjacent bold markers like **3****_._3**
    s = re.sub(r"\*{4,}", "", s)
    # Unwrap italicized punctuation and math operators inside numbers/expressions
    s = re.sub(r"_([.\u00b7\u00b1\u00d7\u2212\u2192+\-/=<>∗*])_", r"\1", s)
    s = re.sub(r"\*\*(\d+)\*\*\.\*\*(\d+)\*\*", r"**\1.\2**", s)

    def _sup_repl(m: re.Match) -> str:
        raw_inner = m.group(1).strip()
        is_bold = raw_inner.startswith("**") and raw_inner.endswith("**")
        inner = re.sub(r"[\*_]", "", raw_inner).strip()
        if not inner:
            return ""
        return f"^{inner}**" if is_bold else f"^{inner}"

    def _sub_repl(m: re.Match) -> str:
        inner = re.sub(r"[\*_]", "", m.group(1)).strip()
        return f"_{inner}" if inner else ""

    s = re.sub(r"<sup>(.*?)</sup>", _sup_repl, s, flags=re.I)
    s = re.sub(r"<sub>(.*?)</sub>", _sub_repl, s, flags=re.I)
    # Merge **10**^18** -> **10^18**
    s = re.sub(r"\*\*([^*]+)\*\*(\^[\d+\-a-zA-Z]+)\*\*", r"**\1\2**", s)
    s = re.sub(r"\*\*([^*]+)\*\*\s*\*\*([^*]+)\*\*", r"**\1 \2**", s)
    s = re.sub(r"(\d)\s*\u00b7\s*(\d)", r"\1 · \2", s)
    s = re.sub(r"(\d)\s*\u00b1\s*(\d)", r"\1 ± \2", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def norm_key(s: str) -> str:
    s = clean_cell_text(s, is_header=True)
    s = re.sub(r"[\*_`~^]", "", s)
    s = re.sub(r"\s+", "", s)
    return s.lower()


def extract_visual_info(
    pages: pymupdf.Page | Sequence[pymupdf.Page] | None,
    bbox: tuple[float, float, float, float] | None = None,
) -> tuple[list[dict[str, Any]], list[tuple[float, float, float]], set[str]]:
    """Extract intact visual phrases, horizontal rules, and word vocabulary from PDF page(s)."""
    phrases: list[dict[str, Any]] = []
    cmidrules: list[tuple[float, float, float]] = []
    pdf_words: set[str] = set()
    if pages is None:
        return phrases, cmidrules, pdf_words
    if isinstance(pages, pymupdf.Page):
        pages = [pages]

    for page in pages:
        clip = (
            pymupdf.Rect(bbox[0] - 6, bbox[1] - 6, bbox[2] + 6, bbox[3] + 6)
            if bbox
            else page.rect
        )
        for w in page.get_text("words", clip=clip):
            wn = norm_key(w[4])
            if wn:
                pdf_words.add(wn)

        d = page.get_text("dict", clip=clip)
        tokens: list[tuple[float, float, float, float, str, float]] = []
        for b in d.get("blocks", []):
            if b.get("type") != 0:
                continue
            for l in b.get("lines", []):
                spans = [s for s in l.get("spans", []) if s.get("text", "").strip()]
                if not spans:
                    continue
                base_sz = max(float(s.get("size", 10.0)) for s in spans)
                base_y0 = min(
                    float(s["bbox"][1])
                    for s in spans
                    if float(s.get("size", 10.0)) >= base_sz * 0.85
                )
                for s in spans:
                    t = s.get("text", "")
                    if not t.strip():
                        continue
                    sz = float(s.get("size", 10.0))
                    sx0, sy0, sx1, sy1 = map(float, s["bbox"])
                    is_sup = (
                        sz < base_sz * 0.84
                        and sy0 < base_y0 - 0.3
                        and bool(re.fullmatch(r"[\d+\-∗*a-zA-Z]+", t.strip()))
                    )
                    txt_out = f"^{t.strip()}" if is_sup else t.strip()
                    tokens.append((sx0, base_y0, sx1, sy1, txt_out, base_sz))

        tokens.sort(key=lambda x: (round(x[1] / 3.0) * 3.0, x[0]))
        merged: list[tuple[float, float, float, float, str, float]] = []
        for tx0, ty0, tx1, ty1, ttxt, bsz in tokens:
            max_gap = max(4.2, bsz * 0.42)
            if (
                merged
                and abs(ty0 - merged[-1][1]) < 3.2
                and -1.5 <= (tx0 - merged[-1][2]) <= max_gap
            ):
                px0, py0, px1, py1, ptxt, pbsz = merged[-1]
                if (
                    ttxt.startswith("^")
                    or (tx0 - px1) < 0.8
                    or ptxt.endswith(("-", "(", "["))
                    or ttxt.startswith((".", ",", ")", "]"))
                ):
                    sep = ""
                else:
                    sep = " "
                merged[-1] = (
                    min(px0, tx0),
                    min(py0, ty0),
                    max(px1, tx1),
                    max(py1, ty1),
                    (ptxt + sep + ttxt).strip(),
                    max(pbsz, bsz),
                )
            else:
                merged.append((tx0, ty0, tx1, ty1, ttxt, bsz))

        for px0, py0, px1, py1, ptxt, _ in merged:
            phrases.append(
                {
                    "x0": px0,
                    "y0": py0,
                    "x1": px1,
                    "y1": py1,
                    "text": clean_cell_text(ptxt, is_header=True),
                    "norm": norm_key(ptxt),
                }
            )

        for dr in page.get_drawings():
            r = dr.get("rect")
            if not r:
                continue
            if abs(r.y1 - r.y0) < 3.0 and (r.x1 - r.x0) > 20.0:
                if not bbox or (
                    bbox[1] - 8 <= r.y0 <= bbox[3] + 8
                    and r.x1 >= bbox[0] - 8
                    and r.x0 <= bbox[2] + 8
                ):
                    cmidrules.append((float(r.y0), float(r.x0), float(r.x1)))

    return phrases, cmidrules, pdf_words


def _is_data_value(cell: str) -> bool:
    c = clean_cell_text(cell, is_header=True)
    if not c:
        return False
    c_plain = re.sub(r"[\*_]", "", c).strip()
    if c_plain in ("-", "—", "–", "N/A"):
        return True
    if _NUMERIC_CELL_RE.match(c_plain):
        return True
    if re.match(r"^\d+\.\d+\s*\(", c_plain):
        return True
    return False


def _is_subheader_row(
    row: list[str],
    prev_headers: list[list[str]],
    prev_spans: list[list[tuple[int, int, str]]],
) -> bool:
    non_empty = [c for c in row if c.strip()]
    if not non_empty:
        return False
    plain = [re.sub(r"[\*_]", "", c).strip() for c in non_empty]
    # 编号条目行（"5. S | Simple declarative clause"）与带长描述的尾列是正文行。
    # 早先没有这条判据，词性标签表的整段正文都被当成子表头并进了表头。
    if any(re.match(r"^\d+[.)]\s+\S", c) for c in plain):
        return False
    if any(len(c.split()) >= 3 for c in plain[1:]):
        return False
    dash_count = sum(
        1 for c in non_empty if re.sub(r"[\*_]", "", c).strip() in ("-", "—", "–")
    )
    if dash_count >= 2:
        return False
    if any(
        _is_data_value(c) and not re.match(r"^\d+\s*hops?$", c.strip(), re.I)
        for c in non_empty
    ):
        return False
    ncols = len(row)
    prev_filled = [any(h[c].strip() for h in prev_headers) for c in range(ncols)]
    if not all(prev_filled):
        return True
    if any(bool(sp) for sp in prev_spans):
        return True
    if not row[0].strip():
        return True
    if any(not all(h[c].strip() for c in range(ncols)) for h in prev_headers):
        return True
    return False


def _has_chopped_boundary_word(pieces: list[str], pdf_words: set[str]) -> bool:
    for i in range(len(pieces) - 1):
        lp = re.sub(r"[\*_]", "", pieces[i]).strip()
        rp = re.sub(r"[\*_]", "", pieces[i + 1]).strip()
        if re.search(r"[·×*]$", lp) and re.match(r"^10\^", rp):
            continue
        ltoks = lp.split()
        rtoks = rp.split()
        if not ltoks or not rtoks:
            return False
        lw = norm_key(ltoks[-1])
        rw = norm_key(rtoks[0])
        if pdf_words:
            # 只有当两部分都不可能是页面上独立存在的词时，才可能是"同一个词被劈成两列"。
            # 早先这里有 len(lw)<=2 or len(rw)<=2 的宽容分支，它专门放行单个数字：
            # 页面上别处出现过 "12"，于是单元格 "1" 和 "2" 就被判成同一个词并成 "12"。
            if (lw + rw) in pdf_words and (lw not in pdf_words or rw not in pdf_words):
                continue
            return False
        else:
            if (ltoks[-1], rtoks[0]) in {
                ("BL", "EU"),
                ("V", "BZ"),
                ("D", "T"),
                ("MemN", "2N"),
                ("A", "()"),
                ("D", "()"),
            }:
                continue
            if (
                re.search(r"[A-Za-z]$", ltoks[-1])
                and re.match(r"^[a-z]", rtoks[0])
                and (len(lw) <= 4 or len(rw) <= 3 or not lw.isalpha())
            ):
                continue
            return False
    return True


def _repair_row_colspans(
    row: list[str],
    phrase_map: dict[str, str],
    single_norms: set[str],
    pdf_words: set[str],
    is_header: bool = False,
) -> tuple[list[str], list[tuple[int, int, str]]]:
    K = len(row)
    out = [clean_cell_text(c, is_header=is_header) for c in row]
    spans: list[tuple[int, int, str]] = []
    c = 0
    while c < K:
        if not out[c]:
            c += 1
            continue
        matched = False
        for d in range(min(K - 1, c + 7), c, -1):
            if not out[d]:
                continue
            sub_cols = [k for k in range(c, d + 1) if out[k]]
            if len(sub_cols) < 2 or len(sub_cols) != (d - c + 1):
                continue
            pieces = [out[k] for k in sub_cols]
            # 数值单元格永不拼接：|1|2| 并成 |12| 是凭空造出论文里没有的值，
            # 比丢一张表更糟 —— 它给裁判喂的是假证据而不是没证据。
            # 数学式的跨格值（如 3.3 · / 10^18）不被 _is_data_value 认作数值，仍走原有修复。
            if any(_is_data_value(p) for p in pieces):
                continue
            norms = [norm_key(p) for p in pieces]
            comb = "".join(norms)
            restored = None
            all_bold = all(p.startswith("**") and p.endswith("**") for p in pieces)
            all_italic = all(
                p.startswith("_") and p.endswith("_") and not p.startswith("__")
                for p in pieces
            )

            if comb in phrase_map and (
                any(n not in single_norms for n in norms)
                or _has_chopped_boundary_word(pieces, pdf_words)
            ):
                restored = phrase_map[comb]
            elif _has_chopped_boundary_word(pieces, pdf_words):
                plain_pieces = [re.sub(r"[\*_]", "", p).strip() for p in pieces]
                if (
                    len(plain_pieces) == 2
                    and re.search(r"[·×*]$", plain_pieces[0])
                    and re.match(r"^10\^", plain_pieces[1])
                ):
                    restored = f"{plain_pieces[0]} {plain_pieces[1]}"
                else:
                    restored = "".join(plain_pieces)

            if restored is not None:
                if all_bold and not restored.startswith("**"):
                    restored = f"**{restored}**"
                elif all_italic and not restored.startswith("_"):
                    restored = f"_{restored}_"
                spans.append((c, d, restored))
                out[c] = restored
                for k in range(c + 1, d + 1):
                    out[k] = ""
                c = d + 1
                matched = True
                break
        if not matched:
            c += 1
    return out, spans


def _table_bbox(
    pages: Sequence[pymupdf.Page], raw_rows: list[list[str]]
) -> tuple[float, float, float, float] | None:
    """用表格自身的单元格去匹配页面上的短语，圈出表格的竖向区域。

    不圈的话词表覆盖整页 —— 页码、脚注、正文里的数字全都算数，
    合并判据就会凭"这页别处出现过 12"就认定 |1|2| 是被劈开的同一个词。
    圈不住就退回整页：宁可少合并，也不要错合并。
    """
    cell_norms = {norm_key(c) for row in raw_rows for c in row if c.strip()}
    ys: list[float] = []
    x0s: list[float] = []
    x1s: list[float] = []
    for page in pages:
        phrases, _, _ = extract_visual_info(page)
        hits = [p for p in phrases if p["norm"] and p["norm"] in cell_norms]
        if len(hits) < 3:
            continue
        for h in hits:
            ys.extend((h["y0"], h["y1"]))
            x0s.append(h["x0"])
            x1s.append(h["x1"])
    if len(ys) < 3:
        return None
    page_h = pages[0].rect.height
    return (
        max(0.0, min(x0s) - 6),
        max(0.0, min(ys) - 10),
        min(pages[0].rect.width, max(x1s) + 6),
        min(max(ys) + 10, page_h),
    )


def reconstruct_table(
    raw_md: str,
    pages: pymupdf.Page | Sequence[pymupdf.Page] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> str:
    """
    Reconstruct a clean, colspan-repaired, single-header Markdown table using PDF visual BBox
    coordinates (when available) plus deterministic cell & header normalization.
    """
    lines = [
        ln.strip()
        for ln in raw_md.strip().splitlines()
        if ln.strip().startswith("|")
    ]
    if len(lines) < 2:
        return raw_md
    sep_idx = next(
        (i for i, ln in enumerate(lines) if re.match(r"^\|[\s:\-|]+\|$", ln)), 1
    )
    raw_rows: list[list[str]] = []
    for i, ln in enumerate(lines):
        if i == sep_idx:
            continue
        cells = [c.strip() for c in ln.split("|")[1:-1]]
        raw_rows.append(cells)
    if not raw_rows:
        return raw_md

    ncols = max(len(r) for r in raw_rows)
    for r in raw_rows:
        while len(r) < ncols:
            r.append("")

    if bbox is None and pages is not None:
        bbox = _table_bbox([pages] if isinstance(pages, pymupdf.Page) else list(pages), raw_rows)
    phrases, _, pdf_words = extract_visual_info(pages, bbox=bbox)
    phrase_map: dict[str, str] = {}
    single_norms: set[str] = set()
    for p in phrases:
        if p["norm"]:
            phrase_map.setdefault(p["norm"], p["text"])
            if len(p["norm"]) >= 3:
                single_norms.add(p["norm"])

    r0, spans0 = _repair_row_colspans(
        raw_rows[0], phrase_map, single_norms, pdf_words, is_header=True
    )
    header_rows = [r0]
    header_spans = [spans0]
    body_start = 1
    while body_start < min(len(raw_rows), 4):
        cand, cand_spans = _repair_row_colspans(
            raw_rows[body_start], phrase_map, single_norms, pdf_words, is_header=True
        )
        if _is_subheader_row(cand, header_rows, header_spans):
            header_rows.append(cand)
            header_spans.append(cand_spans)
            body_start += 1
        else:
            break

    # Propagate spanning parent headers across child columns only when lower header rows exist
    if len(header_rows) > 1:
        for h_idx in range(len(header_rows) - 1):
            p_row = header_rows[h_idx]
            for c_start, c_end, text in header_spans[h_idx]:
                for k in range(c_start, c_end + 1):
                    if k == c_start or any(
                        header_rows[low][k].strip()
                        for low in range(h_idx + 1, len(header_rows))
                    ):
                        p_row[k] = text

        child = header_rows[-1]
        for h_idx in range(len(header_rows) - 1):
            p_row = header_rows[h_idx]
            if child[1:]:
                first_child_lbl = norm_key(child[1])
                group_starts = [
                    c
                    for c in range(1, ncols)
                    if norm_key(child[c]) == first_child_lbl and first_child_lbl
                ]
                if len(group_starts) >= 2:
                    for gi, gs in enumerate(group_starts):
                        ge = (
                            (group_starts[gi + 1] - 1)
                            if (gi + 1 < len(group_starts))
                            else (ncols - 1)
                        )
                        vals = [p_row[c] for c in range(gs, ge + 1) if p_row[c]]
                        if len(set(vals)) == 1:
                            uniq = vals[0]
                            for c in range(gs, ge + 1):
                                p_row[c] = uniq
                else:
                    for c in range(1, ncols - 1):
                        if not p_row[c] and child[c] and p_row[c + 1]:
                            if any(cs == c + 1 for cs, ce, _ in header_spans[h_idx]):
                                p_row[c] = p_row[c + 1]

    flat_header: list[str] = []
    for c in range(ncols):
        parts: list[str] = []
        for h_row in header_rows:
            val = h_row[c].strip()
            if val and (not parts or norm_key(val) != norm_key(parts[-1])):
                parts.append(val)
        if not parts:
            flat_header.append("")
        elif len(parts) == 1:
            flat_header.append(parts[0])
        else:
            parent = parts[0]
            child_str = " / ".join(parts[1:])
            p_clean = re.sub(r"^\*\*(.*)\*\*$", r"\1", parent).strip()
            c_clean = re.sub(r"^\*\*(.*)\*\*$", r"\1", child_str).strip()
            flat_header.append(f"{p_clean} ({c_clean})")

    body_rows: list[list[str]] = []
    for r_idx in range(body_start, len(raw_rows)):
        repaired_row, _ = _repair_row_colspans(
            raw_rows[r_idx], phrase_map, single_norms, pdf_words, is_header=False
        )
        non_empty_cells = [c for c in repaired_row if c]
        br_counts = [
            len(re.split(r"<br\s*/?>", c, flags=re.I)) for c in non_empty_cells
        ]
        multi_br = [cnt for cnt in br_counts if cnt > 1]
        if (
            non_empty_cells
            and multi_br
            and len( set(multi_br) ) == 1
            and (len(multi_br) == len(non_empty_cells) or len(multi_br) >= 2)
        ):
            n_sub = multi_br[0]
            for s_i in range(n_sub):
                sub_row = []
                for c in repaired_row:
                    if not c:
                        sub_row.append("")
                    else:
                        segs = [
                            seg.strip()
                            for seg in re.split(r"<br\s*/?>", c, flags=re.I)
                        ]
                        chosen = segs[s_i] if len(segs) == n_sub else segs[0]
                        sub_row.append(clean_cell_text(chosen, is_header=True))
                body_rows.append(sub_row)
        else:
            body_rows.append(
                [clean_cell_text(c, is_header=True) for c in repaired_row]
            )

    # Vertical split repair for rotated/multi-line group labels like A / () -> (A)
    for r_i in range(len(body_rows) - 1):
        if ncols > 0:
            c0_cur = body_rows[r_i][0].strip()
            c0_nxt = body_rows[r_i + 1][0].strip()
            if re.fullmatch(r"[A-Z]", c0_cur) and c0_nxt == "()":
                body_rows[r_i][0] = f"({c0_cur})"
                body_rows[r_i + 1][0] = ""

    keep_cols = [
        c
        for c in range(ncols)
        if flat_header[c].strip() or any(r[c].strip() for r in body_rows)
    ]
    if not keep_cols:
        return raw_md

    out_lines = [
        "|" + "|".join(flat_header[c] for c in keep_cols) + "|",
        "|" + "|".join("---" for _ in keep_cols) + "|",
    ]
    for r in body_rows:
        out_lines.append("|" + "|".join(r[c] for c in keep_cols) + "|")
    return "\n".join(out_lines)


def reconstruct_tables_in_markdown(
    md_text: str,
    pages: pymupdf.Page | Sequence[pymupdf.Page] | None = None,
    bbox: tuple[float, float, float, float] | None = None,
) -> str:
    """Find all Markdown tables in `md_text` and reconstruct each using visual BBox info."""
    if "|" not in md_text or "---" not in md_text:
        return md_text

    def _repl(m: re.Match) -> str:
        block = m.group(0)
        if "---" not in block:
            return block
        trailing_nl = "\n" if block.endswith("\n") else ""
        rebuilt = reconstruct_table(block.strip(), pages=pages, bbox=bbox)
        return rebuilt + trailing_nl

    return _MD_TABLE_BLOCK_RE.sub(_repl, md_text)
