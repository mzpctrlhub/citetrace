from __future__ import annotations

from collections.abc import Sequence

from ..model.layout import LayoutBlock


class LayoutNormalizer:
    """Reconstruct reading order using center-gutter projection and column heuristics."""

    FULL_WIDTH_RATIO = 0.55

    @classmethod
    def reorder(cls, items: Sequence[LayoutBlock], page_width: float) -> list[LayoutBlock]:
        if not items:
            return []
        mid_x = page_width / 2.0

        body_blocks = [
            b for b in items
            if (b.x1 - b.x0) > 60 and not (b.x1 < 60 or b.x0 > page_width - 60)
        ]
        left_blocks = [b for b in body_blocks if b.x1 <= mid_x + 30 and (b.x1 - b.x0) < page_width * cls.FULL_WIDTH_RATIO]
        right_blocks = [b for b in body_blocks if b.x0 >= mid_x - 30 and (b.x1 - b.x0) < page_width * cls.FULL_WIDTH_RATIO]
        gutter_crossers = [
            b for b in body_blocks
            if (b.x1 - b.x0) < page_width * cls.FULL_WIDTH_RATIO
            and b.x0 < mid_x - 25
            and b.x1 > mid_x + 25
        ]

        is_two_col = False
        if len(left_blocks) >= 2 and len(right_blocks) >= 2:
            overlap_count = 0
            for lb in left_blocks:
                for rb in right_blocks:
                    if max(lb.y0, rb.y0) < min(lb.y1, rb.y1) - 10:
                        overlap_count += 1
            # 若存在明确垂直中缝（无单栏块跨越中轴线），即使左右栏因插图在 y 轴错开也属于双栏
            if overlap_count >= 2 or (overlap_count >= 1 and not gutter_crossers) or (
                not gutter_crossers and len(left_blocks) >= 3 and len(right_blocks) >= 2
            ):
                is_two_col = True

        if not is_two_col:
            return sorted(items, key=lambda item: (round(item.y0 / 8) * 8, item.x0))

        full_width = sorted(
            (item for item in items if (item.x1 - item.x0) / page_width >= cls.FULL_WIDTH_RATIO),
            key=lambda item: item.y0,
        )
        columns = [item for item in items if (item.x1 - item.x0) / page_width < cls.FULL_WIDTH_RATIO]

        def column_order(blocks: Sequence[LayoutBlock]) -> list[LayoutBlock]:
            left = sorted((b for b in blocks if (b.x0 + b.x1) / 2.0 < mid_x), key=lambda b: b.y0)
            right = sorted((b for b in blocks if (b.x0 + b.x1) / 2.0 >= mid_x), key=lambda b: b.y0)
            return left + right

        if not full_width:
            return column_order(columns)

        ordered: list[LayoutBlock] = []
        current_y = 0.0
        for wide_block in full_width:
            band = [b for b in columns if current_y <= (b.y0 + b.y1) / 2.0 < wide_block.y0]
            ordered.extend(column_order(band))
            ordered.append(wide_block)
            current_y = wide_block.y1
        ordered.extend(column_order([b for b in columns if (b.y0 + b.y1) / 2.0 >= current_y]))
        return ordered
