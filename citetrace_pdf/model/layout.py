from dataclasses import dataclass, field
from typing import Literal

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class LayoutBlock:
    text: str
    bbox: BBox
    page: int
    kind: Literal["text", "table"] = "text"

    @property
    def x0(self) -> float:
        return self.bbox[0]

    @property
    def y0(self) -> float:
        return self.bbox[1]

    @property
    def x1(self) -> float:
        return self.bbox[2]

    @property
    def y1(self) -> float:
        return self.bbox[3]


@dataclass(slots=True)
class PageLayout:
    number: int
    width: float
    height: float
    blocks: list[LayoutBlock] = field(default_factory=list)
