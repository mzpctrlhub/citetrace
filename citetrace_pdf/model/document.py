from dataclasses import dataclass, field


@dataclass(slots=True)
class Section:
    title: str
    start_page: int
    end_page: int
    content: str = ""
    level: int = 0
    number: str | None = None


@dataclass(slots=True)
class Document:
    paper_id: str
    paper_title: str
    total_pages: int
    sections: list[Section] = field(default_factory=list)
