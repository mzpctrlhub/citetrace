"""CiteTrace academic PDF processing package."""

from .chunker import chunk_paper_folder, locate_quote
from .processor import PDFProcessor

__all__ = ["PDFProcessor", "chunk_paper_folder", "locate_quote"]
