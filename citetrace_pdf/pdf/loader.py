from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

logger = logging.getLogger(__name__)


class PDFLoader:
    """Resolve legacy local PDFs or stream-download and persist a paper PDF with size and integrity safeguards."""

    DEFAULT_MAX_SIZE = 100 * 1024 * 1024  # 100 MB
    DEFAULT_MIN_SIZE = 1000                # 1 KB
    CHUNK_SIZE = 64 * 1024                 # 64 KB per stream read block

    def __init__(
        self,
        cache_dir: str | Path,
        timeout: int = 30,
        max_size: int = DEFAULT_MAX_SIZE,
        min_size: int = DEFAULT_MIN_SIZE,
        chunk_size: int = CHUNK_SIZE,
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.timeout = timeout
        self.max_size = max_size
        self.min_size = min_size
        self.chunk_size = chunk_size

    @classmethod
    def is_valid_pdf(cls, path: Path) -> bool:
        """校验文件是否为有效 PDF（大小 >= min_size，以 %PDF- 魔数开头，且 PyMuPDF 可打开）"""
        try:
            if not path.is_file() or path.stat().st_size < cls.DEFAULT_MIN_SIZE:
                return False
            with open(path, "rb") as f:
                if f.read(5) != b"%PDF-":
                    return False
            try:
                import pymupdf
                with pymupdf.open(path) as doc:
                    return doc.page_count >= 1
            except Exception:
                return False
        except OSError:
            return False

    def load(self, paper_id: str, pdf_url: str) -> Path:
        paper_dir = self.cache_dir / paper_id
        paper_dir.mkdir(parents=True, exist_ok=True)
        local_pdf = paper_dir / "raw.pdf"
        legacy_pdf = self.cache_dir / f"{paper_id}.pdf"
        if not local_pdf.exists() and legacy_pdf.exists():
            shutil.copy2(legacy_pdf, local_pdf)
            logger.info("Migrated legacy PDF cache %s", legacy_pdf)
        if local_pdf.exists():
            if self.is_valid_pdf(local_pdf):
                return local_pdf
            logger.warning("发现损坏的伪 PDF 缓存 (实为 HTML 网页)，自动清理: %s", local_pdf)
            local_pdf.unlink(missing_ok=True)

        # 若输入本身就是一个本地存在的 PDF 文件，直接复制并使用
        candidate_local = Path(pdf_url)
        if candidate_local.is_file():
            if candidate_local.resolve() != local_pdf.resolve():
                shutil.copy2(candidate_local, local_pdf)
            logger.info("Loaded local PDF %s -> %s", candidate_local, local_pdf)
            return local_pdf

        logger.info("Downloading paper %s from %s (streaming)", paper_id, pdf_url)
        request = urllib.request.Request(
            pdf_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "Accept": "application/pdf,application/octet-stream,*/*;q=0.9",
                "Accept-Language": "en-US,en;q=0.9",
            }
        )
        temporary = local_pdf.with_suffix(".pdf.part")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                total_bytes = 0
                first_block = True
                with temporary.open("wb") as output:
                    while True:
                        block = response.read(self.chunk_size)
                        if not block:
                            break
                        if first_block:
                            if not block.startswith(b"%PDF-"):
                                raise ValueError(f"下载内容不是有效 PDF (返回了 HTML 落地页或付费墙，URL: {pdf_url})")
                            first_block = False
                        total_bytes += len(block)
                        if total_bytes > self.max_size:
                            raise ValueError(
                                f"PDF 体积超过上限 ({total_bytes} > {self.max_size} bytes, URL: {pdf_url})"
                            )
                        output.write(block)

                if total_bytes < self.min_size:
                    raise ValueError(f"下载内容体积过小 ({total_bytes} bytes)，非有效 PDF 文件 (URL: {pdf_url})")

            # PyMuPDF 结构有效性校验
            import pymupdf
            try:
                with pymupdf.open(temporary) as doc:
                    if doc.page_count < 1:
                        raise ValueError(f"PDF 结构异常：页数为 0 (URL: {pdf_url})")
            except Exception as e:
                raise ValueError(f"下载文件虽然有 %PDF- 标头，但 PyMuPDF 无法解析: {e}") from e

            # 原子替换至目标路径
            temporary.replace(local_pdf)
        finally:
            temporary.unlink(missing_ok=True)
        logger.info("Downloaded PDF to %s (%d bytes)", local_pdf, local_pdf.stat().st_size)
        return local_pdf
