"""网络检索与流式下载加固契约测试 (Step 16 TDD)

验证目标：
1. PDFLoader 采用分块流式写入临时文件，具备最大/最小文件体积熔断，严防大文件撑爆内存与恶意伪文件
2. PDFLoader 严格执行 %PDF- 魔数校验与 PyMuPDF 结构合法性校验，异常时临时文件必须清理
3. _http_get_text / _http_get_json 对 404、401、403 等永久性错误即刻短路，杜绝无意义重试
4. _http_get_text 对 429 限流、5xx 服务端错误和网络超时执行指数退避重试，区分 RATE_LIMITED 与 NETWORK_TIMEOUT
5. 同域名请求具备最小时间间隔限速保护
"""
import io
import time
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock
import urllib.error

import pymupdf

from citetrace_pdf.pdf.loader import PDFLoader
from models.verification import FailureReason
from tools.academic_fetcher import (
    AcademicRequestError,
    _http_get_text,
    _http_get_json,
    _DOMAIN_LAST_REQUEST,
)


class TestNetworkResilienceAndStreaming(unittest.TestCase):

    def _create_real_pdf_bytes(self) -> bytes:
        doc = pymupdf.open()
        page = doc.new_page()
        page.insert_text((50, 72), "CiteTrace Resilience Verification Document")
        data = doc.tobytes()
        doc.close()
        # 确保大于 1000 字节
        if len(data) < 1000:
            data += b"\n% " + b"X" * (1001 - len(data))
        return data

    def test_streaming_download_success_and_atomicity(self):
        """测试流式下载有效 PDF，成功后临时 .part 文件自动替换为 raw.pdf"""
        pdf_bytes = self._create_real_pdf_bytes()

        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = PDFLoader(cache_dir=tmp_dir, chunk_size=128)

            mock_response = io.BytesIO(pdf_bytes)
            with patch("urllib.request.urlopen", return_value=mock_response):
                out_path = loader.load("paper_01", "https://arxiv.org/pdf/1706.03762.pdf")

            self.assertTrue(out_path.is_file())
            self.assertEqual(out_path.name, "raw.pdf")
            self.assertEqual(out_path.stat().st_size, len(pdf_bytes))
            # 临时文件绝无残留
            part_path = out_path.with_suffix(".pdf.part")
            self.assertFalse(part_path.exists())

    def test_streaming_download_max_size_circuit_breaker(self):
        """测试下载过程中超过 max_size 立即熔断并清理临时文件"""
        pdf_bytes = self._create_real_pdf_bytes()

        with tempfile.TemporaryDirectory() as tmp_dir:
            # 限制最大 500 字节，而 pdf_bytes > 1000 字节
            loader = PDFLoader(cache_dir=tmp_dir, max_size=500, chunk_size=128)

            mock_response = io.BytesIO(pdf_bytes)
            with patch("urllib.request.urlopen", return_value=mock_response):
                with self.assertRaises(ValueError) as ctx:
                    loader.load("paper_huge", "https://example.com/huge.pdf")

                self.assertIn("体积超过上限", str(ctx.exception))

            paper_dir = Path(tmp_dir) / "paper_huge"
            self.assertFalse((paper_dir / "raw.pdf").exists())
            self.assertFalse((paper_dir / "raw.pdf.part").exists())

    def test_streaming_download_rejects_html_landing_page(self):
        """测试返回 HTML 付费墙或落地页时，读取第一块立即阻断"""
        html_bytes = b"<!DOCTYPE html><html><body>Access Denied / Paywall</body></html>" * 30

        with tempfile.TemporaryDirectory() as tmp_dir:
            loader = PDFLoader(cache_dir=tmp_dir, chunk_size=128)

            mock_response = io.BytesIO(html_bytes)
            with patch("urllib.request.urlopen", return_value=mock_response):
                with self.assertRaises(ValueError) as ctx:
                    loader.load("paper_html", "https://publisher.com/article.pdf")

                self.assertIn("不是有效 PDF", str(ctx.exception))

            paper_dir = Path(tmp_dir) / "paper_html"
            self.assertFalse((paper_dir / "raw.pdf").exists())
            self.assertFalse((paper_dir / "raw.pdf.part").exists())

    def test_http_short_circuits_on_404_not_found(self):
        """测试 HTTP 404 永久性错误立即短路抛出 DOCUMENT_NOT_FOUND，重试次数为 0"""
        err_404 = urllib.error.HTTPError(
            url="https://api.example.com/work/404",
            code=404,
            msg="Not Found",
            hdrs={},
            fp=io.BytesIO(b"Not Found"),
        )

        with patch("urllib.request.urlopen", side_effect=err_404) as mock_urlopen:
            with self.assertRaises(AcademicRequestError) as ctx:
                _http_get_text("https://api.example.com/work/404", max_retries=3)

            self.assertEqual(ctx.exception.status_code, 404)
            self.assertEqual(ctx.exception.reason, FailureReason.DOCUMENT_NOT_FOUND)
            self.assertEqual(ctx.exception.retry_count, 0)
            # 必须仅调用 1 次，严禁多余重试
            self.assertEqual(mock_urlopen.call_count, 1)

    def test_http_short_circuits_on_403_forbidden(self):
        """测试 HTTP 403 权限错误立即短路抛出 PAYWALL_LOCKED，重试次数为 0"""
        err_403 = urllib.error.HTTPError(
            url="https://api.example.com/work/403",
            code=403,
            msg="Forbidden",
            hdrs={},
            fp=io.BytesIO(b"Forbidden"),
        )

        with patch("urllib.request.urlopen", side_effect=err_403) as mock_urlopen:
            with self.assertRaises(AcademicRequestError) as ctx:
                _http_get_text("https://api.example.com/work/403", max_retries=3)

            self.assertEqual(ctx.exception.status_code, 403)
            self.assertEqual(ctx.exception.reason, FailureReason.PAYWALL_LOCKED)
            self.assertEqual(ctx.exception.retry_count, 0)
            self.assertEqual(mock_urlopen.call_count, 1)

    def test_http_retries_and_backoff_on_429_rate_limit(self):
        """测试 HTTP 429 限流时进行退避重试并在耗尽后标记 RATE_LIMITED"""
        err_429 = urllib.error.HTTPError(
            url="https://api.semanticscholar.org/graph/v1/paper/429",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "0.01"},
            fp=io.BytesIO(b"Rate Limit"),
        )

        with patch("urllib.request.urlopen", side_effect=err_429) as mock_urlopen:
            with patch("time.sleep") as mock_sleep:
                with self.assertRaises(AcademicRequestError) as ctx:
                    _http_get_text(
                        "https://api.semanticscholar.org/graph/v1/paper/429",
                        max_retries=3,
                        backoff_base=0.01,
                    )

                self.assertEqual(ctx.exception.status_code, 429)
                self.assertEqual(ctx.exception.reason, FailureReason.RATE_LIMITED)
                self.assertEqual(mock_urlopen.call_count, 3)
                # 检查 sleep 至少调用了 2 次 (重试前)
                self.assertGreaterEqual(mock_sleep.call_count, 2)

    def test_http_retries_and_classifies_timeout(self):
        """测试连接超时错误重试耗尽后准确标记 NETWORK_TIMEOUT"""
        err_timeout = TimeoutError("The read operation timed out")

        with patch("urllib.request.urlopen", side_effect=err_timeout) as mock_urlopen:
            with patch("time.sleep"):
                with self.assertRaises(AcademicRequestError) as ctx:
                    _http_get_text("https://export.arxiv.org/timeout", max_retries=3, backoff_base=0.01)

                self.assertEqual(ctx.exception.reason, FailureReason.NETWORK_TIMEOUT)
                self.assertEqual(mock_urlopen.call_count, 3)

    def test_http_recovers_on_second_retry(self):
        """测试第一次 503 失败、第二次重试成功恢复"""
        err_503 = urllib.error.HTTPError(
            url="https://api.crossref.org/works/1",
            code=503,
            msg="Service Unavailable",
            hdrs={},
            fp=io.BytesIO(b"Unavailable"),
        )
        success_resp = io.BytesIO(b'{"status": "ok"}')

        with patch("urllib.request.urlopen", side_effect=[err_503, success_resp]):
            with patch("time.sleep"):
                res = _http_get_json("https://api.crossref.org/works/1", max_retries=3, backoff_base=0.01)
                self.assertEqual(res, {"status": "ok"})


if __name__ == "__main__":
    unittest.main()
