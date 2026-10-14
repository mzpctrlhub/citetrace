# citetrace/tools/academic_fetcher.py
import os
import re
import json
import time
import logging
import difflib
import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import hashlib
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Tuple, Any, List

from citetrace_pdf import PDFProcessor
from citetrace_pdf.storage.cache import MetadataCache
from models.citation import CitationRef, CitationStatus
from models.verification import FailureReason

logger = logging.getLogger(__name__)

# 设置通用网络请求头，防止被学术平台 403 拦截
DEFAULT_HEADERS = {
    "Accept": "application/json, application/pdf, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}

ARXIV_HEADERS = {
    "Accept": "application/atom+xml, application/xml, text/xml, */*",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
}

KNOWN_BLOCKED_GATEWAYS = (
    "dl.acm.org/ft_gateway",
    "dl.acm.org/doi/pdf",
    "ieeexplore.ieee.org/stamp",
    "sciencedirect.com/science/article",
)


def _is_blocked_url(url: str) -> bool:
    return any(gw in url for gw in KNOWN_BLOCKED_GATEWAYS)


def _normalize_pdf_url(url: Optional[str]) -> Optional[str]:
    """将 ACL Anthology 或 arXiv 摘要页链接规范化为直接可下载的 PDF 直链"""
    if not url:
        return None
    u = url.strip()
    m_acl = re.match(r"^https?://(?:www\.)?(?:aclweb\.org/anthology|aclanthology\.org)/([A-Za-z0-9\-]+?)(?:\.pdf)?/?$", u, re.I)
    if m_acl:
        return f"https://aclanthology.org/{m_acl.group(1)}.pdf"
    m_arxiv = re.match(r"^https?://arxiv\.org/abs/(\d{4}\.\d{4,5}(?:v\d+)?)$", u, re.I)
    if m_arxiv:
        return f"https://arxiv.org/pdf/{m_arxiv.group(1)}.pdf"
    return u


class AcademicRequestError(Exception):
    """学术请求异常，携带分类错误、HTTP 状态码及诊断指标"""

    def __init__(
        self,
        message: str,
        source: str = "",
        query: str = "",
        status_code: Optional[int] = None,
        reason: FailureReason = FailureReason.DOCUMENT_NOT_FOUND,
        retry_count: int = 0,
        elapsed_ms: float = 0.0,
    ):
        super().__init__(message)
        self.source = source
        self.query = query
        self.status_code = status_code
        self.reason = reason
        self.retry_count = retry_count
        self.elapsed_ms = elapsed_ms


# 域名限速保护：记录各域名的上次请求时间戳
_DOMAIN_LAST_REQUEST: Dict[str, float] = {}
_MIN_DOMAIN_INTERVAL: float = 0.05  # 秒，默认最小间隔


def _http_get_text(
    url: str,
    headers: Dict[str, str] = DEFAULT_HEADERS,
    timeout: int = 10,
    max_retries: int = 3,
    backoff_base: float = 0.1,
    source_name: str = "",
) -> str:
    """
    通用 HTTP GET 请求工具，支持：
    1. 域名限速保护（防止瞬时突发打垮外部接口）
    2. 针对 429、5xx、网络超时与连接重置进行指数退避重试 (最多 max_retries 次)
    3. 识别 Retry-After 标头
    4. 针对 404、401、403、400 等永久性错误立即短路不重试
    5. 结构化异常抛出与诊断日志输出
    """
    parsed = urllib.parse.urlparse(url)
    domain = parsed.netloc.lower()
    source = source_name or domain or "academic_api"

    # 限速控制：同域名最小间隔
    now = time.time()
    last = _DOMAIN_LAST_REQUEST.get(domain, 0.0)
    wait_time = _MIN_DOMAIN_INTERVAL - (now - last)
    if 0 < wait_time < 5.0:
        time.sleep(wait_time)
    _DOMAIN_LAST_REQUEST[domain] = time.time()

    last_exception = None
    last_status = None
    start_time = time.time()

    for attempt in range(max_retries):
        req_start = time.time()
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed_ms = (time.time() - req_start) * 1000.0
                logger.debug(
                    "[HTTP 成功] source=%s query=%s status=200 retry=%d elapsed_ms=%.1f",
                    source, url[:80], attempt, elapsed_ms
                )
                return resp.read().decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            last_status = e.code
            last_exception = e
            elapsed_ms = (time.time() - req_start) * 1000.0

            # 404 Not Found 或 401/403 权限问题：短路直接抛出，无需重试
            if e.code in (404, 401, 403, 400):
                logger.info(
                    "[HTTP 短路] source=%s status=%d error=%s retry=%d elapsed_ms=%.1f",
                    source, e.code, e.reason, attempt, elapsed_ms
                )
                failure_reason = (
                    FailureReason.PAYWALL_LOCKED if e.code in (401, 403)
                    else FailureReason.DOCUMENT_NOT_FOUND
                )
                raise AcademicRequestError(
                    f"HTTP {e.code}: {e.reason}",
                    source=source,
                    query=url,
                    status_code=e.code,
                    reason=failure_reason,
                    retry_count=attempt,
                    elapsed_ms=elapsed_ms,
                ) from e

            # 429 限流 或 5xx 服务端错误：计算退避时间并重试
            retry_after = 0.0
            if hasattr(e, "headers") and e.headers and "Retry-After" in e.headers:
                try:
                    retry_after = float(e.headers["Retry-After"])
                except (ValueError, TypeError):
                    pass

            if attempt < max_retries - 1:
                sleep_dur = retry_after if retry_after > 0 else (backoff_base * (2 ** attempt))
                logger.warning(
                    "[HTTP 重试] source=%s status=%d attempt=%d/%d sleep=%.1fs url=%s",
                    source, e.code, attempt + 1, max_retries, sleep_dur, url[:80]
                )
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
            else:
                logger.error(
                    "[HTTP 耗尽] source=%s status=%d retries=%d elapsed_ms=%.1f url=%s",
                    source, e.code, max_retries, elapsed_ms, url[:80]
                )
                reason = FailureReason.RATE_LIMITED if e.code == 429 else FailureReason.DOWNLOAD_FAILED
                raise AcademicRequestError(
                    f"HTTP {e.code} after {max_retries} attempts",
                    source=source,
                    query=url,
                    status_code=e.code,
                    reason=reason,
                    retry_count=attempt,
                    elapsed_ms=elapsed_ms,
                ) from e

        except (urllib.error.URLError, TimeoutError, OSError) as e:
            last_exception = e
            elapsed_ms = (time.time() - req_start) * 1000.0
            is_timeout = isinstance(e, TimeoutError) or ("timed out" in str(e).lower())
            reason = FailureReason.NETWORK_TIMEOUT if is_timeout else FailureReason.DOWNLOAD_FAILED

            if attempt < max_retries - 1:
                sleep_dur = backoff_base * (2 ** attempt)
                logger.warning(
                    "[网络异常重试] source=%s error=%s attempt=%d/%d sleep=%.1fs url=%s",
                    source, type(e).__name__, attempt + 1, max_retries, sleep_dur, url[:80]
                )
                if sleep_dur > 0:
                    time.sleep(sleep_dur)
            else:
                logger.error(
                    "[网络异常耗尽] source=%s error=%s retries=%d elapsed_ms=%.1f url=%s",
                    source, type(e).__name__, max_retries, elapsed_ms, url[:80]
                )
                raise AcademicRequestError(
                    f"Network error: {e}",
                    source=source,
                    query=url,
                    status_code=last_status,
                    reason=reason,
                    retry_count=attempt,
                    elapsed_ms=elapsed_ms,
                ) from e

    raise AcademicRequestError(
        f"Request failed: {last_exception}",
        source=source,
        query=url,
        status_code=last_status,
        reason=FailureReason.DOCUMENT_NOT_FOUND,
        retry_count=max_retries,
        elapsed_ms=(time.time() - start_time) * 1000.0,
    )


def _http_get_json(
    url: str,
    headers: Dict[str, str] = DEFAULT_HEADERS,
    timeout: int = 10,
    max_retries: int = 3,
    backoff_base: float = 0.1,
    source_name: str = "",
) -> Dict[str, Any]:
    text = _http_get_text(
        url,
        headers=headers,
        timeout=timeout,
        max_retries=max_retries,
        backoff_base=backoff_base,
        source_name=source_name,
    )
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        logger.warning("JSON 解析失败 (%s): %s", url[:80], e)
        raise AcademicRequestError(
            f"Invalid JSON response: {e}",
            source=source_name or "json_parser",
            query=url,
            status_code=200,
            reason=FailureReason.PARSE_FAILED,
        ) from e


def sanitize_filename(identifier: str) -> str:
    """
    文件名安全清洗：将 DOI/arXiv 中的斜杠、冒号等危险字符替换为下划线
    例: '10.18653/v1/2020.emnlp' -> '10.18653_v1_2020.emnlp'
    """
    clean = re.sub(r'[/\\:*?"<>|]', '_', identifier)
    return clean.strip("._")


TITLE_STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "on", "in", "for", "to", "with", "by",
    "at", "from", "is", "are", "its", "their", "this", "that", "new", "via",
    "using", "based", "towards", "toward", "arxiv", "corr", "preprint", "abs",
}


def is_title_match(query_title: str, candidate_title: str, threshold: float = 0.65) -> bool:
    """
    标题严格相似度比对（对称 Jaccard + 词序序列双重检验）：
    1. 统一转小写，剥离标点符号与停用词
    2. 至少有 2 个实质信息词重合，坚决拦截单个常见词（如 'Le'）误匹配到不相关文献
    3. 对称计算词汇集交并比 Jaccard = |Q ∩ C| / |Q ∪ C| >= threshold
    4. 检验词序序列相似度 (SequenceMatcher) >= 0.80，防止 "Long Short-Term" 误匹配成 "Short-Term/Long-Term" 等颠倒词序
    """
    if not query_title or not candidate_title:
        return False

    def _clean_words(s: str) -> list[str]:
        return [w for w in re.findall(r'[a-zA-Z0-9]+', s.lower()) if w not in TITLE_STOPWORDS and len(w) > 1]

    q_tokens = _clean_words(query_title)
    c_tokens = _clean_words(candidate_title)

    if not q_tokens or not c_tokens:
        return False

    set_q = set(q_tokens)
    set_c = set(c_tokens)

    intersection = set_q.intersection(set_c)
    union = set_q.union(set_c)

    # 至少要有 2 个以上实质词重合（若查询总词数较少则至少要有 1 个）
    min_overlap = min(2, len(set_q))
    if len(intersection) < min_overlap:
        return False

    # 支持主标题省略副标题匹配（仅当候选标题在主标题后紧跟 ':' 或 ' for ' 副标题引导词，且覆盖率 >= 0.5）
    q_norm = re.sub(r"\s+", " ", query_title.strip().lower().rstrip("."))
    c_norm = re.sub(r"\s+", " ", candidate_title.strip().lower().rstrip("."))
    if (
        len(q_tokens) >= 3
        and c_tokens[:len(q_tokens)] == q_tokens
        and len(q_tokens) / len(c_tokens) >= 0.5
        and c_norm.startswith(q_norm)
        and re.match(r"^\s*(?::|for\b)", c_norm[len(q_norm):])
    ):
        return True

    jaccard = len(intersection) / len(union)
    if jaccard < threshold:
        return False

    # 词序与字符序列双重校验，杜绝颠倒词序或严重冗余标题
    seq_ratio = difflib.SequenceMatcher(None, " ".join(q_tokens), " ".join(c_tokens)).ratio()
    return seq_ratio >= 0.80


# ==============================================================================
# 落地文献校验：确认拿到的 PDF 确实是所请求的那一篇，而非付费墙网关返回的任意文献
# ==============================================================================

# 标题在 PDF 里常是 small-caps 排版，抽取结果可能只剩大写字母（如 "N GPU L A"），
# 因此校验文本域必须带上正文章节开头，那里标题通常是完整的。
TITLE_HAYSTACK_SECTIONS = 2
TITLE_HAYSTACK_CHARS = 1200
TITLE_MATCH_MIN_TOKENS = 3
TITLE_MATCH_COVERAGE = 0.6


def _title_tokens(text: str) -> set:
    """标题实质词集合：小写、去停用词、折叠复数词尾"""
    tokens = set()
    for raw in re.findall(r"[a-z0-9]+", (text or "").lower()):
        if len(raw) <= 2 or raw in TITLE_STOPWORDS:
            continue
        if len(raw) > 3 and raw.endswith("s") and not raw.endswith("ss"):
            raw = raw[:-1]
        tokens.add(raw)
    return tokens


def _document_haystack(metadata: Dict[str, Any], folder: Path) -> str:
    """提取"标题 + 前若干正文章节开头"作为可比对文本域"""
    parts = []
    if metadata.get("title_source") == "reference_metadata":
        if metadata.get("extracted_pdf_title"):
            parts.append(str(metadata.get("extracted_pdf_title")))
    else:
        parts.append(str(metadata.get("paper_title", "")))

    for item in metadata.get("outline", []):
        if len(parts) > TITLE_HAYSTACK_SECTIONS:
            break
        fname = str(item.get("file", ""))
        if not fname or fname.lower() in ("references.md", "other.md"):
            continue
        sec_path = folder / "sections" / fname
        if not sec_path.is_file():
            sec_path = folder / fname
        if not sec_path.is_file():
            continue
        try:
            parts.append(sec_path.read_text(encoding="utf-8", errors="ignore")[:TITLE_HAYSTACK_CHARS])
        except OSError:
            continue
    return " ".join(parts)


def verify_document_title(expected_title: str, metadata: Dict[str, Any], folder: Path) -> Tuple[bool, bool]:
    """
    校验已落地的文献夹是否确实是 expected_title 这篇论文。

    返回 (matched, verifiable)：
    - verifiable=False：期望标题实质词不足，无从判断，调用方不应据此阻断
    - verifiable=True 且 matched=False：拿到的几乎肯定是另一篇文献，必须丢弃
    """
    want = _title_tokens(expected_title)
    if len(want) < TITLE_MATCH_MIN_TOKENS:
        return True, False
    have = _title_tokens(_document_haystack(metadata, folder))
    coverage = len(want & have) / len(want)
    return coverage >= TITLE_MATCH_COVERAGE, True



def _parse_arxiv_entry(entry: ET.Element) -> Optional[Dict[str, Optional[str]]]:
    """从单个 arXiv Atom <entry> 节点提取标题、摘要、arXiv ID 与 PDF 直链"""
    id_node = entry.find("atom:id", ATOM_NS)
    raw_id = (id_node.text or "").strip() if id_node is not None else ""
    if "api/errors" in raw_id:
        return None

    title_node = entry.find("atom:title", ATOM_NS)
    title = title_node.text.strip().replace("\n", " ") if title_node is not None and title_node.text else "Unknown"

    summary_node = entry.find("atom:summary", ATOM_NS)
    summary = summary_node.text.strip().replace("\n", " ") if summary_node is not None and summary_node.text else ""

    m = re.search(r'(\d{4}\.\d{4,5}(?:v\d+)?)', raw_id)
    aid = m.group(1) if m else ""

    pdf_url = None
    for link in entry.findall("atom:link", ATOM_NS):
        if link.attrib.get("title") == "pdf":
            pdf_url = link.attrib.get("href")
            break
    if not pdf_url and aid:
        pdf_url = f"https://arxiv.org/pdf/{aid}.pdf"

    return {
        "title": title,
        "abstract": summary,
        "pdf_url": pdf_url,
        "arxiv_id": aid or None,
    }


def parse_arxiv_atom_xml(xml_content: str) -> Optional[Dict[str, Optional[str]]]:
    """
    解析 arXiv 官方返回的 Atom XML 格式文本，提取标题、摘要与 PDF 直链
    """
    try:
        root = ET.fromstring(xml_content)
        entry = root.find("atom:entry", ATOM_NS)
        return _parse_arxiv_entry(entry) if entry is not None else None
    except Exception as e:
        print(f"[Warn] 解析 arXiv XML 发生异常: {e}")
        return None


def fetch_arxiv_by_id(arxiv_id: str, timeout: int = 10) -> Optional[Dict[str, Optional[str]]]:
    """
    【第一梯队】凭 arXiv 编号精准直查文献 (100% 精确)
    """
    clean_id = arxiv_id.lower().replace("arxiv:", "").strip().rstrip(".")
    api_url = f"https://export.arxiv.org/api/query?id_list={clean_id}"

    try:
        return parse_arxiv_atom_xml(_http_get_text(api_url, headers=ARXIV_HEADERS, timeout=timeout))
    except Exception as e:
        print(f"[Warn] 凭 arXiv ID 抓取失败 ({arxiv_id}): {e}")
        return None


def fetch_arxiv_by_title(title: str, timeout: int = 10) -> Optional[Dict[str, Optional[str]]]:
    """
    【第一点五梯队】凭标题在 arXiv 中检索文献（对无显式 arXiv ID 的 AI/ML 文献极度有效，如 Neural GPUs）
    """
    if not title or not title.strip():
        return None

    clean_title = re.sub(r'[^a-zA-Z0-9\s]', ' ', title).strip()
    words = [w for w in clean_title.split() if len(w) > 2]
    if not words:
        return None

    query_str = "+AND+".join(urllib.parse.quote(w) for w in words[:6])
    clean_phrase = urllib.parse.quote(f'"{title.strip()[:80]}"')
    urls = [
        f"https://export.arxiv.org/api/query?search_query=ti:{query_str}&max_results=3",
        f"https://export.arxiv.org/api/query?search_query=all:{clean_phrase}&max_results=3"
    ]

    for api_url in urls:
        try:
            root = ET.fromstring(_http_get_text(api_url, headers=ARXIV_HEADERS, timeout=timeout))
            for entry in root.findall("atom:entry", ATOM_NS):
                parsed = _parse_arxiv_entry(entry)
                if parsed and parsed.get("pdf_url") and is_title_match(title, parsed["title"] or "", threshold=0.75):
                    return parsed
        except Exception:
            pass
    return None


def fetch_neurips_by_title(
    title: str, 
    year: Optional[int] = None, 
    timeout: int = 10
) -> Optional[Dict[str, Any]]:
    """
    检索 NeurIPS (NIPS) 官方开放论文库 (https://proceedings.neurips.cc)
    NeurIPS 官方对所有录用论文永久提供完整免费 PDF 下载
    """
    if not title or not title.strip():
        return None

    clean_title = re.sub(r'[^a-zA-Z0-9\s]', '', title).lower().strip()
    title_words = [w for w in clean_title.split() if len(w) > 3][:5]
    if not title_words:
        return None

    years_to_check = [year] if year else range(2023, 2005, -1)
    for y in years_to_check:
        if not y:
            continue
        url = f"https://proceedings.neurips.cc/paper_files/paper/{y}"
        try:
            html = _http_get_text(url, timeout=timeout)
            if html:
                for m in re.finditer(r'<a[^>]+href="([^"]+Abstract\.html)"[^>]*>([^<]+)</a>', html):
                    cand_link = m.group(1)
                    cand_title = m.group(2).strip()
                    cand_clean = re.sub(r'[^a-zA-Z0-9\s]', '', cand_title).lower().strip()

                    matched = False
                    if clean_title in cand_clean or cand_clean in clean_title:
                        matched = True
                    elif len(title_words) >= 3 and all(w in cand_clean for w in title_words):
                        matched = True

                    if matched:
                        pdf_link = cand_link.replace("-Abstract.html", "-Paper.pdf").replace("/hash/", "/file/")
                        if not pdf_link.startswith("http"):
                            pdf_link = f"https://proceedings.neurips.cc{pdf_link}"
                        return {
                            "title": cand_title,
                            "abstract": "",
                            "pdf_url": pdf_link,
                            "pdf_candidates": [pdf_link],
                            "source": f"NeurIPS {y}"
                        }
        except Exception:
            pass
    return None


def fetch_acl_anthology_by_title(
    title: str,
    year: Optional[int] = None,
    raw_text: Optional[str] = None,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """
    检索 ACL Anthology 官方开放论文库 (https://aclanthology.org)
    覆盖 ACL / EMNLP / NAACL / EACL / COLING / CL / TACL / CoNLL 等计算语言学顶会与期刊免费 PDF
    """
    if not title or not title.strip() or not year:
        return None

    raw_lower = (raw_text or "").lower()
    venues = []
    if "empirical methods" in raw_lower or "emnlp" in raw_lower:
        venues.append("emnlp")
    if "naacl" in raw_lower or "north american" in raw_lower or "human language technology" in raw_lower:
        venues.append("naacl")
    if "acl" in raw_lower or "association for computational linguistics" in raw_lower:
        venues.append("acl")
    if "coling" in raw_lower or "international conference on computational linguistics" in raw_lower:
        venues.append("coling")
    if "eacl" in raw_lower or "european chapter" in raw_lower:
        venues.append("eacl")
    if "computational linguistics" in raw_lower:
        venues.append("cl")
    if "tacl" in raw_lower:
        venues.append("tacl")
    if "conll" in raw_lower:
        venues.append("conll")
    if not venues:
        venues = ["acl", "emnlp", "naacl", "cl"]

    for v in dict.fromkeys(venues):
        url = f"https://aclanthology.org/events/{v}-{year}/"
        try:
            html = _http_get_text(url, timeout=timeout)
            if not html:
                continue
            for m in re.finditer(
                r'<strong><a[^>]+href=["\x27]?/([A-Za-z0-9\.\-]+)/?["\x27]?[^>]*>(.*?)</a></strong>',
                html,
            ):
                acl_id = m.group(1).strip()
                cand_title = re.sub(r"<[^>]+>", "", m.group(2)).strip()
                if is_title_match(title, cand_title, threshold=0.75):
                    pdf_link = f"https://aclanthology.org/{acl_id}.pdf"
                    return {
                        "title": cand_title,
                        "abstract": "",
                        "pdf_url": pdf_link,
                        "pdf_candidates": [pdf_link],
                        "source": f"ACL Anthology ({acl_id})",
                    }
        except Exception:
            pass
    return None


def fetch_jku_archive_by_title(
    title: str,
    raw_text: Optional[str] = None,
    timeout: int = 10,
) -> Optional[Dict[str, Any]]:
    """检索 JKU Linz 机器学习实验室公开论文归档库 (覆盖 LSTM / Gradient Flow 等奠基文献)"""
    if not title or not title.strip():
        return None
    raw_lower = (raw_text or "").lower()
    if raw_lower and not any(k in raw_lower for k in ("hochreiter", "schmidhuber", "recurrent", "memory")):
        return None
    base_url = "https://www.bioinf.jku.at/publications/"
    try:
        html = _http_get_text(f"{base_url}older.html", timeout=timeout)
        for m in re.finditer(
            r'<td><b>Title:</b></td>\s*<td>(.*?)(?:</td>|<td>).*?href=["\x27]([^"\x27]*\.pdf)["\x27]',
            html,
            re.I | re.S,
        ):
            cand_title = " ".join(re.sub(r"<[^>]+>", "", m.group(1)).split())
            if is_title_match(title, cand_title, threshold=0.75):
                rel = m.group(2).strip().lstrip("./")
                pdf_link = rel if rel.startswith("http") else f"{base_url}{rel}"
                return {
                    "title": cand_title,
                    "abstract": "",
                    "pdf_url": pdf_link,
                    "pdf_candidates": [pdf_link],
                    "source": "JKU Linz Archive",
                }
    except Exception:
        pass
    return None


def fetch_openalex_by_title(title: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    【第二梯队】通过 OpenAlex 全球学术图谱检索文献及最佳开放获取 PDF (覆盖 2.5 亿篇)
    """
    if not title or not title.strip():
        return None

    clean_q = " ".join(re.findall(r"[a-zA-Z0-9]+", title))
    query_str = urllib.parse.quote_plus((clean_q or title.strip())[:150])
    api_urls = [
        f"https://api.openalex.org/works?filter=title.search:{query_str}&per-page=5&mailto=citetrace@gmail.com",
        f"https://api.openalex.org/works?search={query_str}&per-page=5&mailto=citetrace@gmail.com",
    ]

    fallback_match: Optional[Dict[str, Any]] = None
    for api_url in api_urls:
        try:
            data = _http_get_json(api_url, timeout=timeout)
            if data:
                results = data.get("results", [])
                if not results:
                    continue

                for work in results:
                    fetched_title = work.get("title", "")
                    if not is_title_match(title, fetched_title, threshold=0.75):
                        continue

                    # 提取开放获取 PDF 直链
                    pdf_candidates = []
                    best_oa = work.get("best_oa_location") or {}
                    if isinstance(best_oa, dict) and best_oa.get("pdf_url"):
                        pdf_candidates.append(_normalize_pdf_url(best_oa.get("pdf_url")))

                    for loc in work.get("locations", []):
                        p_url = _normalize_pdf_url(loc.get("pdf_url"))
                        landing = _normalize_pdf_url(loc.get("landing_page_url") or "")
                        if p_url and p_url not in pdf_candidates:
                            pdf_candidates.append(p_url)
                        if landing and landing.lower().endswith(".pdf") and landing not in pdf_candidates:
                            pdf_candidates.append(landing)

                    oa_info = work.get("open_access") or {}
                    oa_u = _normalize_pdf_url(oa_info.get("oa_url")) if isinstance(oa_info, dict) else None
                    if oa_u and oa_u not in pdf_candidates:
                        pdf_candidates.append(oa_u)

                    # 优先选择非已知 paywall 网关的开放直链
                    clean_candidates = [u for u in pdf_candidates if u and not _is_blocked_url(u)]
                    pdf_url = clean_candidates[0] if clean_candidates else None

                    doi = (work.get("doi") or "").replace("https://doi.org/", "")
                    if not pdf_url and doi:
                        pdf_url = resolve_doi_to_pdf(doi)
                        if pdf_url:
                            clean_candidates.insert(0, pdf_url)

                    inv = work.get("abstract_inverted_index") or {}
                    abstract = ""
                    if isinstance(inv, dict) and inv:
                        words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
                        abstract = " ".join(w for _, w in words)

                    entry = {
                        "title": fetched_title,
                        "abstract": abstract,
                        "pdf_url": pdf_url,
                        "doi": doi,
                        "pdf_candidates": clean_candidates or [u for u in pdf_candidates if u],
                    }
                    if pdf_url:
                        return entry
                    if doi and fallback_match is None:
                        fallback_match = entry
        except Exception:
            pass
    return fallback_match


def fetch_semantic_scholar_by_title(title: str, timeout: int = 10) -> Optional[Dict[str, str]]:
    """
    【第三梯队】通过 Semantic Scholar API 检索文献及 openAccessPdf (覆盖 2.1 亿篇)
    """
    if not title or not title.strip():
        return None

    query_str = urllib.parse.quote_plus(title.strip()[:150])
    api_url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query_str}&limit=1&fields=title,abstract,openAccessPdf,externalIds"

    headers = dict(DEFAULT_HEADERS)
    s2_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if s2_key:
        headers["x-api-key"] = s2_key

    try:
        data = _http_get_json(api_url, headers=headers, timeout=timeout)
        if data:
            items = data.get("data", [])
            if not items:
                return None

            item = items[0]
            fetched_title = item.get("title", "")
            if not is_title_match(title, fetched_title, threshold=0.75):
                return None

            pdf_url = None
            oa_pdf = item.get("openAccessPdf")
            if isinstance(oa_pdf, dict):
                pdf_url = _normalize_pdf_url(oa_pdf.get("url"))

            ext_ids = item.get("externalIds") or {}
            arxiv_id = ext_ids.get("ArXiv")
            doi = ext_ids.get("DOI")

            if not pdf_url and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            if not pdf_url and doi:
                pdf_url = resolve_doi_to_pdf(doi)

            if pdf_url and not _is_blocked_url(pdf_url):
                return {
                    "title": fetched_title,
                    "abstract": item.get("abstract", "") or "",
                    "pdf_url": pdf_url,
                    "arxiv_id": arxiv_id,
                    "doi": doi
                }
    except Exception:
        pass
    return None


def fetch_unpaywall_record_by_doi(doi: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    【第四梯队逆向工具】通过 Unpaywall API 把 DOI 逆向解析为文献元数据与开放获取 PDF 候选列表
    """
    if not doi:
        return None

    clean_doi = doi.strip()
    clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean_doi)
    clean_doi = re.sub(r"^doi:\s*", "", clean_doi)
    api_url = f"https://api.unpaywall.org/v2/{urllib.parse.quote(clean_doi, safe='/')}?email=citetrace@gmail.com"

    try:
        data = _http_get_json(api_url, timeout=timeout)
        if not isinstance(data, dict):
            return None
        locs = []
        if data.get("best_oa_location"):
            locs.append(data["best_oa_location"])
        for loc in (data.get("oa_locations") or []):
            if loc not in locs:
                locs.append(loc)

        candidates = []
        for loc in locs:
            if isinstance(loc, dict):
                cand = _normalize_pdf_url(loc.get("url_for_pdf") or loc.get("url"))
                if cand and not _is_blocked_url(cand) and "doi.org/" not in cand:
                    if cand not in candidates:
                        candidates.append(cand)

        title = data.get("title")
        year = data.get("year")
        return {
            "title": title,
            "year": year,
            "doi": clean_doi,
            "pdf_url": candidates[0] if candidates else None,
            "pdf_candidates": candidates,
        }
    except Exception:
        return None


def fetch_unpaywall_by_doi(doi: str, timeout: int = 10) -> Optional[str]:
    """
    【第四梯队逆向工具】通过 Unpaywall API 把 DOI 逆向解析为合法开放获取 PDF 直链
    """
    rec = fetch_unpaywall_record_by_doi(doi, timeout=timeout)
    return rec["pdf_url"] if rec else None


def fetch_crossref_by_doi(doi: str, timeout: int = 10) -> Optional[Dict[str, Any]]:
    """
    通过 Crossref API 根据 DOI 精确反查文献元数据与发布商直链
    """
    if not doi:
        return None
    clean_doi = doi.strip()
    clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean_doi)
    clean_doi = re.sub(r"^doi:\s*", "", clean_doi)
    api_url = f"https://api.crossref.org/works/{urllib.parse.quote(clean_doi, safe='/')}"
    try:
        data = _http_get_json(api_url, timeout=timeout)
        if not isinstance(data, dict):
            return None
        message = data.get("message", {})
        title_list = message.get("title", [])
        title = title_list[0] if title_list else None
        year = None
        issued = message.get("issued", {}).get("date-parts", [])
        if issued and issued[0]:
            try:
                year = int(issued[0][0])
            except (ValueError, TypeError):
                pass

        candidates = []
        for link in message.get("link", []):
            if isinstance(link, dict) and "pdf" in link.get("content-type", "").lower():
                u = _normalize_pdf_url(link.get("URL"))
                if u and not _is_blocked_url(u) and u not in candidates:
                    candidates.append(u)

        return {
            "title": title,
            "year": year,
            "doi": clean_doi,
            "pdf_url": candidates[0] if candidates else None,
            "pdf_candidates": candidates,
        }
    except Exception:
        return None


def resolve_doi_to_pdf(doi: Optional[str]) -> Optional[str]:
    """
    知名学术会议与预印本平台的 DOI 极速直链转换器
    支持 ACL Anthology (EMNLP/ACL/NAACL), arXiv, BioRxiv 等
    """
    if not doi:
        return None
    d = doi.strip()
    # 1. ACL Anthology (10.18653/v1/...) -> aclanthology.org/{id}.pdf
    if "10.18653/v1/" in d:
        acl_id = re.sub(r"^.*?10\.18653/v1/", "", d).strip()
        return f"https://aclanthology.org/{acl_id}.pdf"
    # 2. arXiv DOI (10.48550/arXiv....) -> arxiv.org/pdf/{aid}.pdf
    if "10.48550/arxiv." in d.lower():
        aid = re.sub(r"^.*?10\.48550/arxiv\.", "", d, flags=re.I).strip()
        return f"https://arxiv.org/pdf/{aid}.pdf"
    return None


def fetch_crossref_by_title(
    title: str, 
    raw_bibliographic: Optional[str] = None,
    timeout: int = 10
) -> Optional[Dict[str, str]]:
    """
    【第五梯队】通过 Crossref API 根据标题/引文条目检索文献元数据与 DOI
    - 优先使用 raw_bibliographic (包含作者、标题、年份等综合特征) 进行联合检索
    - 每次检索拉取 Top-5 条候选，并执行严格相似度与词序序列筛选
    """
    if not title or not title.strip():
        return None

    queries = []
    # 策略 1: 若提供了原始参考文献段落 (例如含作者名 Hochreiter)，优先用 query.bibliographic 联合检索
    if raw_bibliographic and len(raw_bibliographic.strip()) > 10:
        clean_raw = re.sub(r'^\s*\[[^\]]+\]\s*', '', raw_bibliographic).strip()
        queries.append(f"query.bibliographic={urllib.parse.quote_plus(clean_raw[:200])}")
    # 策略 2: 纯标题检索
    queries.append(f"query.title={urllib.parse.quote_plus(title.strip()[:150])}")

    for q_param in queries:
        api_url = f"https://api.crossref.org/works?{q_param}&rows=5"
        for attempt in range(2):
            try:
                data = _http_get_json(api_url, timeout=timeout)
                if data:
                    items = data.get("message", {}).get("items", [])
                    if not items:
                        break

                    for item in items:
                        titles = item.get("title", [])
                        fetched_title = titles[0] if titles else ""
                        if not fetched_title:
                            continue

                        if not is_title_match(title, fetched_title, threshold=0.75):
                            continue

                        doi = item.get("DOI", "")
                        pdf_url = resolve_doi_to_pdf(doi)

                        return {
                            "title": fetched_title,
                            "abstract": item.get("abstract", ""),
                            "pdf_url": pdf_url,
                            "doi": doi
                        }
                    break
            except Exception:
                if attempt == 0:
                    time.sleep(2)
                else:
                    break
    return None


def _apply_paper_info(ref: CitationRef, info: Dict[str, Any], pdf_url: Optional[str]) -> None:
    ref.title = ref.title or info.get("title")
    if not ref.year and info.get("year"):
        try:
            ref.year = int(info["year"])
        except (ValueError, TypeError):
            pass
    ref.pdf_url = _normalize_pdf_url(pdf_url)
    ref.pdf_candidates = [
        nu for u in info.get("pdf_candidates", [])
        if (nu := _normalize_pdf_url(u))
    ]
    if ref.pdf_url and ref.pdf_url not in ref.pdf_candidates:
        ref.pdf_candidates.insert(0, ref.pdf_url)
    ref.status = CitationStatus.FETCHED if ref.pdf_url else CitationStatus.PAYWALL_LOCKED


def get_global_cache_key(ref: CitationRef) -> Optional[str]:
    """
    计算文献全局稳态缓存键，杜绝不同主论文之间因序号相同（如 [1]、[17]）产生键冲突。

    优先级：
    1. arXiv ID -> arxiv_<id>
    2. DOI 哈希 -> doi_<hash_16>
    3. 规范化(title + year + first_author) 哈希 -> title_<hash_16>

    严禁将 [1]、[17] 或 ref.key 作为全局缓存键！
    若无稳定身份，返回 None（仅保留在当前主论文局部目录，不写入全局 metadata 缓存）。
    """
    if ref.arxiv_id:
        clean_arxiv = ref.arxiv_id.strip().lower()
        clean_arxiv = re.sub(r"^arxiv:\s*", "", clean_arxiv)
        safe_arxiv = re.sub(r"[^a-z0-9._-]", "_", clean_arxiv)
        return f"arxiv_{safe_arxiv}"

    if ref.doi:
        clean_doi = ref.doi.strip().lower()
        clean_doi = re.sub(r"^https?://(?:dx\.)?doi\.org/", "", clean_doi)
        clean_doi = re.sub(r"^doi:\s*", "", clean_doi)
        doi_hash = hashlib.sha256(clean_doi.encode("utf-8")).hexdigest()[:16]
        return f"doi_{doi_hash}"

    if ref.title and len(ref.title.strip()) >= 5:
        norm_title = re.sub(r"[^a-z0-9]+", " ", ref.title.strip().lower()).strip()
        if len(norm_title) >= 5:
            year_str = str(ref.year).strip() if ref.year else ""
            author_str = ""
            if ref.authors:
                first_auth = ref.authors[0].strip().lower()
                first_auth_surname = first_auth.split()[-1] if first_auth else ""
                author_str = re.sub(r"[^a-z0-9]+", "", first_auth_surname)
            sig = f"{norm_title}|{year_str}|{author_str}".encode("utf-8")
            title_hash = hashlib.sha256(sig).hexdigest()[:16]
            return f"title_{title_hash}"

    return None


def enrich_citation_with_cache(
    ref: CitationRef, 
    cache_dir: str = "cache/papers"
) -> Tuple[CitationRef, Optional[str]]:
    """
    【主入口（纯函数）】：五级自适应降级阶梯 + 本地磁盘缓存
    1. 优先查本地磁盘缓存（0 秒响应）
    2. 阶梯一：arXiv ID 直查 (100% 精确)
    3. 阶梯二：arXiv 标题检索 (对 AI/ML 预印本极佳)
    4. 阶梯三：OpenAlex 全球学术知识图谱检索 (覆盖 2.5 亿篇，全球机构知识库 OA PDF)
    5. 阶梯四：Semantic Scholar 检索 (覆盖 2.1 亿篇，提取 openAccessPdf)
    6. 阶梯五：Crossref 检索 DOI -> 联合 Unpaywall 逆向抓取合法免费 PDF
    7. 均失败：标记为 PAYWALL_LOCKED
    """
    os.makedirs(cache_dir, exist_ok=True)

    meta_cache_dir = os.path.join(cache_dir, ".metadata_cache")
    os.makedirs(meta_cache_dir, exist_ok=True)

    cache_key = get_global_cache_key(ref)
    cache_file = os.path.join(meta_cache_dir, f"{cache_key}.json") if cache_key else None
    if not (cache_file and os.path.isfile(cache_file)) and ref.key:
        fallback_file = os.path.join(meta_cache_dir, f"{ref.key}.json")
        if os.path.isfile(fallback_file):
            cache_file = fallback_file

    # 1. 命中本地缓存 (仅在存在稳态全局缓存键或明确键名时尝试读取)
    if cache_file and os.path.isfile(cache_file):
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                cached_data = json.load(f)
            cached_pdf = _normalize_pdf_url(cached_data.get("pdf_url"))
            cached_doi = cached_data.get("doi")
            cached_title = cached_data.get("title")

            # 关键：检查缓存中的标题是否与当前文献真正匹配（防止旧版词序倒置误匹配的错误缓存被重复读取）
            if ref.title and cached_title and not is_title_match(ref.title, cached_title):
                cached_pdf = None

            # 关键：检查缓存中的链接是否是伪链接 (如旧版写入的 https://doi.org/...)
            # 若是纯 DOI 页面，尝试自动解析为真实 PDF 或穿透至新阶梯重新查找
            if cached_pdf and "doi.org/" in cached_pdf:
                real_pdf = resolve_doi_to_pdf(cached_doi or cached_pdf)
                if real_pdf:
                    cached_data["pdf_url"] = real_pdf
                    cached_pdf = real_pdf
                    with open(cache_file, "w", encoding="utf-8") as fw:
                        json.dump(cached_data, fw, indent=2, ensure_ascii=False)
                else:
                    cached_pdf = None  # 标记为无效，穿透到底下多梯队重新查！

            # 关键：检查缓存中的链接是否属于已知付费墙拦截网关 (如 dl.acm.org)
            if cached_pdf and _is_blocked_url(cached_pdf):
                cached_cands = [_normalize_pdf_url(u) for u in cached_data.get("pdf_candidates", [])]
                clean_cands = [u for u in cached_cands if u and not _is_blocked_url(u)]
                if clean_cands:
                    cached_pdf = clean_cands[0]
                    cached_data["pdf_url"] = cached_pdf
                else:
                    cached_pdf = None

            if cached_pdf:
                _apply_paper_info(ref, cached_data, cached_pdf)
                success_path = f"阶梯 1 [本地磁盘缓存] -> {os.path.abspath(cache_file)}"
                return ref, success_path
        except Exception:
            pass


    # 2. 联网调度多梯队流水线（若前序阶梯仅拿到元数据而无有效 pdf_url，继续向下游阶梯穿透检索）
    paper_info: Optional[Dict[str, Any]] = None
    step_source: Optional[str] = None

    def _has_pdf() -> bool:
        return bool(paper_info and paper_info.get("pdf_url"))

    # 阶梯 2: arXiv ID 直查 (100% 精确)
    if ref.arxiv_id:
        paper_info = fetch_arxiv_by_id(ref.arxiv_id)
        if _has_pdf():
            step_source = f"阶梯 2 [arXiv ID 直查] -> {ref.arxiv_id}"

    # 阶梯 3: arXiv 标题检索 (对 AI/ML 预印本极佳)
    if not _has_pdf() and ref.title:
        arx_info = fetch_arxiv_by_title(ref.title)
        if arx_info and arx_info.get("pdf_url"):
            paper_info = arx_info
            step_source = f"阶梯 3 [arXiv 标题检索] -> {ref.title[:30]}..."
            if paper_info.get("arxiv_id") and not ref.arxiv_id:
                ref.arxiv_id = paper_info["arxiv_id"]

    def _try_neurips(stage_label: str) -> None:
        nonlocal paper_info, step_source
        nips_info = fetch_neurips_by_title(ref.title, year=ref.year)
        if nips_info and nips_info.get("pdf_url"):
            paper_info = nips_info
            step_source = f"{stage_label} -> {nips_info.get('source', '')}"

    raw_lower = (ref.raw_text or "").lower()
    # 阶梯 3.5: NeurIPS / NIPS 官方论文库检索 (覆盖所有 NeurIPS 录用论文，100% 官方免费 PDF)
    is_nips = any(k in raw_lower for k in ("nips", "neurips", "neural information processing systems"))
    if not _has_pdf() and ref.title and (is_nips or ref.year):
        _try_neurips("阶梯 3.5 [NeurIPS 官方仓储直达]")

    # 阶梯 3.6: ACL Anthology 官方论文库检索 (覆盖 ACL / EMNLP / NAACL / COLING / CL 等计算语言学文献)
    is_acl = any(
        k in raw_lower
        for k in ("acl", "emnlp", "naacl", "coling", "empirical methods", "computational linguistics", "human language")
    )
    if not _has_pdf() and ref.title and ref.year and is_acl:
        acl_info = fetch_acl_anthology_by_title(ref.title, year=ref.year, raw_text=ref.raw_text)
        if acl_info and acl_info.get("pdf_url"):
            paper_info = acl_info
            step_source = f"阶梯 3.6 [ACL Anthology 官方仓储直达] -> {acl_info.get('source', '')}"

    # 阶梯 3.7: JKU Linz 机器学习实验室公开论文归档库直达 (覆盖 Hochreiter / Schmidhuber 早期奠基文献)
    if not _has_pdf() and ref.title and any(k in raw_lower for k in ("hochreiter", "schmidhuber")):
        jku_info = fetch_jku_archive_by_title(ref.title, raw_text=ref.raw_text)
        if jku_info and jku_info.get("pdf_url"):
            paper_info = jku_info
            step_source = f"阶梯 3.7 [JKU 机构开放仓储直达] -> {jku_info.get('source', '')}"

    # 阶梯 4: OpenAlex 全球学术图谱检索 (覆盖 2.5 亿篇，提取高校机构开放版 PDF)
    if not _has_pdf() and ref.title:
        oa_info = fetch_openalex_by_title(ref.title)
        if oa_info:
            if oa_info.get("doi") and not ref.doi:
                ref.doi = oa_info["doi"]
            if oa_info.get("pdf_url") or not paper_info:
                paper_info = oa_info
                step_source = f"阶梯 4 [OpenAlex 全球仓储检索] -> {ref.title[:30]}..."

    # 阶梯 5: Semantic Scholar 检索 (覆盖 2.1 亿篇，提取 openAccessPdf)
    if not _has_pdf() and ref.title:
        s2_info = fetch_semantic_scholar_by_title(ref.title)
        if s2_info:
            if s2_info.get("arxiv_id") and not ref.arxiv_id:
                ref.arxiv_id = s2_info["arxiv_id"]
            if s2_info.get("doi") and not ref.doi:
                ref.doi = s2_info["doi"]
            if s2_info.get("pdf_url") or not paper_info:
                paper_info = s2_info
                step_source = f"阶梯 5 [Semantic Scholar 检索] -> {ref.title[:30]}..."

    # 阶梯 5.5: 若已有 DOI 但尚无 PDF（例如纯 DOI 文献，或前序阶梯解析出的 DOI），优先尝试 Unpaywall 与 DOI 直链解析
    if not _has_pdf() and ref.doi:
        upw_rec = fetch_unpaywall_record_by_doi(ref.doi)
        direct_pdf = resolve_doi_to_pdf(ref.doi)
        # 若 Unpaywall 或直链解析出有效内容
        if (upw_rec and upw_rec.get("pdf_url")) or direct_pdf:
            paper_info = paper_info or {
                "title": ref.title,
                "abstract": "",
                "doi": ref.doi,
                "arxiv_id": ref.arxiv_id,
                "pdf_candidates": [],
            }
            if upw_rec:
                if upw_rec.get("title") and (not ref.title or ref.title == "Unknown"):
                    ref.title = upw_rec["title"]
                    paper_info["title"] = upw_rec["title"]
                if upw_rec.get("year") and not ref.year:
                    try:
                        ref.year = int(upw_rec["year"])
                        paper_info["year"] = ref.year
                    except (ValueError, TypeError):
                        pass
                for c in upw_rec.get("pdf_candidates", []):
                    if c not in paper_info.setdefault("pdf_candidates", []):
                        paper_info["pdf_candidates"].append(c)
                if upw_rec.get("pdf_url") and not paper_info.get("pdf_url"):
                    paper_info["pdf_url"] = upw_rec["pdf_url"]

            if direct_pdf:
                cands = paper_info.setdefault("pdf_candidates", [])
                if direct_pdf not in cands:
                    cands.insert(0, direct_pdf)
                paper_info["pdf_url"] = direct_pdf

            step_source = f"阶梯 5.5 [Unpaywall OA 逆向解析] -> {ref.doi}"
        elif upw_rec and upw_rec.get("title"):
            # Unpaywall 虽无 OA PDF 但含有标题元数据
            if not ref.title or ref.title == "Unknown":
                ref.title = upw_rec["title"]
            if upw_rec.get("year") and not ref.year:
                try:
                    ref.year = int(upw_rec["year"])
                except (ValueError, TypeError):
                    pass
            paper_info = upw_rec
            step_source = f"阶梯 5.5 [Unpaywall 仅元数据/付费墙] -> {ref.doi}"

        # 若至此仍缺少标题或缺少 PDF，通过 Crossref by DOI 补充
        if not ref.title or ref.title == "Unknown" or not _has_pdf():
            cr_rec = fetch_crossref_by_doi(ref.doi)
            if cr_rec:
                if cr_rec.get("title") and (not ref.title or ref.title == "Unknown"):
                    ref.title = cr_rec["title"]
                if cr_rec.get("year") and not ref.year:
                    ref.year = cr_rec["year"]
                if cr_rec.get("pdf_url") and not _has_pdf():
                    for c in cr_rec.get("pdf_candidates", []):
                        if paper_info:
                            if c not in paper_info.setdefault("pdf_candidates", []):
                                paper_info["pdf_candidates"].append(c)
                    if not paper_info:
                        paper_info = cr_rec
                    else:
                        paper_info["pdf_url"] = cr_rec["pdf_url"]
                    step_source = f"阶梯 5.5 [Crossref DOI 直链] -> {ref.doi}"
                elif not paper_info:
                    paper_info = cr_rec
                    step_source = f"阶梯 5.5 [Crossref DOI 元数据补全] -> {ref.doi}"

    # 阶梯 6: Crossref 检索 DOI -> 联合 Unpaywall 逆向抓取合法免费 PDF
    if not _has_pdf() and ref.title:
        cr_info = fetch_crossref_by_title(ref.title, raw_bibliographic=ref.raw_text)
        if cr_info:
            doi = cr_info.get("doi")
            if doi and not ref.doi:
                ref.doi = doi
            oa_rec = fetch_unpaywall_record_by_doi(doi) if doi else None
            if oa_rec and oa_rec.get("pdf_url"):
                cr_info["pdf_url"] = oa_rec["pdf_url"]
                for c in oa_rec.get("pdf_candidates", []):
                    if c not in cr_info.setdefault("pdf_candidates", []):
                        cr_info["pdf_candidates"].append(c)
                paper_info = cr_info
                step_source = f"阶梯 6 [Crossref + Unpaywall OA] -> {doi}"
            elif cr_info.get("pdf_url"):
                paper_info = cr_info
                step_source = f"阶梯 6 [Crossref 预解析直链] -> {doi}"
            elif not paper_info:
                cr_info["pdf_url"] = None
                paper_info = cr_info
                step_source = f"阶梯 6 [Crossref 仅元数据/付费墙保护] -> {doi}"

    # 阶梯 6.5: 机构与学会开放论文库兜底检索 (JKU Linz / ACL Anthology / NeurIPS)
    if not _has_pdf() and ref.title:
        jku_info = fetch_jku_archive_by_title(ref.title, raw_text=ref.raw_text)
        if jku_info and jku_info.get("pdf_url"):
            paper_info = jku_info
            step_source = f"阶梯 6.5 [机构开放仓储兜底] -> {jku_info.get('source', '')}"
        elif ref.year and not is_acl:
            acl_info = fetch_acl_anthology_by_title(ref.title, year=ref.year, raw_text=ref.raw_text)
            if acl_info and acl_info.get("pdf_url"):
                paper_info = acl_info
                step_source = f"阶梯 6.5 [ACL Anthology 仓储兜底] -> {acl_info.get('source', '')}"
        elif not (is_nips or ref.year):
            _try_neurips("阶梯 6.5 [NeurIPS 官方仓储兜底]")

    # 3. 状态更新与缓存写入
    if paper_info:
        _apply_paper_info(ref, paper_info, paper_info.get("pdf_url"))

        if cache_file:
            try:
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump({
                        "title": ref.title,
                        "abstract": paper_info.get("abstract", ""),
                        "pdf_url": ref.pdf_url,
                        "pdf_candidates": ref.pdf_candidates,
                        "doi": ref.doi,
                        "arxiv_id": ref.arxiv_id
                    }, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"[Warn] 缓存写入失败 ({cache_file}): {e}")

        success_path = f"{step_source} -> 成功抓取" + (f"并缓存到 {cache_file}" if cache_file else "")
        return ref, success_path
    else:
        ref.status = CitationStatus.PAYWALL_LOCKED
        return ref, None


@dataclass
class ArtifactValidationResult:
    is_valid: bool
    errors: List[str] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    pdf_path: Optional[Path] = None
    chunks_path: Optional[Path] = None


def validate_paper_artifacts(
    folder: Path | str,
    expected_title: Optional[str] = None,
    verify_pdf_bytes: bool = True,
) -> ArtifactValidationResult:
    """
    严密校验单篇论文缓存目录中的产物契约与完整性。

    必须通过的检查：
    1. metadata.json 存在且为有效 JSON
    2. raw.pdf 存在且大小 >= 100 字节
    3. raw.pdf 魔数必须以 %PDF- 开头，且 PyMuPDF 能成功加载打开
    4. chunks.json 存在，为有效 JSON，且 chunks 列表非空
    5. metadata.json 中记录的 pdf_sha256 与真实 raw.pdf SHA-256 强一致
    6. outline_audit 存在且 passed 为 True
    7. outline 中登记的各 section 文件真实存在
    8. 若传入 expected_title，标题必须大致匹配
    """
    folder = Path(folder)
    errors: List[str] = []

    if not folder.is_dir():
        return ArtifactValidationResult(is_valid=False, errors=["folder_not_found"])

    metadata_path = folder / "metadata.json"
    if not metadata_path.is_file():
        return ArtifactValidationResult(is_valid=False, errors=["missing_metadata"])

    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if not isinstance(metadata, dict):
            return ArtifactValidationResult(is_valid=False, errors=["invalid_metadata_format"])
    except Exception:
        return ArtifactValidationResult(is_valid=False, errors=["corrupt_metadata"])

    # 0. 版本与契约检查
    if metadata.get("artifact_schema_version") != MetadataCache.ARTIFACT_SCHEMA_VERSION:
        errors.append("artifact_schema_version_mismatch")
    if metadata.get("parser_version") != MetadataCache.PARSER_VERSION:
        errors.append("parser_version_mismatch")
    if metadata.get("chunker_version") != MetadataCache.CHUNKER_VERSION:
        errors.append("chunker_version_mismatch")

    # 1. 检查 raw.pdf
    raw_pdf = folder / "raw.pdf"
    if not raw_pdf.is_file():
        errors.append("missing_raw_pdf")
    else:
        pdf_size = raw_pdf.stat().st_size
        if pdf_size < 100:
            errors.append("corrupt_pdf")
        else:
            try:
                with open(raw_pdf, "rb") as f:
                    magic = f.read(5)
                if not magic.startswith(b"%PDF-"):
                    errors.append("corrupt_pdf")
            except Exception:
                errors.append("corrupt_pdf")

        if verify_pdf_bytes and "corrupt_pdf" not in errors:
            try:
                import pymupdf
                with pymupdf.open(raw_pdf) as doc:
                    if len(doc) == 0:
                        errors.append("empty_pdf")
            except Exception:
                errors.append("corrupt_pdf")

    # 2. 检查 chunks.json
    chunks_path = folder / "chunks.json"
    if not chunks_path.is_file():
        errors.append("missing_chunks")
    else:
        try:
            chunks_data = json.loads(chunks_path.read_text(encoding="utf-8"))
            if not isinstance(chunks_data, list) or len(chunks_data) == 0:
                errors.append("empty_chunks")
        except Exception:
            errors.append("corrupt_chunks")

    # 3. 检查 PDF SHA-256
    if "missing_raw_pdf" not in errors and "corrupt_pdf" not in errors:
        try:
            h = hashlib.sha256()
            with open(raw_pdf, "rb") as f:
                for block in iter(lambda: f.read(1 << 20), b""):
                    h.update(block)
            actual_sha = h.hexdigest()
            meta_sha = metadata.get("pdf_sha256")
            if meta_sha and meta_sha.strip().lower() != actual_sha.lower():
                errors.append("sha256_mismatch")
        except Exception:
            errors.append("sha256_check_failed")

    # 4. 检查 outline_audit
    audit = metadata.get("outline_audit")
    if not audit or not isinstance(audit, dict) or not audit.get("passed", False):
        errors.append("outline_audit_failed")

    # 5. 检查 outline 各 section 文件
    outline = metadata.get("outline", [])
    if isinstance(outline, list):
        for item in outline:
            fname = item.get("file")
            if fname:
                sec_root = folder / fname
                sec_sub = folder / "sections" / fname
                if not sec_root.is_file() and not sec_sub.is_file():
                    errors.append("missing_section_file")
                    break

    # 6. 检查标题匹配
    if expected_title:
        paper_title = metadata.get("paper_title") or ""
        if paper_title and not is_title_match(expected_title, paper_title):
            errors.append("title_mismatch")

    is_valid = len(errors) == 0
    return ArtifactValidationResult(
        is_valid=is_valid,
        errors=errors,
        metadata=metadata if is_valid else None,
        pdf_path=raw_pdf if "missing_raw_pdf" not in errors else None,
        chunks_path=chunks_path if "missing_chunks" not in errors else None,
    )


def _quarantine_file(file_path: Path, quarantine_dir: Optional[Path] = None) -> Optional[Path]:
    """将损坏或标题不匹配的 PDF 文件原子移入 .quarantine/ 目录"""
    file_path = Path(file_path)
    if not file_path.is_file():
        return None
    q_dir = quarantine_dir or (file_path.parent / ".quarantine")
    q_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time())
    dest = q_dir / f"{timestamp}_{file_path.name}"
    try:
        shutil.move(str(file_path), str(dest))
        return dest
    except Exception as e:
        logger.warning("移入隔离区失败 (%s -> %s): %s", file_path, dest, e)
        file_path.unlink(missing_ok=True)
        return None


def fetch_paper_document(
    ref: CitationRef, 
    cache_dir: str = "cache/papers",
    paper_id: Optional[str] = None,
    global_cache_dir: Optional[str] = "cache/papers"
) -> Tuple[CitationRef, Dict[str, Any]]:
    """
    【一站式文献获取与切分入口】：
    1. 确定序列号标识（如 [1]、[17]），并建立专属于该序列号的存放子目录
    2. 优先检索本地全局缓存库（若已下载对应 arXiv/DOI，直接秒级复用，杜绝重复网络请求）
    3. 若无本地缓存，调度 5 级学术阶梯在线检索与抓取
    4. 调用 PDFProcessor 切分各章节 Markdown
    5. 将正文 PDF 规范命名为序列号文件（如 [17].pdf 及 17.pdf），便于用户一目了然直查
    """
    os.makedirs(cache_dir, exist_ok=True)

    # 1. 规范化序列号标识 (如 [17] 或 17)
    raw_key = paper_id or ref.key or ref.arxiv_id or ref.doi or "unknown"
    clean_digits = re.search(r'\d+', raw_key)
    if clean_digits and re.match(r"^\[?\d+\]?$", raw_key.strip()):
        safe_dir_name = f"[{clean_digits.group(0)}]"
    else:
        safe_dir_name = sanitize_filename(raw_key)
    target_paper_dir = Path(cache_dir) / safe_dir_name
    target_paper_dir.mkdir(parents=True, exist_ok=True)
    pdf_processor = PDFProcessor(cache_dir=cache_dir)

    # 2. 检查当前目录下是否已经切分好并且有效（且确实是所请求的那篇文献）
    raw_pdf_path = target_paper_dir / "raw.pdf"
    chunks_path = target_paper_dir / "chunks.json"
    metadata_path = target_paper_dir / "metadata.json"

    # 若 metadata 与 PDF 存在但 chunks 缺失或 chunker_version 不匹配，尝试自动补齐重建
    if raw_pdf_path.is_file() and metadata_path.is_file():
        try:
            curr_meta = json.loads(metadata_path.read_text(encoding="utf-8"))
        except Exception:
            curr_meta = {}
        needs_rechunk = False
        if not chunks_path.is_file() or chunks_path.stat().st_size <= 2:
            needs_rechunk = True
        elif curr_meta.get("chunker_version") != MetadataCache.CHUNKER_VERSION:
            needs_rechunk = True

        if needs_rechunk:
            try:
                from citetrace_pdf.chunker import chunk_paper_folder
                chunk_paper_folder(target_paper_dir)
                curr_meta["chunker_version"] = MetadataCache.CHUNKER_VERSION
                metadata_path.write_text(json.dumps(curr_meta, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception as e:
                logger.warning("自动重建 chunks.json 失败: %s", e)

    audit = validate_paper_artifacts(target_paper_dir, expected_title=ref.title)
    force_reprocess = False
    if audit.is_valid and audit.metadata:
        ref.status = CitationStatus.FETCHED
        ref.pdf_url = str(raw_pdf_path)
        ref.last_failure = None
        if (not ref.title or ref.title == "Unknown") and audit.metadata.get("paper_title"):
            ref.title = audit.metadata["paper_title"]
        return ref, audit.metadata
    elif metadata_path.is_file() or raw_pdf_path.is_file():
        logger.warning(
            "本地缓存《%s》产物契约校验未通过 (%s)，判定为不完整/错误产物，移入隔离区并强制重新获取: %s",
            target_paper_dir, audit.errors, ref.title
        )
        if raw_pdf_path.is_file():
            _quarantine_file(raw_pdf_path)
        metadata_path.unlink(missing_ok=True)
        chunks_path.unlink(missing_ok=True)
        force_reprocess = True

    # 3. 联网或查库丰富文献元数据
    resolved_ref, _ = enrich_citation_with_cache(ref, cache_dir=global_cache_dir or cache_dir)

    # 4. 汇总所有可能的候选 URL / 路径，绝不把已失效的本地 raw.pdf 当作唯一来源
    candidate_urls: List[str] = []

    # 4.1 本地已校验的 raw.pdf（仅当有效且无需强制重跑时才加入）
    if raw_pdf_path.is_file() and not force_reprocess and pdf_processor.loader.is_valid_pdf(raw_pdf_path):
        candidate_urls.append(str(raw_pdf_path))
    # 4.2 本地全局稳定缓存候选（基于 arXiv ID / DOI / Title Hash）
    elif global_cache_dir and os.path.isdir(global_cache_dir):
        g_path = Path(global_cache_dir)
        global_key = get_global_cache_key(resolved_ref)
        id_candidates = []
        if resolved_ref.arxiv_id:
            id_candidates.append(f"arxiv_{sanitize_filename(resolved_ref.arxiv_id)}")
            id_candidates.append(sanitize_filename(resolved_ref.arxiv_id))
        if resolved_ref.doi:
            doi_hash = hashlib.sha256(resolved_ref.doi.strip().lower().encode("utf-8")).hexdigest()[:16]
            id_candidates.append(f"doi_{doi_hash}")
        if global_key and global_key not in id_candidates:
            id_candidates.append(global_key)

        for sid in id_candidates:
            for cand in (g_path / sid / "raw.pdf", g_path / f"{sid}.pdf"):
                if pdf_processor.loader.is_valid_pdf(cand):
                    candidate_urls.append(str(cand))
                    break
            if candidate_urls:
                break

    # 4.3 网络候选直链（确保即使有本地候选，剩余候选也不被阻断）
    if resolved_ref.pdf_url and resolved_ref.pdf_url not in candidate_urls:
        candidate_urls.append(resolved_ref.pdf_url)
    for u in getattr(resolved_ref, "pdf_candidates", []):
        if u and u not in candidate_urls:
            candidate_urls.append(u)

    # 付费墙网关过滤
    download_urls = [u for u in candidate_urls if not _is_blocked_url(u)]
    blocked = [u for u in candidate_urls if _is_blocked_url(u)]
    if blocked and not download_urls:
        resolved_ref.status = CitationStatus.PAYWALL_LOCKED
        resolved_ref.last_failure = FailureReason.PAYWALL_LOCKED.value
        logger.warning("仅有付费墙网关链接，拒绝抓取不可校验的来源: %s", blocked[0])

    if not download_urls:
        if resolved_ref.status == CitationStatus.PAYWALL_LOCKED:
            print(f"[Info] 文献受版权/付费墙保护 (PAYWALL): {resolved_ref.title or raw_key}")
        else:
            print(f"[Warn] 无法获取文献下载链接: {resolved_ref.title or raw_key}")
        return resolved_ref, {}

    # 5. 调用 PDFProcessor 落地并切分，遍历候选并记录 attempts
    metadata: Dict[str, Any] = {}
    attempts: List[Dict[str, Any]] = []
    last_err = None
    title_mismatch = False

    for u in download_urls:
        attempt_record: Dict[str, Any] = {"url": u}
        try:
            got = pdf_processor.process_paper(
                paper_id=safe_dir_name,
                pdf_url=u,
                force=True if force_reprocess or attempts else False,
                expected_title=resolved_ref.title,
            )
        except Exception as e:
            last_err = e
            attempt_record["status"] = "download_failed"
            attempt_record["error"] = str(e)
            attempts.append(attempt_record)
            continue

        if not got:
            attempt_record["status"] = "download_failed"
            attempt_record["error"] = "process_paper returned empty"
            attempts.append(attempt_record)
            continue

        landed_folder = Path(got.get("folder") or target_paper_dir)
        matched, verifiable = verify_document_title(resolved_ref.title or "", got, landed_folder)
        if verifiable and not matched:
            logger.warning(
                "标题校验不通过: 请求《%s》实得《%s》，移入隔离区并尝试下一候选: %s",
                resolved_ref.title, got.get("paper_title"), u
            )
            title_mismatch = True
            attempt_record["status"] = "title_mismatch"
            attempt_record["actual_title"] = got.get("paper_title")
            attempts.append(attempt_record)

            # 移入隔离区
            if (landed_folder / "raw.pdf").is_file():
                _quarantine_file(landed_folder / "raw.pdf")
            (landed_folder / "metadata.json").unlink(missing_ok=True)
            force_reprocess = True
            continue

        # 校验成功
        attempt_record["status"] = "success"
        attempts.append(attempt_record)
        metadata = got
        resolved_ref.pdf_url = u
        resolved_ref.status = CitationStatus.FETCHED
        resolved_ref.last_failure = None
        break

    # 记录 attempts.json 日志
    try:
        (target_paper_dir / "attempts.json").write_text(
            json.dumps({"attempts": attempts}, indent=2, ensure_ascii=False),
            encoding="utf-8"
        )
    except Exception as e:
        logger.warning("记录 attempts.json 失败: %s", e)

    if not metadata:
        if title_mismatch:
            resolved_ref.last_failure = FailureReason.TITLE_MISMATCH.value
        elif last_err is not None:
            err_msg = str(last_err).lower()
            if any(k in err_msg for k in ["invalid pdf", "不是有效 pdf", "magic", "非有效 pdf", "无法解析"]):
                resolved_ref.last_failure = FailureReason.INVALID_PDF.value
            elif any(k in err_msg for k in ["rate limit", "429"]):
                resolved_ref.last_failure = FailureReason.RATE_LIMITED.value
            elif any(k in err_msg for k in ["timeout", "timed out"]):
                resolved_ref.last_failure = FailureReason.NETWORK_TIMEOUT.value
            elif any(k in err_msg for k in ["download", "404", "connection"]):
                resolved_ref.last_failure = FailureReason.DOWNLOAD_FAILED.value
            else:
                resolved_ref.last_failure = FailureReason.PARSE_FAILED.value
            logger.warning("文献解析失败 [%s]: %s", safe_dir_name, last_err)
        else:
            resolved_ref.last_failure = FailureReason.DOWNLOAD_FAILED.value
        return resolved_ref, {}

    if (not resolved_ref.title or resolved_ref.title == "Unknown") and metadata.get("paper_title"):
        resolved_ref.title = metadata["paper_title"]

    return resolved_ref, metadata


class AcademicFetcher:
    """面向对象的文献抓取门面类"""
    def __init__(self, cache_dir: str = "cache/papers", global_cache_dir: Optional[str] = "cache/papers"):
        self.cache_dir = cache_dir
        self.global_cache_dir = global_cache_dir

    def fetch_paper_document(
        self,
        citation_ref: CitationRef,
        cache_dir: Optional[str] = None,
        paper_id: Optional[str] = None
    ) -> Tuple[CitationRef, Dict[str, Any]]:
        target_dir = cache_dir or self.cache_dir
        return fetch_paper_document(
            citation_ref,
            cache_dir=target_dir,
            paper_id=paper_id,
            global_cache_dir=self.global_cache_dir,
        )
