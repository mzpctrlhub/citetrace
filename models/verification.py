from enum import Enum


class CitationVerdict(str, Enum):
    """学术引用细粒度核查判决"""
    CORRECT = "correct"                     # 正确引用 (证据充分支持)
    DATA_ERROR = "data_error"               # 数据不正确 (指标、数值不符)
    STATEMENT_ERROR = "statement_error"     # 表述不正确 (因果倒置、断章取义、结论相反)
    INSUFFICIENT = "insufficient"           # 证据不足 (未能获取正文或当前片段不足)
    INVALID_CITATION = "invalid_citation"   # 无效引用 (整篇论文遍历完毕仍无相关证据)
    SEARCH_INCOMPLETE = "search_incomplete" # 检索未完成 (轮次耗尽仍有片段未检索，不下确定性结论)


class FailureReason(str, Enum):
    """结构化失败原因：区分"文献不存在 / 下载失败 / 解析失败 / 模型故障"等不同性质的问题"""
    DOCUMENT_NOT_FOUND = "document_not_found"    # 学术平台检索不到该文献
    RATE_LIMITED = "rate_limited"                # API/服务限流 (429)
    NETWORK_TIMEOUT = "network_timeout"          # 网络连接或读取超时
    PAYWALL_LOCKED = "paywall_locked"            # 付费墙/版权限制
    DOWNLOAD_FAILED = "download_failed"          # 下载异常
    INVALID_PDF = "invalid_pdf"                  # 下载内容不是有效 PDF
    TITLE_MISMATCH = "title_mismatch"            # 拿到的 PDF 经标题校验确认不是所请求的文献
    PARSE_FAILED = "parse_failed"                # PDF 解析/切分失败
    CITATION_UNMATCHED = "citation_unmatched"    # 引用键未能对齐到参考文献
    QUOTE_NOT_LOCATED = "quote_not_located"      # 模型给出的证据引文无法在原文定位
    LLM_CALL_FAILED = "llm_call_failed"          # 大模型调用失败
    LLM_PARSE_FAILED = "llm_parse_failed"        # 大模型输出无法解析


class ClaimIntent(str, Enum):
    """阶段 1：单条 (claim_id, ref_key) 原子主张的引用意图分类"""
    BACKGROUND = "background"               # 领域背景泛引（如 "RNN [13] and GRU [7] are widely used"）
    METHOD_USAGE = "method_usage"           # 方法/组件/数据集采用（如 "We use Adam [20] / residual connections [11]"）
    FACTUAL_CLAIM = "factual_claim"         # 关于被引文献的具体事实/性质/结论断言（如 "ByteNet grows logarithmically [18]"）
    METRIC_COMPARISON = "metric_comparison" # 精确实验数值/指标/复杂度对比（如 "GNMT [38] BLEU 24.6 / 39.92"）
    UNVERIFIABLE = "unverifiable"           # 无法核查或纯主论文自身陈述、与被引文献内容无关


class EffortRoute(str, Enum):
    """阶段 1：Orchestrator (Effort Allocator) 算力分配四级路由"""
    ROUTE_SKIP = "route_skip"               # 跳过正文检索（unverifiable，0 次检索/判决 LLM 调用）
    ROUTE_FAST_BG = "route_fast_bg"         # 极速背景核查（仅读 Abstract + Introduction 先验，fast 模型一轮判完）
    ROUTE_STANDARD = "route_standard"       # 标准证据核查（先验 ∪ BM25 ∪ 章节地图排序，default 模型精判）
    ROUTE_DEEP_METRIC = "route_deep_metric" # 深度数值核查（优先召回含数字/表格块 + 精确数值锚点比对，reasoning 模型精判）

