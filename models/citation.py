from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, ConfigDict, Field


class CitationStatus(str, Enum):
    UNRESOLVED = "unresolved"      # 尚未通过外部工具解析
    RESOLVED = "resolved"          # 已获取元数据（标题、作者、年份）
    FETCHED = "fetched"            # 已成功下载并解析全文/摘要
    NOT_FOUND = "not_found"        # 学术搜索引擎未找到该文献
    PAYWALL_LOCKED = "paywall"     # 存在付费墙，仅获取到摘要


class CitationRef(BaseModel):
    """
    表示论文文末参考文献表中的单条引用元数据
    """
    key: str = Field(
        ...,
        description="正文中引用的唯一标识符，如 '[1]', '[Vaswani2017]' 或 '(Devlin et al., 2018)'"
    )
    raw_text: str = Field(
        ...,
        description="文末参考文献列表中的原始文本条目"
    )
    title: Optional[str] = Field(
        default=None,
        description="解析或对齐后的论文标题"
    )
    authors: List[str] = Field(
        default_factory=list,
        description="作者列表"
    )
    year: Optional[int] = Field(
        default=None,
        description="发表年份"
    )
    doi: Optional[str] = Field(
        default=None,
        description="文献数字对象唯一标识符 (DOI)"
    )
    arxiv_id: Optional[str] = Field(
        default=None,
        description="arXiv 编号（若有）"
    )
    pdf_url: Optional[str] = Field(
        default=None,
        description="文献全文 PDF 下载链接或本地缓存路径"
    )
    pdf_candidates: List[str] = Field(
        default_factory=list,
        description="候选 PDF 下载直链列表"
    )
    status: CitationStatus = Field(
        default=CitationStatus.UNRESOLVED,
        description="文献检索与下载状态"
    )
    last_failure: Optional[str] = Field(
        default=None,
        description="最近一次获取失败的结构化原因 (FailureReason 值)，成功获取后为 None"
    )

    model_config = ConfigDict(frozen=False)
