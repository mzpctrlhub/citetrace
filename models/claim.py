from typing import List
from pydantic import BaseModel, Field

from .verification import ClaimIntent


class SourceSpan(BaseModel):
    """主张在原文段落中的字符区间。text 必须逐字来自 paragraph。"""
    start: int = Field(ge=0, description="起始下标（含）")
    end: int = Field(ge=0, description="结束下标（不含）")
    text: str = Field(description="paragraph[start:end] 的逐字内容")


class AtomicClaimDraft(BaseModel):
    """LLM 单次输出的原始结构（不含任何由程序盖章的字段）"""
    source_spans: List[SourceSpan]
    verbatim_claim: str
    normalized_claim: str
    citation_keys: List[str]
    relationship: ClaimIntent
    requires_review: bool = False
    review_reason: str = ""


class AtomicClaimResponse(BaseModel):
    """LLM 完整响应"""
    claims: List[AtomicClaimDraft]
    unassigned_citation_keys: List[str] = Field(default_factory=list)


class AtomicClaim(AtomicClaimDraft):
    """落盘结构：在 Draft 之上补程序盖章字段，并允许 validator 就地修正 span 下标"""
    claim_id: str                 # 形如 "claim_001.a01"，由 extractor 生成
    parent_claim_id: str          # 来自 extracted_claims.json 的 claim_id
    context_id: str               # 透传
