# ==============================================================================
# 模块名称: models/claim.py
# 模块职责: 阶段 1 原子主张抽取（Atomic Claim Extraction）的核心数据契约定义
# 核心技术: 基于 Pydantic v2 构建严格的数据模式与自动校验边界
# ==============================================================================

from typing import List                              # 导入类型注解 List，用于声明泛型列表
from pydantic import BaseModel, Field                # BaseModel: 具有强类型校验的数据模型基类; Field: 字段约束配置器

from .verification import ClaimIntent                # 从同级模块导入引用意图枚举类（ClaimIntent）


class SourceSpan(BaseModel):
    """
    【物理字符定位碎片】
    表示某条原子主张在原始段落（paragraph）中所截取的连续文本片段及其绝对字符坐标。
    约束核心：paragraph[start:end] 必须与 text 100% 逐字对应，不允许有任何偏差。
    """
    # 起始下标（包含该位置），ge=0 限制最小为 0；容许大模型吐占位符 0
    start: int = Field(ge=0, description="起始字符下标（含，基于0开始）")

    # 结束下标（不含该位置），ge=0 允许大模型初始输出 0 占位符，由后续 Validator 就地修复
    end: int = Field(ge=0, description="结束字符下标（不含，基于0开始）")

    # 从原始段落中一字不差复制出来的真实子串内容
    text: str = Field(description="paragraph[start:end] 的逐字切片内容")


class AtomicClaimDraft(BaseModel):
    """
    【LLM 原始作业草稿】
    这是大模型单次执行抽取后直接吐出的数据结构。
    此时尚未经过系统程序盖章，只包含大模型对段落自然语言理解的业务字段。
    去除了 verbatim_claim 冗余字段（由 Python 在落盘时根据 source_spans 确定性合成）。
    严格去除所有默认值，确保转换为 JSON Schema 时所有字段均在 required 中，100% 兼容 Strict Mode (CFG)。
    """
    # 支撑此主张的原文片段列表（支持多段拼接，例如主谓在句首、宾语在句尾）
    source_spans: List[SourceSpan] = Field(description="支撑该主张的原文物理片段列表")

    # 独立主张：对原句进行规范化消除代词（如把 They 改为 Transformer），使其脱离上下文依然语义自洽
    normalized_claim: str = Field(description="消除代词、自包含的独立断言陈述句")

    # 此原子主张归属的文献引用键列表，如 ['[13]'] 或 ['[2]', '[5]']
    citation_keys: List[str] = Field(description="归属于该主张的具体引用文献标记列表")

    # 引用意图分类枚举（background, method_usage, factual_claim, metric_comparison, unverifiable）
    relationship: ClaimIntent = Field(description="作者引用该文献的学术意图分类")

    # 大模型举手求助开关：强制必须显式输出 true 或 false，无默认值
    requires_review: bool = Field(description="是否需要人工复核介入标记（必须显式输出 true 或 false）")

    # 举手求助的具体原因说明；无需复核时强制显式输出 ""，无默认值
    review_reason: str = Field(description="需要人工复核的具体原因说明（无需复核时必须输出空字符串 \"\"）")


class AtomicClaimResponse(BaseModel):
    """
    【LLM 顶级输出响应包】
    大模型返回的最外层完整 JSON 对象。
    除了抽出的主张列表，还强制要求交待未分配的引用，实施引用守恒定律。
    严格去除默认值，确保 100% 满足 Strict Mode 语法约束。
    """
    # 成功从当前段落中提取出的原子主张草稿列表
    claims: List[AtomicClaimDraft] = Field(description="当前段落提取出的原子主张列表")

    # 未能归属于任何主张的孤儿引用键列表；若全部已分配则必须显式输出 []
    unassigned_citation_keys: List[str] = Field(
        description="段落中出现但未分配给任何主张的引用标记列表（若全部已分配则必须显式输出空列表 []）"
    )


class AtomicClaim(AtomicClaimDraft):
    """
    【最终落盘实体模型】
    继承自 AtomicClaimDraft，拥有其全部业务属性。
    由系统的 Python 编排器为其盖上全局唯一的身份证号（claim_id），并绑定父级段落上下文。
    """
    # 系统赋予的全局原子主张 ID，格式规范为 'claim_001.a01'（由 Extractor 严格自增生成）
    claim_id: str = Field(description="全局唯一的原子主张ID（形如 claim_001.a01）")

    # 溯源上游：来自 extracted_claims.json 的原始段落 claim_id（形如 claim_001）
    parent_claim_id: str = Field(description="所属父段落的原始 claim_id")

    # 透传段落的上下文环境唯一标识，用于跨阶段追溯
    context_id: str = Field(description="所属段落的上下文 context_id")

    # 由 Python 代码根据 source_spans 自动确切拼接生成的原文原句（无需大模型耗费 Token 重复生成）
    verbatim_claim: str = Field(description="由 Python 自动从 source_spans 拼接而成的原文完整原句")
