# ==============================================================================
# 模块名称: parsers/atomic_claim_extractor.py
# 模块职责: 阶段 1 原子主张抽取全流程高层编排器（Orchestrator）
# 核心功能: 串联 Prompt、LLMClient、Pydantic 与 ClaimValidator，实现两阶段自愈反思闭环
# ==============================================================================

from __future__ import annotations                     # 启用类型注解延迟求值

import logging                                         # 导入日志模块，记录抽取全生命周期追踪信息
from dataclasses import dataclass, field               # 导入数据类装饰器与字段生成器
from pathlib import Path                               # 导入面向对象的跨平台路径操作库
from typing import List, Optional                      # 导入泛型列表与可选类型注解

from llm_config import render_prompt                   # 导入模板引擎：安全替换 Prompt 中的 {{PARAGRAPH}} 等变量
from models.claim import AtomicClaim, AtomicClaimResponse  # 导入落盘实体模型与顶级响应模型
from parsers.claim_validator import ClaimValidator, ValidationResult  # 导入离线校验器与校验结果报告

# 获取专属抽取器命名空间的日志记录器
logger = logging.getLogger("citetrace.extractor")


@dataclass
class ExtractionOutcome:
    """
    【段落级抽取最终产出报告】
    封装单条原始段落经过编排器处理后的最终归宿（包含抽取主张、校验状态、重试次数等）。
    """
    parent_claim_id: str                               # 父段落的主张 ID（如 'claim_001'）
    context_id: str                                    # 父段落的上下文追踪 ID（如 'ctx_001'）
    status: str                                        # 最终执行状态: "ok" | "validation_failed" | "llm_failed" | "skipped_no_citation"
    claims: List[AtomicClaim] = field(default_factory=list)  # 成功抽取并经程序盖章的原子主张列表
    unassigned_citation_keys: List[str] = field(default_factory=list)  # 合法未被任何主张绑定的孤儿引用列表
    validation: Optional[ValidationResult] = None      # 最后一轮的确定性校验结果详细报告
    error: str = ""                                    # 失败时的聚合错误描述信息
    raw_response: str = ""                             # 仅在模型解析失败时保留原始字符串，专供开发者人工排错
    attempts: int = 1                                  # 实际经历的尝试/反思轮数（1 到 max_attempts）


class AtomicClaimExtractor:
    """
    原子主张提取器（总调度核心）
    负责管理 Prompt 模板装配、调用大模型、与 Validator 校验器进行闭环博弈并实施自愈。
    """

    def __init__(
        self,
        llm,                                           # LLM 客户端实例（支持 generate_json 接口）
        model_name: str,                               # 当前使用的大模型名称标识
        validator: ClaimValidator,                     # 确定性校验器实例
        prompt_path: Optional[Path] = None,            # 提示词文件路径（可选，默认定位 prompts/atomic_claim_v1.txt）
        max_tokens: int = 4000,                        # 大模型最大输出 Token 上限
        max_attempts: int = 3,                         # 最大反思重试尝试次数（默认最多尝试 3 轮）
    ):
        self.llm = llm                                 # 绑定底层大模型客户端
        self.model_name = model_name                   # 记录模型名字
        self.validator = validator                     # 绑定校验法官
        self.max_tokens = max_tokens                   # 单次最大生成长度
        self.max_attempts = max(1, max_attempts)       # 保底至少尝试 1 次

        # 若未显式传入 prompt 路径，则自动根据相对路径寻址 prompts/atomic_claim_v1.txt
        if prompt_path is None:
            prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "atomic_claim_v1.txt"
        self.prompt_path = prompt_path                 # 保存路径
        self.prompt_text = self.prompt_path.read_text(encoding="utf-8")  # 一次性读取提示词模版至内存缓存

    def _payload(self, context: dict) -> dict:
        """从上游段落 context 字典中提取段落文本与引用列表，组装成模板替换字典"""
        citation_keys = context.get("citation_keys", [])
        return {
            "paragraph": context.get("paragraph", ""),
            "citation_keys": ", ".join(citation_keys),
        }

    def preview(self, context: dict) -> str:
        """【调试专用接口】返回占位符全部渲染完毕后的完整 Prompt，不调网不花钱"""
        return render_prompt(self.prompt_text, self._payload(context))

    def _get_last_raw_response(self) -> str:
        """获取底层大模型最近一次吐出的未经清洗的原始文本（用于记录失败现场）"""
        return getattr(self.llm, "last_raw_response", "") or ""

    def extract(self, context: dict) -> ExtractionOutcome:
        """
        【主执行管道】：对单条段落执行抽取、强校验与反思闭环
        :param context: 来自 extracted_claims.json 的原始段落字典
        :return: 包含抽取状态与盖章后主张的 ExtractionOutcome 对象
        """
        parent_claim_id = context.get("claim_id", "")  # 获取父段落 ID，如 'claim_001'
        context_id = context.get("context_id", "")      # 获取上下文追踪 ID，如 'ctx_001'
        citation_keys = context.get("citation_keys", [])# 本段落中出现的引用列表
        paragraph = context.get("paragraph", "")        # 段落原始正文

        # ==============================================================================
        # 【前置零成本过滤】：若本段落根本没有任何引用标记，直接秒级跳过！
        # 绝不调用大模型，既省下昂贵的 Token，又防止模型对着纯陈述句瞎编引用。
        # ==============================================================================
        if not citation_keys:
            return ExtractionOutcome(
                parent_claim_id=parent_claim_id,
                context_id=context_id,
                status="skipped_no_citation",
                error="段落无引用标记，已跳过",
            )

        # 准备模板注入插槽数据
        payload = self._payload(context)
        current_prompt = self.prompt_text              # 初始提示词模版
        round_errors_history: List[str] = []           # 记录各轮失败原因的历史档案
        response: Optional[AtomicClaimResponse] = None # 模型响应对象容器
        val_res: Optional[ValidationResult] = None     # 校验结果容器
        raw_resp = ""                                  # 原始字符串容器
        attempts = 0                                   # 记录实际尝试轮次

        # ==============================================================================
        # 【两阶段 / 多轮反思纠错闭环循环】
        # 允许大模型在最多 max_attempts 轮内不断根据校验器的判词自我纠错
        # ==============================================================================
        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                # 1. 调用大模型：执行 Prompt 渲染、网络请求、防爆清洗与 Pydantic 反序列化
                response = self.llm.generate_json(
                    prompt=current_prompt,
                    payload=payload,
                    schema=AtomicClaimResponse,
                    max_tokens=self.max_tokens,
                )
                raw_resp = self._get_last_raw_response()  # 记录本轮原始字符串
                
                # 2. 呈交校验法官：执行 10 项严苛确定性校验与坐标就地修补
                val_res = self.validator.validate_and_repair(paragraph, citation_keys, response)
            except Exception as e:
                # 捕获网络异常、超时或无法解析的非 JSON 格式翻车
                raw_resp = self._get_last_raw_response()
                logger.error(f"[{parent_claim_id}] LLM 调用/解析异常 (尝试 {attempt}): {e}")
                prior_info = f"; 前序校验错误: {' | '.join(round_errors_history)}" if round_errors_history else ""
                return ExtractionOutcome(
                    parent_claim_id=parent_claim_id,
                    context_id=context_id,
                    status="llm_failed",
                    error=f"第 {attempt} 次尝试异常: {e}{prior_info}",
                    raw_response=raw_resp,
                    attempts=attempt,
                )

            # 3. 校验通过！直接跳出重试循环，无需进行下一轮
            if val_res.valid:
                break

            # 4. 校验失败：记录本轮错误历史
            round_err_str = f"第{attempt}轮: " + "; ".join(val_res.errors)
            round_errors_history.append(round_err_str)

            # 5. 如果还有重试机会，组装“纠错判词（Correction Directive）”注入下一轮 Prompt
            if attempt < self.max_attempts:
                logger.warning(
                    f"[{parent_claim_id}] 第 {attempt} 轮校验未通过，触发语义级重试: {val_res.errors}"
                )
                # 动态拼接 Correction 指令，明确告知大模型上一轮错在哪里，逼它针对性反思
                current_prompt = (
                    self.prompt_text
                    + "\n\n--- Correction required ---\nYour previous output was rejected by the checker for these reasons:\n"
                    + "\n".join(f"- {e}" for e in val_res.errors)
                    + "\nRe-read the paragraph and return the full corrected JSON object."
                )

        # ==============================================================================
        # 【收尾判定 1：所有尝试耗尽后依然未能通过校验】
        # 绝不让脏数据蒙混过关，老老实实标记为 validation_failed 并封存错误证据供人工排查
        # ==============================================================================
        if val_res is None or not val_res.valid:
            combined_errs = " | ".join(round_errors_history) if round_errors_history else "未知校验错误"
            return ExtractionOutcome(
                parent_claim_id=parent_claim_id,
                context_id=context_id,
                status="validation_failed",
                validation=val_res,
                error=combined_errs,
                raw_response=raw_resp,
                attempts=attempts,
            )

        # ==============================================================================
        # 【收尾判定 2：校验完全合规通过，程序正式盖章入库】
        # 将无状态的大模型草稿 AtomicClaimDraft 升级为正式落盘实体 AtomicClaim
        # ==============================================================================
        stamped_claims: List[AtomicClaim] = []
        for idx, draft in enumerate(response.claims, start=1):
            # 由 Python 代码根据 source_spans 自动拼接完整的原文原句（省去大模型双写 Token）
            generated_verbatim = " ... ".join(s.text for s in draft.source_spans)
            
            # 盖章操作：赋予自增 claim_id（形如 'claim_001.a01'），并锁定父级段落血统
            claim_obj = AtomicClaim(
                **draft.model_dump(),
                verbatim_claim=generated_verbatim,
                claim_id=f"{parent_claim_id}.a{idx:02d}",
                parent_claim_id=parent_claim_id,
                context_id=context_id,
            )
            stamped_claims.append(claim_obj)

        # 返回执行成功的终局成果包
        return ExtractionOutcome(
            parent_claim_id=parent_claim_id,
            context_id=context_id,
            status="ok",
            claims=stamped_claims,
            unassigned_citation_keys=response.unassigned_citation_keys,
            validation=val_res,
            attempts=attempts,
        )
