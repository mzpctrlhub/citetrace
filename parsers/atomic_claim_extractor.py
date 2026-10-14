from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from llm_config import render_prompt
from models.claim import AtomicClaim, AtomicClaimResponse
from parsers.claim_validator import ClaimValidator, ValidationResult

logger = logging.getLogger("citetrace.extractor")


@dataclass
class ExtractionOutcome:
    parent_claim_id: str
    context_id: str
    status: str                       # "ok" | "validation_failed" | "llm_failed" | "skipped_no_citation"
    claims: List[AtomicClaim] = field(default_factory=list)
    unassigned_citation_keys: List[str] = field(default_factory=list)
    validation: Optional[ValidationResult] = None
    error: str = ""
    raw_response: str = ""            # 仅失败时保留，便于人工排查
    attempts: int = 1


class AtomicClaimExtractor:
    def __init__(
        self,
        llm,
        model_name: str,
        validator: ClaimValidator,
        prompt_path: Optional[Path] = None,
        max_tokens: int = 4000,
    ):
        self.llm = llm
        self.model_name = model_name
        self.validator = validator
        self.max_tokens = max_tokens

        if prompt_path is None:
            prompt_path = Path(__file__).resolve().parents[1] / "prompts" / "atomic_claim_v1.txt"
        self.prompt_path = prompt_path
        self.prompt_text = self.prompt_path.read_text(encoding="utf-8")

    def _payload(self, context: dict) -> dict:
        citation_keys = context.get("citation_keys", [])
        return {
            "paragraph": context.get("paragraph", ""),
            "citation_keys": ", ".join(citation_keys),
        }

    def preview(self, context: dict) -> str:
        """返回该 context 拼装完成、占位符已全部替换的最终 prompt。"""
        return render_prompt(self.prompt_text, self._payload(context))

    def extract(self, context: dict) -> ExtractionOutcome:
        """context 是 extracted_claims.json 里的一条 claim 记录（原样传入）"""
        parent_claim_id = context.get("claim_id", "")
        context_id = context.get("context_id", "")
        citation_keys = context.get("citation_keys", [])
        paragraph = context.get("paragraph", "")

        # 规则 1: citation_keys 为空直接跳过
        if not citation_keys:
            return ExtractionOutcome(
                parent_claim_id=parent_claim_id,
                context_id=context_id,
                status="skipped_no_citation",
                error="段落无引用标记，已跳过",
            )

        # 首次调用
        payload = self._payload(context)
        attempts = 1
        raw_resp = ""
        response: Optional[AtomicClaimResponse] = None
        val_res: Optional[ValidationResult] = None

        try:
            response = self.llm.generate_json(
                prompt=self.prompt_text,
                payload=payload,
                schema=AtomicClaimResponse,
                max_tokens=self.max_tokens,
            )
            val_res = self.validator.validate_and_repair(paragraph, citation_keys, response)
        except Exception as e:
            logger.error(f"[{parent_claim_id}] LLM 调用/解析异常 (尝试 1): {e}")
            return ExtractionOutcome(
                parent_claim_id=parent_claim_id,
                context_id=context_id,
                status="llm_failed",
                error=str(e),
                attempts=1,
            )

        # 若校验失败，启动语义级重试（第 2 次尝试）
        if not val_res.valid:
            attempts = 2
            logger.warning(f"[{parent_claim_id}] 校验未通过，触发语义级重试: {val_res.errors}")
            first_errors = list(val_res.errors)

            prompt_2 = (
                self.prompt_text
                + "\n\n--- Correction required ---\nYour previous output was rejected by the checker for these reasons:\n"
                + "\n".join(f"- {e}" for e in first_errors)
                + "\nRe-read the paragraph and return the full corrected JSON object."
            )

            try:
                response = self.llm.generate_json(
                    prompt=prompt_2,
                    payload=payload,
                    schema=AtomicClaimResponse,
                    max_tokens=self.max_tokens,
                )
                val_res = self.validator.validate_and_repair(paragraph, citation_keys, response)
            except Exception as e:
                logger.error(f"[{parent_claim_id}] 重试轮次 LLM 异常: {e}")
                return ExtractionOutcome(
                    parent_claim_id=parent_claim_id,
                    context_id=context_id,
                    status="llm_failed",
                    error=f"重试时异常: {e}; 首轮错误: {first_errors}",
                    attempts=2,
                )

        # 二次校验仍未通过
        if not val_res.valid:
            combined_errs = "; ".join(val_res.errors)
            return ExtractionOutcome(
                parent_claim_id=parent_claim_id,
                context_id=context_id,
                status="validation_failed",
                validation=val_res,
                error=combined_errs,
                raw_response=raw_resp,
                attempts=attempts,
            )

        # 校验成功：程序盖章生成 AtomicClaim
        stamped_claims: List[AtomicClaim] = []
        for idx, draft in enumerate(response.claims, start=1):
            claim_obj = AtomicClaim(
                **draft.model_dump(),
                claim_id=f"{parent_claim_id}.a{idx:02d}",
                parent_claim_id=parent_claim_id,
                context_id=context_id,
            )
            stamped_claims.append(claim_obj)

        return ExtractionOutcome(
            parent_claim_id=parent_claim_id,
            context_id=context_id,
            status="ok",
            claims=stamped_claims,
            unassigned_citation_keys=response.unassigned_citation_keys,
            validation=val_res,
            attempts=attempts,
        )
