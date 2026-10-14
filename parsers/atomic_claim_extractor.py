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
        max_attempts: int = 3,
    ):
        self.llm = llm
        self.model_name = model_name
        self.validator = validator
        self.max_tokens = max_tokens
        self.max_attempts = max(1, max_attempts)

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

    def _get_last_raw_response(self) -> str:
        return getattr(self.llm, "last_raw_response", "") or ""

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

        payload = self._payload(context)
        current_prompt = self.prompt_text
        round_errors_history: List[str] = []
        response: Optional[AtomicClaimResponse] = None
        val_res: Optional[ValidationResult] = None
        raw_resp = ""
        attempts = 0

        for attempt in range(1, self.max_attempts + 1):
            attempts = attempt
            try:
                response = self.llm.generate_json(
                    prompt=current_prompt,
                    payload=payload,
                    schema=AtomicClaimResponse,
                    max_tokens=self.max_tokens,
                )
                raw_resp = self._get_last_raw_response()
                val_res = self.validator.validate_and_repair(paragraph, citation_keys, response)
            except Exception as e:
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

            if val_res.valid:
                break

            round_err_str = f"第{attempt}轮: " + "; ".join(val_res.errors)
            round_errors_history.append(round_err_str)

            if attempt < self.max_attempts:
                logger.warning(
                    f"[{parent_claim_id}] 第 {attempt} 轮校验未通过，触发语义级重试: {val_res.errors}"
                )
                current_prompt = (
                    self.prompt_text
                    + "\n\n--- Correction required ---\nYour previous output was rejected by the checker for these reasons:\n"
                    + "\n".join(f"- {e}" for e in val_res.errors)
                    + "\nRe-read the paragraph and return the full corrected JSON object."
                )

        # 所有尝试耗尽后仍未通过校验
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

        # 校验成功：程序盖章生成 AtomicClaim
        stamped_claims: List[AtomicClaim] = []
        for idx, draft in enumerate(response.claims, start=1):
            generated_verbatim = " ... ".join(s.text for s in draft.source_spans)
            claim_obj = AtomicClaim(
                **draft.model_dump(),
                verbatim_claim=generated_verbatim,
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
