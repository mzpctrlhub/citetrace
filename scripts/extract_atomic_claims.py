# scripts/extract_atomic_claims.py
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# 确保项目根目录在 sys.path 中
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_config import get_llm_client
from parsers.atomic_claim_extractor import AtomicClaimExtractor, ExtractionOutcome
from parsers.claim_validator import ClaimValidator

logger = logging.getLogger("citetrace.script")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CiteTrace 阶段 1：原子主张提取器与契约强校验")
    parser.add_argument("--paper-folder", type=str, required=True, help="论文目录路径（含 extracted_claims.json）")
    parser.add_argument("--limit", type=int, default=10, help="处理的段落数量，默认 10（传 0 表示处理全部剩余段落）")
    parser.add_argument("--start", type=int, default=0, help="起始段落索引，默认 0")
    parser.add_argument("--role", type=str, default="fast", help="大模型角色，默认 fast")
    parser.add_argument("--max-tokens", type=int, default=4000, help="大模型最大输出 token，默认 4000")
    parser.add_argument("--sleep", type=float, default=1.0, help="LLM 调用间隔秒数，默认 1.0")
    parser.add_argument("--out", type=str, default=None, help="atomic_claims.json 输出路径")
    parser.add_argument("--review-out", type=str, default=None, help="atomic_claims_review.md 输出路径")
    parser.add_argument("--resume", action="store_true", help="断点续跑：复用已有输出文件中 status=='ok' 的段落结果")
    parser.add_argument("--dry-run", action="store_true", help="只打印第一条拼装好的完整 Prompt，不调网不写盘")
    return parser.parse_args()


def write_atomic_json(target_path: Path, data: dict) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    tmp_path.replace(target_path)


def write_atomic_markdown(target_path: Path, content: str) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    tmp_path.replace(target_path)


def compute_summary(items: List[dict]) -> dict:
    summary = {
        "parents_processed": len(items),
        "parents_ok": 0,
        "parents_failed": 0,
        "atomic_claims_total": 0,
        "requires_review_total": 0,
        "unassigned_keys_total": 0,
        "repaired_spans_total": 0,
        "errors_total": 0,
        "warnings_total": 0,
    }
    for item in items:
        status = item.get("status", "")
        if status in ("ok", "skipped_no_citation"):
            summary["parents_ok"] += 1
        else:
            summary["parents_failed"] += 1
        claims = item.get("claims", [])
        val = item.get("validation", {}) or {}
        summary["atomic_claims_total"] += len(claims)
        summary["requires_review_total"] += sum(1 for c in claims if c.get("requires_review"))
        summary["unassigned_keys_total"] += len(item.get("unassigned_citation_keys", []))
        summary["repaired_spans_total"] += int(val.get("repaired_spans", 0))
        summary["errors_total"] += len(val.get("errors", []))
        summary["warnings_total"] += len(val.get("warnings", []))
    return summary


def build_review_markdown(
    paper_id: str,
    paper_title: str,
    model_name: str,
    summary: dict,
    items: List[dict],
) -> str:
    lines = []
    lines.append("# CiteTrace 原子主张提取人工审查报告 (Atomic Claims Review)")
    lines.append("")
    lines.append(f"- **论文**: {paper_title} (`{paper_id}`)")
    lines.append(f"- **模型**: `{model_name}`")
    lines.append(f"- **生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("## 总体执行摘要 (Summary)")
    lines.append("| 处理段落数 | 成功 (OK) | 失败 (Failed) | 原子主张总数 | 待复核数 | 修补下标数 | 硬错误数 | 软告警数 |")
    lines.append("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")
    lines.append(
        f"| {summary['parents_processed']} | {summary['parents_ok']} | {summary['parents_failed']} | "
        f"{summary['atomic_claims_total']} | {summary['requires_review_total']} | "
        f"{summary['repaired_spans_total']} | {summary['errors_total']} | {summary['warnings_total']} |"
    )
    lines.append("")
    lines.append("---")
    lines.append("")

    for item in items:
        pid = item["parent_claim_id"]
        sec = item.get("source_section", "")
        status = item["status"]
        status_icon = "✓" if status in ("ok", "skipped_no_citation") else "✗"
        lines.append(f"## [{status_icon}] {pid} ({sec}) —— 状态: `{status}`")
        lines.append("")
        lines.append("**原段落**:")
        lines.append(f"> {item['paragraph']}")
        lines.append("")
        lines.append(f"**期望引用**: {', '.join(item['expected_citation_keys'])}")
        lines.append("")

        claims = item.get("claims", [])
        if claims:
            lines.append(f"### 提取出的原子主张列表 ({len(claims)} 条)")
            lines.append("| claim_id | relationship | citation_keys | normalized_claim | review |")
            lines.append("| :--- | :--- | :--- | :--- | :--- |")
            for c in claims:
                cid = c["claim_id"]
                rel = c["relationship"]
                keys = ", ".join(c["citation_keys"])
                norm = str(c["normalized_claim"]).replace("|", "\\|")
                rev = f"⚠️ {c['review_reason']}" if c["requires_review"] else "✓"
                lines.append(f"| {cid} | `{rel}` | {keys} | {norm} | {rev} |")
            lines.append("")

        val = item.get("validation", {})
        errs = val.get("errors", [])
        warns = val.get("warnings", [])
        if errs:
            lines.append("**硬错误清单 (Errors)**:")
            for e in errs:
                lines.append(f"- ❌ {e}")
            lines.append("")
        if warns:
            lines.append("**校验软告警 (Warnings)**:")
            for w in warns:
                lines.append(f"- ⚠️ {w}")
            lines.append("")

        unassigned = item.get("unassigned_citation_keys", [])
        if unassigned:
            lines.append(f"**未归属引用**: `{', '.join(unassigned)}`\n")

        if item.get("raw_response"):
            lines.append(f"**原始输出 (Raw Output)**:\n```\n{item['raw_response']}\n```\n")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    paper_dir = Path(args.paper_folder).resolve()
    input_claims_file = paper_dir / "extracted_claims.json"

    if not input_claims_file.is_file():
        print(f"[!] 错误: 输入文件不存在: {input_claims_file}", file=sys.stderr)
        return 1

    with open(input_claims_file, "r", encoding="utf-8") as f:
        root_data = json.load(f)

    all_claims = root_data.get("claims", [])
    paper_id = root_data.get("paper_id", paper_dir.name)
    paper_title = root_data.get("paper_title", paper_id)

    effective_limit = args.limit if args.limit > 0 else len(all_claims)
    target_claims = all_claims[args.start : args.start + effective_limit]
    if not target_claims:
        print(f"[!] 警告: 指定范围 [{args.start}:{args.start+effective_limit}] 内无可用段落。")
        return 0

    validator = ClaimValidator()

    # --dry-run 逻辑
    if args.dry_run:
        extractor = AtomicClaimExtractor(
            llm=None,
            model_name="dry-run",
            validator=validator,
            max_tokens=args.max_tokens,
        )
        sample = target_claims[0]
        preview_prompt = extractor.preview(sample)
        print("=" * 80)
        print(f">> [DRY-RUN] 预览首个段落 ({sample.get('claim_id')}) 渲染后的完整 Prompt:")
        print("=" * 80)
        print(preview_prompt)
        print("=" * 80)
        return 0

    # 实际运行逻辑
    out_json_path = Path(args.out) if args.out else paper_dir / "atomic_claims.json"
    out_review_path = Path(args.review_out) if args.review_out else paper_dir / "atomic_claims_review.md"

    cached_ok_items: Dict[str, dict] = {}
    if args.resume and out_json_path.is_file():
        try:
            existing_data = json.loads(out_json_path.read_text(encoding="utf-8"))
            for it in existing_data.get("items", []):
                if it.get("status") in ("ok", "skipped_no_citation") and it.get("parent_claim_id"):
                    cached_ok_items[it["parent_claim_id"]] = it
        except Exception as e:
            logger.warning(f"读取已有断点文件失败，将从头运行: {e}")

    client, model_name = get_llm_client(role=args.role)
    extractor = AtomicClaimExtractor(
        llm=client,
        model_name=model_name,
        validator=validator,
        max_tokens=args.max_tokens,
    )

    print("=" * 80)
    print(">> 启动 CiteTrace 阶段 1：原子主张提取与契约强校验")
    print(f">> 论文: {paper_title} ({paper_id})")
    print(f">> 模型: {model_name} (Role: {args.role})")
    print(f">> 处理区间: claims[{args.start} : {args.start + len(target_claims)}] (共 {len(target_claims)} 条段落)")
    if cached_ok_items:
        print(f">> 断点续跑开启: 已加载 {len(cached_ok_items)} 条历史成功段落")
    print("=" * 80)

    items: List[dict] = []
    table_rows = []

    def _persist_progress() -> dict:
        cur_summary = compute_summary(items)
        full_output = {
            "paper_id": paper_id,
            "paper_title": paper_title,
            "source_file": str(input_claims_file),
            "extractor_version": "atomic_claim_v1",
            "model": model_name,
            "role": args.role,
            "generated_at": datetime.now().isoformat(),
            "range": {"start": args.start, "limit": len(target_claims)},
            "summary": cur_summary,
            "items": items,
        }
        write_atomic_json(out_json_path, full_output)
        review_md_text = build_review_markdown(paper_id, paper_title, model_name, cur_summary, items)
        write_atomic_markdown(out_review_path, review_md_text)
        return cur_summary

    for i, claim_ctx in enumerate(target_claims, start=1):
        pid = claim_ctx.get("claim_id", f"claim_{i}")
        citation_keys = claim_ctx.get("citation_keys", [])

        if pid in cached_ok_items:
            item_entry = cached_ok_items[pid]
            items.append(item_entry)
            val_dict = item_entry.get("validation", {}) or {}
            claims_dump = item_entry.get("claims", [])
            req_review_count = sum(1 for c in claims_dump if c.get("requires_review"))
            print(
                f"[{i}/{len(target_claims)}] 复用缓存 {pid} (涉及 {len(citation_keys)} 个引用) "
                f"-> [{item_entry['status']}] {len(claims_dump)} 条原子主张"
            )
            table_rows.append({
                "pid": pid,
                "status": item_entry["status"],
                "claims_count": len(claims_dump),
                "keys_count": len(citation_keys),
                "errs": len(val_dict.get("errors", [])),
                "warns": len(val_dict.get("warnings", [])),
                "reviews": req_review_count,
            })
            continue

        print(
            f"[{i}/{len(target_claims)}] 正在处理 {pid} (涉及 {len(citation_keys)} 个引用: {', '.join(citation_keys)})...",
            end="",
            flush=True,
        )

        outcome: ExtractionOutcome = extractor.extract(claim_ctx)

        is_ok = outcome.status in ("ok", "skipped_no_citation")
        if is_ok:
            print(f" -> [{outcome.status}] 提取出 {len(outcome.claims)} 条原子主张 (尝试 {outcome.attempts} 次)")
        else:
            print(f" -> [FAILED: {outcome.status}] {outcome.error}")

        val_dict = {
            "valid": outcome.validation.valid if outcome.validation else False,
            "errors": outcome.validation.errors if outcome.validation else ([outcome.error] if outcome.error else []),
            "warnings": outcome.validation.warnings if outcome.validation else [],
            "repaired_spans": outcome.validation.repaired_spans if outcome.validation else 0,
        }

        claims_dump = [c.model_dump() for c in outcome.claims]
        req_review_count = sum(1 for c in outcome.claims if c.requires_review)

        item_entry = {
            "parent_claim_id": pid,
            "context_id": claim_ctx.get("context_id", ""),
            "source_section": claim_ctx.get("source_section", ""),
            "content_type": claim_ctx.get("content_type", "prose"),
            "paragraph": claim_ctx.get("paragraph", ""),
            "expected_citation_keys": citation_keys,
            "matched_references": claim_ctx.get("matched_references", {}),
            "status": outcome.status,
            "attempts": outcome.attempts,
            "validation": val_dict,
            "unassigned_citation_keys": outcome.unassigned_citation_keys,
            "claims": claims_dump,
            "error": outcome.error,
            "raw_response": outcome.raw_response,
        }
        items.append(item_entry)

        table_rows.append({
            "pid": pid,
            "status": outcome.status,
            "claims_count": len(outcome.claims),
            "keys_count": len(citation_keys),
            "errs": len(val_dict["errors"]),
            "warns": len(val_dict["warnings"]),
            "reviews": req_review_count,
        })

        # 每处理完一段立即增量落盘
        _persist_progress()

        if i < len(target_claims) and args.sleep > 0:
            time.sleep(args.sleep)

    summary = _persist_progress()
    print(f"\n[✓] 产物已保存至 JSON: {out_json_path}")
    print(f"[✓] 产物已保存至 Markdown: {out_review_path}")

    # 控制台打印摘要表格
    print("\n" + "=" * 80)
    print(">> 批处理执行摘要总表 (Execution Summary Table):")
    print("=" * 80)
    print("| Parent Claim ID | Status | 主张数 | 涉及引用数 | Errors | Warnings | 待复核数 |")
    print("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
    for r in table_rows:
        print(
            f"| {r['pid']:<15} | {r['status']:<6} | {r['claims_count']:<6} | "
            f"{r['keys_count']:<10} | {r['errs']:<6} | {r['warns']:<8} | {r['reviews']:<8} |"
        )
    print(
        f"| {'TOTAL':<15} | -      | {summary['atomic_claims_total']:<6} | -          | "
        f"{summary['errors_total']:<6} | {summary['warnings_total']:<8} | {summary['requires_review_total']:<8} |"
    )
    print("=" * 80)

    exit_code = 0 if summary["parents_failed"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
