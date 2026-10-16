# ==============================================================================
# 模块名称: scripts/extract_atomic_claims.py
# 模块职责: 阶段 1 原子主张抽取大规模生产批处理 CLI 入口
# 核心功能: 面向整篇论文段落进行自动化批量抽取、原子安全落盘与可视化复核报告生成
# ==============================================================================

from __future__ import annotations                     # 启用类型注解延迟求值

import argparse                                        # 导入标准命令行参数解析库
import json                                            # 导入 JSON 数据编解码库
import logging                                         # 导入日志库
import sys                                             # 导入系统级交互接口（如退出码 sys.exit）
import time                                            # 导入时间库（用于请求间隙睡眠降频）
from datetime import datetime                          # 导入日期时间模块（用于报告时间戳生成）
from pathlib import Path                               # 导入跨平台面向对象路径操作库
from typing import Any, Dict, List                     # 导入常用泛型类型注解

# 【经典路径防呆】：确保无论在哪个目录下执行此脚本，项目根目录都必定位于 Python 导包搜索路径首位
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from llm_config import get_llm_client                  # 从统一配置中心获取配置好的 LLM 客户端与模型名
from parsers.atomic_claim_extractor import AtomicClaimExtractor, ExtractionOutcome  # 导入核心抽取编排器与结果包
from parsers.claim_validator import ClaimValidator     # 导入 10 项确定性事实校验器

# 获取脚本专属命名空间的日志记录器
logger = logging.getLogger("citetrace.script")


def parse_args() -> argparse.Namespace:
    """
    【CLI 命令行参数配置中心】
    支持通过终端传入各种灵活的批跑控制参数
    """
    parser = argparse.ArgumentParser(description="CiteTrace 阶段 1：原子主张提取器与契约强校验")
    
    # 核心目标路径参数：指向论文根目录（如 cache/papers/Attention_Is_All_You_Need）
    parser.add_argument("--paper-folder", type=str, required=True, help="论文目录路径（含 extracted_claims.json）")
    
    # 规模与批次控制参数
    parser.add_argument("--limit", type=int, default=10, help="处理的段落数量，默认 10（传 0 表示处理全部剩余段落）")
    parser.add_argument("--start", type=int, default=0, help="起始段落索引，默认从第 0 个段落开始")
    
    # 大模型行为控制参数
    parser.add_argument("--role", type=str, default="fast", help="大模型角色类型，默认 fast（对应轻量快速模型）")
    parser.add_argument("--max-tokens", type=int, default=4000, help="大模型最大输出 token，默认 4000")
    parser.add_argument("--sleep", type=float, default=1.0, help="LLM 调用间隔秒数，默认 1.0（防止触发 API 限流）")
    
    # 自定义输出文件重定向
    parser.add_argument("--out", type=str, default=None, help="atomic_claims.json 输出路径（默认保存在论文目录下）")
    parser.add_argument("--review-out", type=str, default=None, help="atomic_claims_review.md 输出路径")
    
    # 高级执行模式开关
    parser.add_argument("--resume", action="store_true", help="断点续跑：自动复用已有输出文件中已成功的段落结果")
    parser.add_argument("--dry-run", action="store_true", help="干跑预览：只打印第一条拼装好的完整 Prompt，不调网不写盘")
    
    return parser.parse_args()


def write_atomic_json(target_path: Path, data: dict) -> None:
    """
    【OS 级原子安全写 JSON】
    为什么绝不能直接 open(target_path, 'w')？
    因为如果写到一半突然按了 Ctrl+C 或断电，原有文件就会彻底损坏变成半截乱码！
    方案：先完整写入同目录下的 .tmp 临时文件，写完后调用操作系统底层的 replace 原子重命名替换。
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)       # 确保父级目录存在
    tmp_path = target_path.with_suffix(".tmp")                  # 生成临时文件路径（如 output.json.tmp）
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)       # 完整序列化写入临时文件
    tmp_path.replace(target_path)                               # 操作系统原子覆盖替换，保证文件永远处于完整状态


def write_atomic_markdown(target_path: Path, content: str) -> None:
    """
    【OS 级原子安全写 Markdown】
    与 JSON 原子写同理，利用临时文件与原子重命名保证报告文件绝对不会半途损坏。
    """
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    tmp_path.replace(target_path)


def compute_summary(items: List[dict]) -> dict:
    """
    【指标实时聚合器】
    遍历所有段落的处理结果，动态统计成功率、主张数、修复数与告警数。
    """
    summary = {
        "parents_processed": len(items),                # 已处理的总段落数
        "parents_ok": 0,                                 # 成功提取并通过校验的段落数
        "parents_failed": 0,                             # 提取或校验失败的段落数
        "atomic_claims_total": 0,                        # 累计提取出的原子主张总条数
        "requires_review_total": 0,                      # 累计标记为 requires_review 的待复核主张数
        "unassigned_keys_total": 0,                      # 累计合法未归属的引用总数
        "repaired_spans_total": 0,                       # 累计被 Python 自动修复坐标的切片数
        "errors_total": 0,                               # 累计触犯的硬错误总数
        "warnings_total": 0,                             # 累计触发的软告警总数
    }
    for item in items:
        status = item.get("status", "")
        if status in ("ok", "skipped_no_citation"):
            summary["parents_ok"] += 1                   # 成功或合法跳过计入成功
        else:
            summary["parents_failed"] += 1               # 校验或网络失败计入失败
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
    """
    【生成人类可读的 Markdown 复核审查报告】
    生成一份结构极其清晰、支持 GitHub 渲染的 atomic_claims_review.md。
    包含总体指标摘要看板、逐段提取表格、原句对比、错误清单与软告警。
    """
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

    # 遍历每个段落，生成卡片式审查块
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
                norm = str(c["normalized_claim"]).replace("|", "\\|")  # 防呆转义 Markdown 表格竖线
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

        # 若发生失败，附带大模型未经清洗的原始输出，方便开发者直接在报告里排查
        if item.get("raw_response"):
            lines.append(f"**原始输出 (Raw Output)**:\n```\n{item['raw_response']}\n```\n")

        lines.append("---")
        lines.append("")

    return "\n".join(lines)


def main() -> int:
    """
    【批处理主执行函数】
    控制整篇论文从读取、过滤、断点续跑、并发限流到增量落盘的全生命周期。
    """
    args = parse_args()                                         # 解析命令行参数
    paper_dir = Path(args.paper_folder).resolve()               # 将论文目录解析为绝对路径
    input_claims_file = paper_dir / "extracted_claims.json"     # 定位上游 PDF 阶段提取出的原始段落 JSON

    # 防呆校验：确保输入文件存在
    if not input_claims_file.is_file():
        print(f"[!] 错误: 输入文件不存在: {input_claims_file}", file=sys.stderr)
        return 1

    # 加载整篇论文的原始段落数据
    with open(input_claims_file, "r", encoding="utf-8") as f:
        root_data = json.load(f)

    all_claims = root_data.get("claims", [])                    # 获取全篇所有段落主张列表
    paper_id = root_data.get("paper_id", paper_dir.name)        # 获取论文唯一标识
    paper_title = root_data.get("paper_title", paper_id)        # 获取论文标题

    # 计算本次要处理的目标切片范围（支持 --start 和 --limit 切片）
    effective_limit = args.limit if args.limit > 0 else len(all_claims)
    target_claims = all_claims[args.start : args.start + effective_limit]
    if not target_claims:
        print(f"[!] 警告: 指定范围 [{args.start}:{args.start+effective_limit}] 内无可用段落。")
        return 0

    validator = ClaimValidator()                                # 实例化确定性校验法官

    # ==============================================================================
    # 【--dry-run 干跑预览模式】：一分钱不花，打印渲染后的完整 Prompt 供调试
    # ==============================================================================
    if args.dry_run:
        extractor = AtomicClaimExtractor(
            llm=None,                                           # 干跑模式无需真实 LLM 实例
            model_name="dry-run",
            validator=validator,
            max_tokens=args.max_tokens,
        )
        sample = target_claims[0]
        preview_prompt = extractor.preview(sample)              # 生成首个段落的渲染后文本
        print("=" * 80)
        print(f">> [DRY-RUN] 预览首个段落 ({sample.get('claim_id')}) 渲染后的完整 Prompt:")
        print("=" * 80)
        print(preview_prompt)
        print("=" * 80)
        return 0

    # 实际运行：解析目标输出文件路径
    out_json_path = Path(args.out) if args.out else paper_dir / "atomic_claims.json"
    out_review_path = Path(args.review_out) if args.review_out else paper_dir / "atomic_claims_review.md"

    # ==============================================================================
    # 【--resume 断点续跑机制】：读取已有成功记录，避免重复调用浪费 Token
    # ==============================================================================
    cached_ok_items: Dict[str, dict] = {}
    if args.resume and out_json_path.is_file():
        try:
            existing_data = json.loads(out_json_path.read_text(encoding="utf-8"))
            for it in existing_data.get("items", []):
                if it.get("status") in ("ok", "skipped_no_citation") and it.get("parent_claim_id"):
                    cached_ok_items[it["parent_claim_id"]] = it # 缓存历史成功项
        except Exception as e:
            logger.warning(f"读取已有断点文件失败，将从头运行: {e}")

    # 获取真实的大模型客户端连接
    client, model_name = get_llm_client(role=args.role)
    extractor = AtomicClaimExtractor(
        llm=client,
        model_name=model_name,
        validator=validator,
        max_tokens=args.max_tokens,
    )

    # 打印启动横幅信息
    print("=" * 80)
    print(">> 启动 CiteTrace 阶段 1：原子主张提取与契约强校验")
    print(f">> 论文: {paper_title} ({paper_id})")
    print(f">> 模型: {model_name} (Role: {args.role})")
    print(f">> 处理区间: claims[{args.start} : {args.start + len(target_claims)}] (共 {len(target_claims)} 条段落)")
    if cached_ok_items:
        print(f">> 断点续跑开启: 已加载 {len(cached_ok_items)} 条历史成功段落")
    print("=" * 80)

    items: List[dict] = []                                      # 存储当前运行的所有段落结果记录
    table_rows = []                                             # 控制台表格行收集器

    def _persist_progress() -> dict:
        """【增量原子持久化闭包】：每抽取完一段立即刷写落盘，保证进度永不丢失"""
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
        write_atomic_json(out_json_path, full_output)           # 原子写 JSON
        review_md_text = build_review_markdown(paper_id, paper_title, model_name, cur_summary, items)
        write_atomic_markdown(out_review_path, review_md_text)  # 原子写 Markdown
        return cur_summary

    # ==============================================================================
    # 【段落流水线主循环】：逐段落遍历执行
    # ==============================================================================
    for i, claim_ctx in enumerate(target_claims, start=1):
        pid = claim_ctx.get("claim_id", f"claim_{i}")
        citation_keys = claim_ctx.get("citation_keys", [])

        # 1. 命中断点续跑缓存：直接复用已有结果
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

        # 2. 正常处理当前段落
        print(
            f"[{i}/{len(target_claims)}] 正在处理 {pid} (涉及 {len(citation_keys)} 个引用: {', '.join(citation_keys)})...",
            end="",
            flush=True,
        )

        # 调用编排器（进行抽取、校验与自愈闭环）
        outcome: ExtractionOutcome = extractor.extract(claim_ctx)

        # 打印控制台即时状态
        is_ok = outcome.status in ("ok", "skipped_no_citation")
        if is_ok:
            print(f" -> [{outcome.status}] 提取出 {len(outcome.claims)} 条原子主张 (尝试 {outcome.attempts} 次)")
        else:
            print(f" -> [FAILED: {outcome.status}] {outcome.error}")

        # 序列化当前段落的校验报告
        val_dict = {
            "valid": outcome.validation.valid if outcome.validation else False,
            "errors": outcome.validation.errors if outcome.validation else ([outcome.error] if outcome.error else []),
            "warnings": outcome.validation.warnings if outcome.validation else [],
            "repaired_spans": outcome.validation.repaired_spans if outcome.validation else 0,
        }

        claims_dump = [c.model_dump() for c in outcome.claims]
        req_review_count = sum(1 for c in outcome.claims if c.requires_review)

        # 打包该段落的完整数据档案
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

        # 收集控制台展示数据
        table_rows.append({
            "pid": pid,
            "status": outcome.status,
            "claims_count": len(outcome.claims),
            "keys_count": len(citation_keys),
            "errs": len(val_dict["errors"]),
            "warns": len(val_dict["warnings"]),
            "reviews": req_review_count,
        })

        # 【核心安全机制】：每处理完一段立即增量落盘，中途哪怕死机，之前的段落也全部安全保存在硬盘！
        _persist_progress()

        # 请求间隔微休眠：礼貌调用，防止 API 429 报错
        if i < len(target_claims) and args.sleep > 0:
            time.sleep(args.sleep)

    # 循环完毕，最后执行一次落盘统计
    summary = _persist_progress()
    print(f"\n[✓] 产物已保存至 JSON: {out_json_path}")
    print(f"[✓] 产物已保存至 Markdown: {out_review_path}")

    # ==============================================================================
    # 【控制台可视化总表】：以 ASCII 表格形式汇总整批执行质量
    # ==============================================================================
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

    # 规范退出码：若全部成功返回 0；若存在失败段落返回 1，方便 CI/CD 自动化流水线捕获状态
    exit_code = 0 if summary["parents_failed"] == 0 else 1
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
