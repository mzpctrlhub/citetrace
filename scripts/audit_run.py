# scripts/audit_run.py
import json
from pathlib import Path

base = Path("cache/papers/Attention_Is_All_You_Need")
cited_dir = base / "cited_papers"
refs_file = base / "references.json"
references_dir = base / "references"
citations_dir = base / "citations"

ref_catalog = json.loads(refs_file.read_text(encoding="utf-8")) if refs_file.exists() else []

print("=" * 80)
print(f">> CiteTrace 40 篇参考文献划分全量质量审计报告")
print("=" * 80)

# 1. 检查 references/
ref_mds = list(references_dir.glob("*.md")) if references_dir.is_dir() else []
print(f"\n[1. 文末参考文献条目切分 (references/)]")
print(f"  - references.json 解析条目数: {len(ref_catalog)} 篇")
print(f"  - 独立 .md 条目卡片生成数:   {len(ref_mds)} 个文件 (references/*.md)")

# 2. 检查 citations/ (正文引用提取)
claim_mds = [f for f in citations_dir.glob("*.md") if f.name != "index.md"] if citations_dir.is_dir() else []
print(f"\n[2. 正文引用段落提取与对齐 (citations/)]")
print(f"  - 正文引用点卡片生成数:       {len(claim_mds)} 处 (citations/1.md ~ {len(claim_mds)}.md)")

# 3. 检查 cited_papers/ (被引文献全文切分)
print(f"\n[3. 被引文献正文 PDF 下载、章节切分与 Chunker 切块 (cited_papers/)]")
results = []
for item in ref_catalog:
    key = item["canonical_id"]
    title = item.get("title") or item.get("raw_text", "")[:45]
    p_dir = cited_dir / key
    has_pdf = (p_dir / "raw.pdf").is_file()
    has_meta = (p_dir / "metadata.json").is_file()
    has_chunks = (p_dir / "chunks.json").is_file()

    sections = list((p_dir / "sections").glob("*.md")) if (p_dir / "sections").is_dir() else []
    chunks_count = 0
    if has_chunks:
        try:
            chunks_count = len(json.loads((p_dir / "chunks.json").read_text(encoding="utf-8")))
        except Exception:
            chunks_count = -1

    meta_info = {}
    if has_meta:
        try:
            meta_info = json.loads((p_dir / "metadata.json").read_text(encoding="utf-8"))
        except Exception:
            pass

    results.append({
        "key": key,
        "title": meta_info.get("paper_title", title),
        "has_pdf": has_pdf,
        "has_meta": has_meta,
        "has_chunks": has_chunks,
        "sections_count": len(sections),
        "chunks_count": chunks_count,
        "total_pages": meta_info.get("total_pages", 0),
        "audit_passed": meta_info.get("outline_audit", {}).get("passed", False),
        "pdf_size_kb": round((p_dir / "raw.pdf").stat().st_size / 1024, 1) if has_pdf else 0
    })

successful = [r for r in results if r["has_pdf"] and r["has_chunks"] and r["chunks_count"] > 0]
failed = [r for r in results if not (r["has_pdf"] and r["has_chunks"] and r["chunks_count"] > 0)]

total_sections = sum(r["sections_count"] for r in successful)
total_chunks = sum(r["chunks_count"] for r in successful)

print(f"  - 完整切分成功文献数:         {len(successful)} / {len(ref_catalog)} (成功率: {len(successful)/max(len(ref_catalog),1)*100:.1f}%)")
print(f"  - 提取总章节数:               {total_sections} 个独立章节 Markdown")
print(f"  - 生成总证据切块数:           {total_chunks} 个细粒度 Chunks")
if successful:
    print(f"  - 平均单篇文献章节数:         {total_sections / len(successful):.1f} 个章节/篇")
    print(f"  - 平均单篇文献 Chunks 数:      {total_chunks / len(successful):.1f} 个 chunks/篇")

print(f"\n【前 10 篇被引文献切分明细】:")
print(f"| 序号 | 论文标题 | PDF大小 | 章节数 | Chunks数 | 总页数 | 审计 |")
print(f"| :---: | :--- | :---: | :---: | :---: | :---: | :---: |")
for r in results[:10]:
    status = "✓" if r["has_chunks"] and r["chunks_count"] > 0 else "✗"
    audit_icon = "✓" if r["audit_passed"] else "-"
    title_short = r["title"][:32]
    print(f"| {r['key']} | {title_short:<32} | {r['pdf_size_kb']} KB | {r['sections_count']} 节 | {r['chunks_count']} 块 | {r['total_pages']}P | {audit_icon} |")

if len(results) > 10:
    print(f"... (篇目 [11] ~ [{len(results)}] 略，见统计汇总)")

if failed:
    print(f"\n【未成功获取/解析的文献清单 ({len(failed)} 篇)】:")
    for f in failed:
        print(f"  - {f['key']}: 《{f['title']}》 [PDF存在: {f['has_pdf']}, Chunks存在: {f['has_chunks']}]")
else:
    print(f"\n[✓] 绝佳！全部 {len(ref_catalog)} 篇参考文献正文 PDF 100% 成功下载并完成章节切分与细粒度切块！")

print("\n" + "=" * 80)
