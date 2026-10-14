# demo_pdf_split.py
"""
CiteTrace PDF 切分与细粒度切块纯离线测试脚本 (不调用任何大模型)

专用于测试与验证：
1. PDF 文本/样式/标题抽取与大纲切分 (Section Markdown 切分)
2. 边界框表格物理重构 (BBox Table Markdown 还原)
3. 细粒度证据块切分 (chunks.json，段落与表格切块)
"""

import argparse
import json
import os
import sys
from pathlib import Path

from citetrace_pdf import PDFProcessor
from citetrace_pdf.chunker import chunk_paper_folder


def run_pdf_split_demo():
    parser = argparse.ArgumentParser(description="CiteTrace PDF 纯离线切分与切块测试 (无需大模型)")
    parser.add_argument("pdf", nargs="?", default=None, help="待切分 PDF 路径 (默认使用本地基准 Attention 论文)")
    parser.add_argument("--cache-dir", default="cache/papers", help="输出缓存目录 (默认 cache/papers)")
    parser.add_argument("--force", action="store_true", help="强制重新切分，不读取旧缓存")
    parser.add_argument("--show-chunks", type=int, default=3, help="打印前 N 个 chunk 的详细内容 (默认 3)")
    parser.add_argument("--split-cited", type=int, default=1, help="同时下载并切分前 N 篇被引文献正文 PDF (默认 1，0 为跳过)")
    args = parser.parse_args()

    print("=" * 80)
    print(">> CiteTrace 纯离线 PDF 划分与切块测试 (0 大模型调用 / 0 成本)")
    print("=" * 80)

    # 1. 寻找待测试 PDF
    target_pdf = None
    if args.pdf:
        p = Path(args.pdf).resolve()
        if p.is_file():
            target_pdf = p
        else:
            print(f"[!] 指定的 PDF 文件不存在: {p}")
            return
    else:
        candidate_pdfs = [
            Path("cache/papers/Attention_Is_All_You_Need/raw.pdf"),
            Path("cache/papers/1512.03385/raw.pdf"),
        ]
        for c in candidate_pdfs:
            if c.is_file():
                target_pdf = c.resolve()
                break
        if not target_pdf and Path("cache").is_dir():
            for root, _dirs, files in os.walk("cache"):
                for f in files:
                    if f.endswith(".pdf"):
                        target_pdf = (Path(root) / f).resolve()
                        break
                if target_pdf:
                    break

    if not target_pdf:
        print("[*] 本地 cache 暂无 PDF，正在全自动下载基准论文 (Attention Is All You Need, 1706.03762)...")
        from citetrace_pdf.pdf.loader import PDFLoader
        loader = PDFLoader(cache_dir=args.cache_dir)
        try:
            downloaded = loader.load("Attention_Is_All_You_Need", "https://arxiv.org/pdf/1706.03762.pdf")
            target_pdf = Path(downloaded).resolve()
            print(f"[✓] 基准论文下载完成: {target_pdf} ({target_pdf.stat().st_size / 1024:.1f} KB)")
        except Exception as e:
            print(f"[!] 自动下载失败: {e}")
            print("[!] 请手动指定已有 PDF 路径: python demo_pdf_split.py <path_to_pdf>")
            return

    print(f"\n[待切分 PDF]: {target_pdf}")
    print(f"[文件大小]:   {target_pdf.stat().st_size / 1024:.1f} KB")

    # 2. 运行 PDFProcessor 进行抽取与章节切分
    print("\n" + "-" * 80)
    print(">> [阶段 1/2] 正在执行 PDF 文本抽取、多层大纲识别与章节切分...")
    print("-" * 80)

    processor = PDFProcessor(cache_dir=args.cache_dir)
    metadata = processor.process_local_pdf(
        pdf_path=target_pdf,
        force=args.force,
    )

    paper_folder = Path(metadata["folder"])
    outline = metadata.get("outline", [])
    paper_title = metadata.get("paper_title", "Unknown")
    total_pages = metadata.get("total_pages", 0)

    print(f"[✓] 论文标题解析:  《{paper_title}》 (置信度: {metadata.get('title_confidence', 'N/A')})")
    print(f"[✓] 标题抽取来源:   {metadata.get('title_source', 'N/A')}")
    print(f"[✓] PDF 总页数:    {total_pages} 页")
    print(f"[✓] 产物输出目录:   {paper_folder}")

    # 打印章节列表
    print(f"\n【切分出的独立章节列表】 (共 {len(outline)} 个章节):")
    sections_dir = paper_folder / "sections" if (paper_folder / "sections").is_dir() else paper_folder
    for idx, sec in enumerate(outline, 1):
        fname = sec.get("file", "")
        sec_path = sections_dir / fname if fname else None
        char_count = len(sec_path.read_text(encoding="utf-8")) if sec_path and sec_path.is_file() else 0
        page_info = f"P.{sec.get('page_start', '?')}"
        if sec.get("page_end") and sec.get("page_end") != sec.get("page_start"):
            page_info += f"-{sec.get('page_end')}"
        print(f"  {idx:2d}. [{page_info:<6}] {sec.get('title', 'Untitled'):<38} -> {fname} ({char_count} 字符)")

    # 检查是否有重构的表格
    tables_found = []
    for md_file in sections_dir.glob("*.md"):
        content = md_file.read_text(encoding="utf-8", errors="ignore")
        if "| ---" in content or "|:---" in content:
            tables_found.append(md_file.name)
    if tables_found:
        print(f"\n[✓] 在以下章节中成功重构/包含结构化 Markdown 表格: {', '.join(tables_found)}")

    # 3. 运行 Chunker 进行细粒度切块
    print("\n" + "-" * 80)
    print(">> [阶段 2/2] 正在执行段落级语义切块 (Chunking) 并生成 chunks.json...")
    print("-" * 80)

    chunks = chunk_paper_folder(paper_folder)
    chunks_file = paper_folder / "chunks.json"

    prose_count = sum(1 for c in chunks if c.get("content_type") == "prose")
    table_count = sum(1 for c in chunks if c.get("content_type") == "table")
    avg_len = sum(c.get("char_count", len(c.get("text", ""))) for c in chunks) / max(len(chunks), 1)

    print(f"[✓] 切块总数:       {len(chunks)} 个 chunks")
    print(f"    - 正文段落块:   {prose_count} 个")
    print(f"    - 结构化表格块: {table_count} 个")
    print(f"    - 平均块长度:   {avg_len:.1f} 字符")
    print(f"[✓] 切块结果已保存: {chunks_file} ({chunks_file.stat().st_size / 1024:.1f} KB)")

    # 4. 打印前 N 个 Chunk 样本预览
    if args.show_chunks > 0 and chunks:
        print(f"\n【前 {min(args.show_chunks, len(chunks))} 个 Chunk 样本预览】:")
        for i, c in enumerate(chunks[:args.show_chunks], 1):
            sec_name = c.get("section_title") or c.get("section_file")
            print(f"\n>> [Chunk #{i}] ID: {c.get('chunk_id')} | 类型: {c.get('content_type')} | 章节: {sec_name}")
            snippet = (c.get("text") or "").strip()
            if len(snippet) > 220:
                snippet = snippet[:220] + " ... [截断]"
            print(f"   内容:\n   \"{snippet}\"")

    # 5. 打印表格样本预览
    table_chunks = [c for c in chunks if c.get("content_type") == "table"]
    if table_chunks:
        print(f"\n【重构的物理表格 Chunk 样本预览】 (共 {len(table_chunks)} 个表格):")
        tc = table_chunks[0]
        sec_name = tc.get("section_title") or tc.get("section_file")
        print(f">> [表格 ID: {tc.get('chunk_id')}] (来自: {sec_name})")
        print(tc.get("text", "").strip())

    # 6. 运行 extract_from_paper_folder 进行参考文献条目细分与正文引用段落对齐
    print("\n" + "-" * 80)
    print(">> [阶段 3/3] 正在执行参考文献条目细分 (References Parsing) 与正文引用对齐...")
    print("-" * 80)
    from parsers.folder_extractor import extract_from_paper_folder
    claims_data = extract_from_paper_folder(str(paper_folder), paper_id=paper_folder.name)

    refs_file = paper_folder / "references.json"
    refs_list = []
    if refs_file.is_file():
        refs_list = json.loads(refs_file.read_text(encoding="utf-8"))

    citations_dir = paper_folder / "citations"
    claims_list = claims_data.get("claims", [])

    print(f"[✓] 参考文献细分条目: {len(refs_list)} 条 (已保存至: {refs_file})")
    print(f"[✓] 正文引用段落提取: {len(claims_list)} 处 (已生成独立卡片: {citations_dir}/)")
    print(f"[✓] 结构化引用清单:   {paper_folder / 'extracted_claims.json'}")
    print(f"[✓] 人类可读对齐报告: {paper_folder / 'extracted_claims.md'}")

    # 打印参考文献条目样本
    if refs_list:
        print(f"\n【解析划分出的参考文献条目样本】 (前 5 篇 / 共 {len(refs_list)} 篇):")
        for ref in refs_list[:5]:
            ref_idx = ref.get("canonical_id") or f"[{ref.get('index')}]"
            title_str = ref.get("title") or ref.get("raw_text", "")[:45]
            year_str = f" ({ref.get('year')})" if ref.get('year') else ""
            author_str = f" - {ref.get('authors')[0]} 等" if ref.get('authors') else ""
            arx_str = f" [arXiv: {ref.get('arxiv_id')}]" if ref.get('arxiv_id') else ""
            doi_str = f" [DOI: {ref.get('doi')}]" if ref.get('doi') else ""
            print(f"  {ref_idx:<6} 《{title_str}》{author_str}{year_str}{arx_str}{doi_str}")

    # 打印正文引用对齐卡片样本
    if claims_list:
        print(f"\n【正文引用段落与参考文献对齐样本】 (前 2 处 / 共 {len(claims_list)} 处):")
        for cl in claims_list[:2]:
            print(f"\n>> [{cl.get('claim_id')}] (来自: {cl.get('source_section')} | 引用标记: {', '.join(cl.get('citation_keys', []))})")
            para_snip = cl.get("paragraph", "").replace("\n", " ").strip()
            if len(para_snip) > 160:
                para_snip = para_snip[:160] + " ... [截断]"
            print(f"   正文段落: \"{para_snip}\"")
            print("   关联的参考文献:")
            for k in cl.get("citation_keys", [])[:3]:
                ref_txt = cl.get("matched_references", {}).get(k, "")
                if len(ref_txt) > 80:
                    ref_txt = ref_txt[:80] + "..."
                print(f"     - {k}: {ref_txt}")
            if len(cl.get("citation_keys", [])) > 3:
                print(f"     ... 其余 {len(cl.get('citation_keys')) - 3} 篇省略")

    # 物理切分文末参考文献为独立条目 Markdown 文件
    references_dir = paper_folder / "references"
    references_dir.mkdir(parents=True, exist_ok=True)
    for ref in refs_list:
        canon_id = ref.get("canonical_id", "").strip("[]")
        ref_file = references_dir / f"{canon_id}.md"
        content_lines = [
            f"# 参考文献 {ref.get('canonical_id')}",
            f"- **标题**: {ref.get('title') or '-'}",
            f"- **作者**: {', '.join(ref.get('authors', [])) or '-'}",
            f"- **年份**: {ref.get('year') or '-'}",
            f"- **arXiv ID**: {ref.get('arxiv_id') or '-'}",
            f"- **DOI**: {ref.get('doi') or '-'}",
            f"- **原始条目**: {ref.get('raw_text')}",
        ]
        ref_file.write_text("\n".join(content_lines) + "\n", encoding="utf-8")

    # 阶段 4：被引文献全文 PDF 的切分与切块 (Cited Papers Slicing)
    cited_papers_dir = paper_folder / "cited_papers"
    if args.split_cited > 0 and refs_list:
        print("\n" + "-" * 80)
        print(f">> [阶段 4/4] 正在执行被引文献正文 PDF 切分与细粒度切块 (演示前 {args.split_cited} 篇)...")
        print("-" * 80)
        from models.citation import CitationRef
        from tools.academic_fetcher import fetch_paper_document

        cited_papers_dir.mkdir(parents=True, exist_ok=True)

        for ref_data in refs_list[:args.split_cited]:
            ref_obj = CitationRef(
                key=ref_data["canonical_id"],
                raw_text=ref_data.get("raw_text", ""),
                title=ref_data.get("title"),
                authors=ref_data.get("authors", []),
                year=ref_data.get("year"),
                doi=ref_data.get("doi"),
                arxiv_id=ref_data.get("arxiv_id")
            )
            print(f"[*] 正在获取并切分被引文献 {ref_obj.key}: 《{ref_obj.title or ref_obj.raw_text[:40]}》...")
            resolved_ref, cited_meta = fetch_paper_document(
                ref_obj,
                cache_dir=str(cited_papers_dir),
                paper_id=ref_obj.key,
                global_cache_dir=str(paper_folder.parent)
            )
            if cited_meta and cited_meta.get("folder"):
                cfolder = Path(cited_meta["folder"])
                csections = list((cfolder / "sections").glob("*.md")) if (cfolder / "sections").is_dir() else []
                cchunks = json.loads((cfolder / "chunks.json").read_text(encoding="utf-8")) if (cfolder / "chunks.json").is_file() else []
                print(f"  [✓] 被引文献 {ref_obj.key} 切分完成！")
                print(f"      - 存放文献夹: {cfolder}")
                print(f"      - 独立章节数: {len(csections)} 个章节 (见 {cfolder / 'sections'}/)")
                print(f"      - 证据切块数: {len(cchunks)} 个细粒度 Chunks (见 {cfolder / 'chunks.json'})")
            else:
                print(f"  [!] 被引文献 {ref_obj.key} 获取失败: {resolved_ref.last_failure}")

    # 7. 查看说明
    print("\n" + "=" * 80)
    print("[✓] 论文全量离线解析与划分顺利完成！")
    print(f"你可以直接在 VS Code 或文件管理器中查看所有划分产物：")
    print(f"  1. 待核查论文章节目录:    {sections_dir}/")
    print(f"  2. 待核查论文切块文件:    {chunks_file}")
    print(f"  3. 参考文献独立条目目录:  {references_dir}/ (包含 1.md ~ {len(refs_list)}.md)")
    print(f"  4. 参考文献结构化元数据:  {refs_file}")
    print(f"  5. 正文引用段落对齐卡片:  {citations_dir}/ (包含 1.md ~ {len(claims_list)}.md)")
    if args.split_cited > 0 and cited_papers_dir.is_dir():
        print(f"  6. 被引文献全文切分库:    {cited_papers_dir}/ (各被引文献独立章节与 chunks)")
    print("=" * 80)


if __name__ == "__main__":
    run_pdf_split_demo()
