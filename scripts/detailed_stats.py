import json
from pathlib import Path

base = Path("cache/papers/Attention_Is_All_You_Need")
cited_dir = base / "cited_papers"

total_tables = 0
total_prose = 0
total_chars = 0
paper_summaries = []

for p in sorted(cited_dir.iterdir(), key=lambda x: int(x.name.strip("[]")) if x.name.strip("[]").isdigit() else 999):
    if not p.is_dir():
        continue
    c_file = p / "chunks.json"
    m_file = p / "metadata.json"
    sec_dir = p / "sections"
    
    tables = 0
    prose = 0
    chars = 0
    if c_file.exists():
        chunks = json.loads(c_file.read_text(encoding="utf-8"))
        for c in chunks:
            c_type = c.get("content_type", "prose")
            if c_type == "table":
                tables += 1
            else:
                prose += 1
            chars += c.get("char_count", len(c.get("text", "")))
    
    total_tables += tables
    total_prose += prose
    total_chars += chars
    
    meta = json.loads(m_file.read_text(encoding="utf-8")) if m_file.exists() else {}
    sec_count = len(list(sec_dir.glob("*.md"))) if sec_dir.exists() else 0
    
    paper_summaries.append({
        "id": p.name,
        "title": meta.get("paper_title", "Unknown"),
        "pages": meta.get("total_pages", 0),
        "sections": sec_count,
        "prose": prose,
        "tables": tables,
        "total_chunks": prose + tables,
        "chars": chars
    })

print(f"Total Papers Processed: {len(paper_summaries)}")
print(f"Total Prose Chunks: {total_prose}")
print(f"Total Table Chunks: {total_tables}")
print(f"Total All Chunks: {total_prose + total_tables}")
print(f"Total Chars: {total_chars}")
print(f"Average Chars per Chunk: {total_chars / (total_prose + total_tables):.1f}")

print("\n| 编号 | 标题 | 页数 | 章节数 | 文本块 (Prose) | 表格块 (Table) | 总块数 | 字符数 |")
print("| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: |")
for p in paper_summaries:
    print(f"| {p['id']} | {p['title'][:36]} | {p['pages']} | {p['sections']} | {p['prose']} | {p['tables']} | {p['total_chunks']} | {p['chars']:,} |")

