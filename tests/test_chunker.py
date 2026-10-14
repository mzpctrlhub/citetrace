"""证据切块层 (chunker) 单元测试 —— 完全离线，用临时目录构造最小文献夹"""
import json
import unittest
from pathlib import Path
import tempfile

from citetrace_pdf.chunker import (
    chunk_paper_folder,
    locate_quote,
    _split_paragraphs,
    _split_table_blocks,
    _split_table,
    _merge_short,
    _split_long,
)


def _make_paper_folder(root: Path) -> Path:
    folder = root / "sample_paper"
    folder.mkdir(parents=True)
    (folder / "metadata.json").write_text(json.dumps({
        "paper_id": "sample_001",
        "paper_title": "Sample Paper",
        "outline": [
            {"index": 0, "title": "Abstract", "file": "00_Abstract.md",
             "start_page": 1, "end_page": 1},
            {"index": 1, "title": "1 Introduction", "file": "01_Introduction.md",
             "start_page": 2, "end_page": 2},
            {"index": 2, "title": "References", "file": "References.md",
             "start_page": 10, "end_page": 11},
        ],
    }), encoding="utf-8")

    abstract = (
        "# Abstract\n\n> **所属论文**: Sample Paper (sample_001)\n> **对应页码**: 第 1 ~ 1 页\n\n---\n\n"
        + ("The transformer architecture removes recurrence and relies entirely on attention mechanisms. " * 8)
    )
    intro = (
        "# 1 Introduction\n\n> **所属论文**: Sample Paper (sample_001)\n> **对应页码**: 第 2 ~ 2 页\n\n---\n\n"
        + "Recurrent networks process tokens sequentially.\n\n"
        + "| Model | EN-DE BLEU |\n| --- | --- |\n| GNMT (Word pieces) | 24.61 |\n| ConvS2S (BPE 40K) | 25.16 |\n\n"
        + "_Table 1._ Accuracy on WMT tasks compared to previous work.\n\n"
        + "> **章节类型**: 导引章节\n\n"
        + ("Self-attention computes weighted sums over all positions in parallel. " * 80)
    )
    refs = "# References\n\n[1] Some Author. Some title. 2020.\n"

    (folder / "00_Abstract.md").write_text(abstract, encoding="utf-8")
    (folder / "01_Introduction.md").write_text(intro, encoding="utf-8")
    (folder / "References.md").write_text(refs, encoding="utf-8")
    return folder


class TestChunker(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = _make_paper_folder(Path(self.tmp.name))

    def tearDown(self):
        self.tmp.cleanup()

    def test_chunk_paper_folder_writes_chunks_json(self):
        chunks = chunk_paper_folder(self.folder)
        self.assertGreaterEqual(len(chunks), 3)

        out = self.folder / "chunks.json"
        self.assertTrue(out.is_file())
        loaded = json.loads(out.read_text(encoding="utf-8"))
        self.assertEqual(len(loaded), len(chunks))

        # References 章节必须被跳过
        for c in chunks:
            self.assertNotEqual(c["section_file"], "References.md")
            self.assertTrue(c["chunk_id"].startswith("sample_001::"))
            self.assertIsNotNone(c["page_start"])
            self.assertGreater(c["char_count"], 0)

        # chunk_id 全局唯一
        ids = [c["chunk_id"] for c in chunks]
        self.assertEqual(len(ids), len(set(ids)))

    def test_max_chars_respected_with_sliding_window(self):
        chunks = chunk_paper_folder(self.folder, max_chars=600)
        self.assertGreater(len(chunks), 1)
        for c in chunks:
            self.assertLessEqual(c["char_count"], 600 * 1.05)

    def test_locate_quote_positive_and_negative(self):
        chunks = chunk_paper_folder(self.folder)
        # 正例：截取某 chunk 原文中真实存在的一句
        probe = chunks[0]["text"].split(". ")[0]
        hit = locate_quote(chunks, probe)
        self.assertIsNotNone(hit)
        self.assertEqual(hit["chunk_id"], chunks[0]["chunk_id"])

        # 反例：不存在的句子必须返回 None
        self.assertIsNone(locate_quote(chunks, "这句话绝对不存在于任何 chunk 中 xyzzy"))
        self.assertIsNone(locate_quote(chunks, ""))

    def test_locate_quote_rejects_truncated_numbers(self):
        """回归锁：小数点曾被当格式噪声删掉，写错一位小数的数字能蒙过【逐字】校验"""
        chunks = [{"chunk_id": "t", "text": "|ConvS2S (BPE 40K)|25.16|\n|GNMT (Word pieces)|24.61|"}]
        self.assertIsNotNone(locate_quote(chunks, "25.16"))
        self.assertIsNotNone(locate_quote(chunks, "24.61"))
        for fabricated in ("25.1", "25.2", "24.6", "24.612", "61", "2516", "25."):
            self.assertIsNone(
                locate_quote(chunks, fabricated),
                msg=f"{fabricated!r} 不存在于证据中，必须驳回",
            )

    def test_locate_quote_accepts_numerically_identical_spelling(self):
        """28.10 / 28.1 / 2,8.10 尾零与千分位差异不是幻觉：PDF 抽取常把 28.10 写成 28.1"""
        text = "the model reaches 28.1 BLEU and 3,028 seconds"
        chunks = [{"chunk_id": "t", "text": text}]
        self.assertIsNotNone(locate_quote(chunks, "28.10"))
        self.assertIsNotNone(locate_quote(chunks, "3028"))
        self.assertIsNotNone(locate_quote(chunks, "3,028"))

    def test_locate_quote_matches_on_token_boundaries(self):
        """引用必须落在词边界上，不再允许从单词中间截取"""
        chunks = [{"chunk_id": "t", "text": "a **convolutional** sequence model — \"attention\""}]
        self.assertIsNotNone(locate_quote(chunks, "convolutional"))     # Markdown 星号被忽略
        self.assertIsNotNone(locate_quote(chunks, '"attention"'))       # 弯引号被忽略
        self.assertIsNotNone(locate_quote(chunks, "attention"))
        self.assertIsNone(locate_quote(chunks, "convolution"))          # 半截词不再算逐字

    def test_locate_quote_keeps_chinese_substring_behaviour(self):
        chunks = [{"chunk_id": "t", "text": "该方法提出了新的注意力机制"}]
        self.assertIsNotNone(locate_quote(chunks, "注意力"))
        self.assertIsNone(locate_quote(chunks, "不存在的方法"))

    def test_locate_quote_number_lists_are_not_swallowed_whole(self):
        """逗号是分隔符不是数字的一部分：[-0.2,0.2] 必须能查到其中的 0.2"""
        chunks = [{"chunk_id": "t", "text": "weights are initialized in [-0.2,0.2]. See 1,234 and 24.119.3.44"}]
        for probe in ("0.2", "-0.2", "1,234", "1234", "24.119", "3.44"):
            self.assertIsNotNone(locate_quote(chunks, probe), msg=f"{probe!r} 确实存在，不应驳回")
        for wrong in ("0.02", "0.3", "0.22", "1,324"):
            self.assertIsNone(locate_quote(chunks, wrong), msg=f"{wrong!r} 不存在，必须驳回")

    def test_split_paragraphs_keeps_tables_and_filters_noise(self):
        text = "# Title\n\n> meta line\n\n---\n\nReal paragraph one.\n\n| a | b |\n| --- | --- |\n\nReal paragraph two."
        paras = _split_paragraphs(text)
        self.assertEqual(len(paras), 3)
        self.assertTrue(paras[0].startswith("Real paragraph one."))
        # 表格必须作为单个完整段落保留，不能再被过滤掉
        self.assertTrue(paras[1].startswith("|"))
        self.assertIn("| a | b |", paras[1])
        self.assertIn("| --- | --- |", paras[1])

    def test_table_survives_into_chunks_with_header_and_caption(self):
        chunks = chunk_paper_folder(self.folder)
        table_chunks = [c for c in chunks if "25.16" in c["text"]]
        self.assertEqual(len(table_chunks), 1, "表格数值必须能在 chunks.json 中找到")

        tc = table_chunks[0]
        # 表头与数据行必须在同一个 chunk，否则数字失去列名
        self.assertIn("Model", tc["text"])
        self.assertIn("EN-DE BLEU", tc["text"])
        self.assertIn("24.61", tc["text"])
        # 表题被并入表格块，而不是作为孤立散文
        self.assertIn("_Table 1._", tc["text"])
        # 表格是原子块：绝不与散文合并
        self.assertNotIn("Self-attention", tc["text"])
        self.assertNotIn("Recurrent networks", tc["text"])
        for c in chunks:
            if not c["text"].lstrip().startswith("_Table") and "|" not in c["text"]:
                self.assertNotIn("25.16", c["text"])

    def test_caption_is_not_emitted_as_standalone_prose(self):
        chunks = chunk_paper_folder(self.folder)
        standalone = [
            c for c in chunks
            if "_Table 1._" in c["text"] and not c["text"].lstrip().startswith("|")
            and "|" not in c["text"]
        ]
        self.assertEqual(standalone, [], "表题不应脱离表格单独成为 chunk")

    def test_split_table_blocks_pairs_caption_in_both_directions(self):
        # pymupdf4llm 常见：表题在表格之后
        after = ["prose", "| a |\n| --- |\n| 1 |", "_Table 1._ cap after"]
        prose, tables = _split_table_blocks(after)
        self.assertEqual(prose, ["prose"])
        self.assertEqual(len(tables), 1)
        self.assertTrue(tables[0].startswith("_Table 1._ cap after"))

        # 手写 Markdown：表题在表格之前
        before = ["Table 2: cap before", "| a |\n| --- |\n| 1 |", "prose"]
        prose2, tables2 = _split_table_blocks(before)
        self.assertEqual(prose2, ["prose"])
        self.assertTrue(tables2[0].startswith("Table 2: cap before"))

    def test_long_table_splits_by_rows_and_repeats_header(self):
        rows = [f"| model-{i} | {i}.00 |" for i in range(60)]
        block = "_Table 3._ Big table.\n| Model | BLEU |\n| --- | --- |\n" + "\n".join(rows)
        pieces = _split_table(block, max_chars=300)

        self.assertGreater(len(pieces), 1, "超长表格必须分片")
        for p in pieces:
            # 每一片都要自带表头与表题，不能出现没有列名的孤立数据行
            self.assertIn("| Model | BLEU |", p)
            self.assertIn("| --- | --- |", p)
            self.assertIn("_Table 3._", p)
            self.assertLessEqual(len(p), 300)
        # 数据行无丢失、无重复
        emitted = [ln for p in pieces for ln in p.splitlines() if ln.startswith("| model-")]
        self.assertEqual(len(emitted), 60)
        self.assertEqual(len(set(emitted)), 60)

    def test_single_overlong_row_is_not_hard_cut(self):
        block = "| H |\n| --- |\n" + "| " + "x" * 500 + " |"
        pieces = _split_table(block, max_chars=100)
        self.assertEqual(len(pieces), 1, "单行超长时整行保留，硬切会毁掉单元格")
        self.assertIn("x" * 500, pieces[0])

    def test_merge_short_combines_fragments(self):
        merged = _merge_short(["a" * 100, "b" * 100, "c" * 100, "d" * 100], min_chars=250)
        self.assertEqual(len(merged), 1)
        self.assertIn("a" * 100, merged[0])
        self.assertIn("d" * 100, merged[0])

    def test_split_long_overlap(self):
        long_text = " ".join(f"Sentence number {i} ends here." for i in range(200))
        pieces = _split_long(long_text, max_chars=800, overlap=100)
        self.assertGreater(len(pieces), 1)
        joined = " ".join(pieces)
        # 内容无丢失（去空白后逐句可还原）
        for i in (0, 50, 199):
            self.assertIn(f"Sentence number {i} ends here.", joined)


    def test_tables_stay_in_reading_order_and_short_prose_bridges_across_table(self):
        sec = (
            "# 5 Results\n\n> **所属论文**: Sample (s)\n\n---\n\n"
            + ("First paragraph before table one with plenty of context. " * 6) + "\n\n"
            + "Short lead-in before floating table:\n\n"
            + "| M | BLEU |\n| --- | --- |\n| A | 25.16 |\n\n"
            + "_Table 1._ First table caption.\n\n"
            + ("Middle paragraph continuation after table one. " * 6) + "\n\n"
            + "**Table 2** Second table caption without punctuation\n\n"
            + "| M | PPL |\n| --- | --- |\n| B | 40.46 |\n\n"
            + ("Final paragraph after table two wrapping up the section. " * 6) + "\n\n"
            + "###\n\n---\n\n### 本章子章节导引\n\n- **5.1 Sub** -> [查看详情](./05.1.md)\n"
        )
        (self.folder / "01_Introduction.md").write_text(sec, encoding="utf-8")
        chunks = [
            c for c in chunk_paper_folder(self.folder)
            if c["section_file"] == "01_Introduction.md"
        ]
        texts = [c["text"] for c in chunks]
        t1_idx = next(i for i, t in enumerate(texts) if "25.16" in t)
        t2_idx = next(i for i, t in enumerate(texts) if "40.46" in t)
        first_prose_idx = next(i for i, t in enumerate(texts) if "First paragraph" in t)
        mid_prose_idx = next(i for i, t in enumerate(texts) if "Middle paragraph" in t)
        final_prose_idx = next(i for i, t in enumerate(texts) if "Final paragraph" in t)

        self.assertLess(first_prose_idx, t1_idx)
        self.assertLess(t1_idx, mid_prose_idx)
        self.assertLess(mid_prose_idx, t2_idx)
        self.assertLess(t2_idx, final_prose_idx)
        # 短引子不能丢失，也不能被并进表格
        self.assertTrue(any("Short lead-in" in t and "|" not in t for t in texts))
        # 尾部子章节导引与空 ### 必须彻底剥离
        for t in texts:
            self.assertNotIn("本章子章节导引", t)
            self.assertNotIn("查看详情", t)
            self.assertNotIn("###", t)


    def test_reconstruct_table_repairs_split_headers_and_math_cells(self):
        from citetrace_pdf.pdf.table_reconstructor import reconstruct_table

        broken = (
            "||BL|EU|Training C|ost (FLOPs)|\n"
            "|---|---|---|---|---|\n"
            "|Model|||||\n"
            "||EN-DE|EN-FR|EN-DE|EN-FR|\n"
            "|ByteNet [18]|23.75||||\n"
            "|Deep-Att + PosUnk [39]||39.2||1_._0_·_10<sup>20</sup>|\n"
            "|<br>GNMT + RL [38]|24.6|39.92|2_._3_·_10<sup>19</sup>|1_._4_·_10<sup>20</sup>|\n"
            "|ConvS2S [9]|25.16|40.46|9_._6_·_10<sup>18</sup>|1_._5_·_10<sup>20</sup>|\n"
            "|Transformer (base model)|27.3|38.1|**3****_._3****_·_**|**10**<sup>**18**</sup>|\n"
            "|Transformer (big)|**28.4**|**41.8**|2_._3_·_|10<sup>19</sup>|"
        )
        rebuilt = reconstruct_table(broken)
        lines = rebuilt.splitlines()
        self.assertEqual(
            lines[0],
            "|Model|BLEU (EN-DE)|BLEU (EN-FR)|Training Cost (FLOPs) (EN-DE)|Training Cost (FLOPs) (EN-FR)|",
        )
        self.assertIn("|Deep-Att + PosUnk [39]||39.2||1.0 · 10^20|", rebuilt)
        self.assertIn("|GNMT + RL [38]|24.6|39.92|2.3 · 10^19|1.4 · 10^20|", rebuilt)
        self.assertIn("|Transformer (base model)|27.3|38.1|**3.3 · 10^18**||", rebuilt)
        self.assertIn("|Transformer (big)|**28.4**|**41.8**|2.3 · 10^19||", rebuilt)


if __name__ == "__main__":
    unittest.main()
