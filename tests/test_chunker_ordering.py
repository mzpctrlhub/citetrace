"""Chunk 表格顺序屏障与元数据规范契约测试

核心规则：
1. 表格前短句（<80 字符）严禁跨越表格与表格后正文合并
2. 表格必须作为硬性顺序屏障：
   - 遇到表格前强制 flush 输出前面所有散文
   - 输出表格 chunk
   - 再输出表格后的散文
3. 每个 chunk 包含 sequence_index 严格单调自增
4. chunk 包含 content_type ("prose" 或 "table")
"""
import json
import tempfile
import unittest
from pathlib import Path

from citetrace_pdf.chunker import chunk_paper_folder


class TestChunkerOrdering(unittest.TestCase):
    """测试 Chunker 对表格作为顺序屏障及短句保留的处理"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self.tmp.name) / "paper_ordering"
        self.folder.mkdir(parents=True)

    def tearDown(self):
        self.tmp.cleanup()

    def test_short_intro_sentence_before_table_remains_before_table(self):
        """表格前不足 80 字符的短引导句必须严格位于表格之前输出，严禁与表后散文合并"""
        metadata = {
            "paper_id": "ordering_001",
            "paper_title": "Order Test Paper",
            "outline": [
                {"index": 0, "title": "1 Results", "file": "01_Results.md", "start_page": 1, "end_page": 1}
            ],
        }
        (self.folder / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        # 构造由三部分组成的章节：
        # 1. 表格前短句 (< 80 字符)
        # 2. Markdown 表格
        # 3. 表格后解释散文
        short_intro = "Table 1 lists the translation accuracy across benchmarks:"
        self.assertLess(len(short_intro), 80, "测试前置条件：短句长度必须小于 80 字符")

        table_md = (
            "| Model | BLEU |\n"
            "| --- | --- |\n"
            "| Base | 27.3 |\n"
            "| Big | 28.4 |"
        )
        post_table_explanation = (
            "The big model achieves a significant improvement over previous recurrent baselines. "
            "All hyper-parameters were tuned on the validation set using grid search."
        )

        section_content = f"# 1 Results\n\n{short_intro}\n\n{table_md}\n\n{post_table_explanation}\n"
        (self.folder / "01_Results.md").write_text(section_content, encoding="utf-8")

        chunks = chunk_paper_folder(self.folder)

        # 必须至少切出 3 个 chunk：
        # Chunk 0: 表格前短句 (prose)
        # Chunk 1: 表格 (table)
        # Chunk 2: 表格后散文 (prose)
        self.assertGreaterEqual(len(chunks), 3, f"预期至少切出 3 个独立 chunk，实际只有 {len(chunks)} 个: {chunks}")

        # 验证每个 chunk 的 content_type 和顺序
        chunk0 = chunks[0]
        chunk1 = chunks[1]
        chunk2 = chunks[2]

        # 断言 1: 第一个 chunk 必须包含 short_intro，且绝不能包含 post_table_explanation
        self.assertIn(short_intro, chunk0["text"], "第一个 chunk 必须包含表格前引导句！")
        self.assertNotIn(post_table_explanation, chunk0["text"], "表格前短句严禁跨越表格与表格后散文合并！")
        self.assertEqual(chunk0.get("content_type"), "prose")
        self.assertEqual(chunk0.get("sequence_index"), 0)

        # 断言 2: 第二个 chunk 必须是表格，且绝不能包含 short_intro
        self.assertTrue("|Model|BLEU|" in chunk1["text"] or "| Model | BLEU |" in chunk1["text"])
        self.assertNotIn(short_intro, chunk1["text"], "表格块内不得包含表前散文！")
        self.assertEqual(chunk1.get("content_type"), "table")
        self.assertEqual(chunk1.get("sequence_index"), 1)

        # 断言 3: 第三个 chunk 必须是表后散文
        self.assertIn("The big model achieves a significant improvement", chunk2["text"])
        self.assertEqual(chunk2.get("content_type"), "prose")
        self.assertEqual(chunk2.get("sequence_index"), 2)


if __name__ == "__main__":
    unittest.main()
