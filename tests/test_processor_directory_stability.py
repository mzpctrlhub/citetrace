"""测试 PDFProcessor 通用 paper ID 目录稳定性 (Step 8)

规则验证：
1. 传入通用或纯编号 paper_id（如 "1706.03762"、"raw"、"[17]"）时，
   产物目录必须严格稳定为该 ID，绝不得自动重命名/迁移至基于论文标题的目录。
2. 缓存目录中不得留下孤儿目录或冗余拷贝的 raw.pdf。
3. 返回的 metadata["folder"] 与 metadata["paper_id"] 必须与请求 ID 保持一致。
4. process_local_pdf 传入明确 paper_id 时，目录同样保持稳定，不被提取标题篡改。
"""
import shutil
import tempfile
import unittest
from pathlib import Path

from citetrace_pdf import PDFProcessor


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class TestProcessorDirectoryStability(unittest.TestCase):
    """测试 PDFProcessor 处理通用 paper_id 时的目录严格稳定性"""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cache_dir = self.root / "papers"
        self.cache_dir.mkdir(parents=True)

        self.sample_pdf = PROJECT_ROOT / "cache" / "papers" / "Attention_Is_All_You_Need" / "raw.pdf"
        if not self.sample_pdf.is_file():
            self.skipTest(f"基准 PDF 不存在: {self.sample_pdf}")

    def tearDown(self):
        self.tmp.cleanup()

    def test_process_paper_preserves_generic_arxiv_id_directory(self):
        """通用 arXiv 编号 (如 1706.03762) 绝不得自动改名为论文标题目录"""
        processor = PDFProcessor(cache_dir=self.cache_dir)
        meta = processor.process_paper(
            paper_id="1706.03762",
            pdf_url=str(self.sample_pdf),
            force=True,
        )

        expected_folder = self.cache_dir / "1706.03762"
        self.assertEqual(meta.get("folder"), str(expected_folder))
        self.assertTrue((expected_folder / "metadata.json").is_file())
        self.assertTrue((expected_folder / "chunks.json").is_file())
        self.assertTrue((expected_folder / "raw.pdf").is_file())

        # 绝不得生成孤儿或次生标题目录 (如 Attention_Is_All_You_Need)
        created_dirs = [p.name for p in self.cache_dir.iterdir() if p.is_dir()]
        self.assertEqual(created_dirs, ["1706.03762"], f"出现了非预期的标题改名目录: {created_dirs}")

    def test_process_paper_preserves_serial_directory(self):
        """引用序号标识 (如 [17]) 必须稳定落地在 [17] 目录下"""
        processor = PDFProcessor(cache_dir=self.cache_dir)
        meta = processor.process_paper(
            paper_id="[17]",
            pdf_url=str(self.sample_pdf),
            force=True,
        )

        expected_folder = self.cache_dir / "[17]"
        self.assertEqual(meta.get("folder"), str(expected_folder))
        self.assertTrue((expected_folder / "metadata.json").is_file())
        self.assertTrue((expected_folder / "chunks.json").is_file())
        self.assertTrue((expected_folder / "raw.pdf").is_file())

        created_dirs = [p.name for p in self.cache_dir.iterdir() if p.is_dir()]
        self.assertEqual(created_dirs, ["[17]"], f"出现了非预期的额外目录: {created_dirs}")

    def test_process_local_pdf_preserves_explicit_paper_id(self):
        """process_local_pdf 显式指定 paper_id 时，绝不得被提取标题覆盖"""
        processor = PDFProcessor(cache_dir=self.cache_dir)
        meta = processor.process_local_pdf(
            pdf_path=self.sample_pdf,
            paper_id="1706.03762",
            force=True,
        )

        expected_folder = self.cache_dir / "1706.03762"
        self.assertEqual(meta.get("folder"), str(expected_folder))
        self.assertTrue((expected_folder / "metadata.json").is_file())
        created_dirs = [p.name for p in self.cache_dir.iterdir() if p.is_dir()]
        self.assertEqual(created_dirs, ["1706.03762"], f"出现了非预期的额外目录: {created_dirs}")


if __name__ == "__main__":
    unittest.main()
