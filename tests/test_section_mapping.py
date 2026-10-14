"""章节页码回退映射 (_find_section_for_page) 修复验证"""
import unittest

from parsers.folder_extractor import _find_section_for_page

OUTLINE = [
    {"index": 0, "title": "Abstract", "file": "00_Abstract.md", "start_page": 1, "end_page": 1},
    {"index": 1, "title": "1 Introduction", "file": "01_Introduction.md", "start_page": 2, "end_page": 2},
    {"index": 2, "title": "2 Background", "file": "02_Background.md", "start_page": 2, "end_page": 3},
    {"index": 3, "title": "3 Model", "file": "03_Model.md", "start_page": 3, "end_page": 6},
    {"index": 4, "title": "References", "file": "References.md", "start_page": 14, "end_page": 15},
]


class TestFindSectionForPage(unittest.TestCase):

    def test_exact_page_hit(self):
        title, fname = _find_section_for_page(1, OUTLINE)
        self.assertEqual(fname, "00_Abstract.md")
        self.assertEqual(title, "Abstract")

    def test_page_with_multiple_sections_prefers_latest_start(self):
        # 第 2 页同时有 Introduction 与 Background，应取 start_page 更大的 Background
        _, fname = _find_section_for_page(2, OUTLINE)
        self.assertEqual(fname, "02_Background.md")

    def test_range_interior_page(self):
        _, fname = _find_section_for_page(5, OUTLINE)
        self.assertEqual(fname, "03_Model.md")

    def test_references_never_returned_as_body_section(self):
        # References 章节必须被跳过，页码 15 不得映射到 References.md
        title, fname = _find_section_for_page(15, OUTLINE)
        self.assertEqual(fname, "main.md")
        self.assertEqual(title, "Main Content")

    def test_out_of_range_falls_back(self):
        _, fname = _find_section_for_page(99, OUTLINE)
        self.assertEqual(fname, "main.md")

    def test_empty_outline(self):
        _, fname = _find_section_for_page(1, [])
        self.assertEqual(fname, "main.md")


if __name__ == "__main__":
    unittest.main()
