import json
import unittest
from pathlib import Path
import pymupdf


from citetrace_pdf.model.document import Section
from citetrace_pdf.parser.auditor import OutlineAuditor
from citetrace_pdf.parser.inspector import CandidateHeading, PageInspector
from citetrace_pdf.parser.referee import LLMSkeletonReferee


class _FakeRefereeClient:
    def __init__(self, target_keyword: str):
        self.target_keyword = target_keyword

    def chat_completion(self, model, messages, temperature=0.0):
        user_msg = messages[-1]["content"]
        selected = []
        for line in user_msg.splitlines():
            if self.target_keyword in line and line.startswith("["):
                idx = int(line.split("]")[0].lstrip("["))
                selected.append(idx)
        return json.dumps({"selected_indices": selected})


class TestAgenticPDFPipeline(unittest.TestCase):

    def test_step1_outline_auditor_detects_gaps_and_pseudo_titles(self):
        broken = [
            Section("Abstract", 1, 1, "abs", 0, None),
            Section("1 Introduction", 1, 2, "intro", 1, "1"),
            Section("5. Output", 2, 6, "diagram", 1, "5"),
            Section("3 Experiments", 6, 7, "exp", 1, "3"),
        ]
        rep = OutlineAuditor.audit(broken, total_pages=9, paper_title="Sample Title")
        self.assertFalse(rep.passed)
        kinds = {i.kind for i in rep.issues}
        self.assertIn("suspicious_title", kinds)
        self.assertIn("missing_section", kinds)

    def test_step2_page_inspector_repairs_missing_section_on_real_fixture(self):
        pdf18 = Path("/root/citetrace/cache/papers/Attention_Is_All_You_Need/cited_papers/[18]/raw.pdf")
        if not pdf18.is_file():
            self.skipTest("[18]/raw.pdf fixture not present")
        with pymupdf.open(pdf18) as doc:
            broken_18 = [
                Section("Abstract", 1, 1, "abs", 0, None),
                Section("1. Introduction", 1, 2, "intro", 1, "1"),
                Section("2. Neural Translation Model", 2, 3, "m", 1, "2"),
                Section("3. ByteNet", 3, 4, "b", 1, "3"),
                Section("4. Model Comparison", 4, 5, "c", 1, "4"),
                Section(
                    "5. Character Prediction",
                    6,
                    6,
                    "The ByteNet decoder achieves 1.31 bits/character. **6. Character-Level Machine Translation** We evaluate the full ByteNet.",
                    1,
                    "5",
                ),
                Section("7. Conclusion", 7, 7, "conc", 1, "7"),
                Section("References", 7, 9, "refs", 0, None),
            ]
            rep_before = OutlineAuditor.audit(broken_18, len(doc), "Neural Machine Translation in Linear Time")
            self.assertFalse(rep_before.passed)
            fixed, rep_after = PageInspector.auto_repair(doc, broken_18, rep_before)
            self.assertTrue(rep_after.passed)
            titles = [s.title for s in fixed]
            self.assertIn("6. Character-Level Machine Translation", titles)

    def test_step3_llm_skeleton_referee_selects_and_splits(self):
        cands = [
            CandidateHeading(page=1, y0=100.0, font_size=12.0, is_bold=True, text="1 Introduction"),
            CandidateHeading(page=2, y0=200.0, font_size=10.0, is_bold=True, text="Input 1 0 1"),
            CandidateHeading(page=3, y0=150.0, font_size=12.0, is_bold=True, text="2 Active Memory Models"),
        ]
        client = _FakeRefereeClient("2 Active Memory Models")
        selected = LLMSkeletonReferee.select_headings_via_llm(cands, client)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].text, "2 Active Memory Models")




if __name__ == "__main__":
    unittest.main()
