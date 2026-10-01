import json
import tempfile
import unittest
from pathlib import Path

import build_rq_llm_cases as builder


class BuildRqCasesTest(unittest.TestCase):
    def test_catalog_loader_accepts_wrapped_640_manifest(self):
        cases = builder.load(builder.DEFAULT_MANIFEST)
        self.assertEqual(len(cases), 640)

    def test_multiple_shapes_and_all_structures(self):
        cases = builder.load(builder.DEFAULT_MANIFEST)
        selected = builder.choose(cases, 3)
        structures = {row["structure"] for row in selected}
        self.assertEqual(structures, {"gqa", "sliding_attention", "sparse_attention", "mla",
                                      "swiglu", "moe", "mamba2", "linear_attention"})
        counts = {}
        for row in selected:
            key = (row["structure"], row["phase"])
            counts[key] = counts.get(key, 0) + 1
        self.assertTrue(all(1 <= value <= 3 for value in counts.values()))
        self.assertTrue(any(value >= 2 for value in counts.values()))

    def test_attention_tsv_has_mha_mqa_gqa_when_selected_from_full_catalog(self):
        selected = builder.choose_attention(builder.load(builder.DEFAULT_MANIFEST), 3)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.tsv"
            count = builder.write_attention_tsv(path, selected)
            text = path.read_text()
        self.assertGreater(count, 0)
        self.assertIn("\tMHA\t", text)
        self.assertIn("\tMQA\t", text)
        self.assertIn("\tGQA\t", text)

    def test_all_case_attention_subset_is_complete(self):
        cases = builder.load(builder.DEFAULT_MANIFEST)
        attention = [case for case in cases
                     if case["structure"] in {"gqa", "sliding_attention"}]
        self.assertEqual(len(attention), 240)
        with tempfile.TemporaryDirectory() as directory:
            count = builder.write_attention_tsv(Path(directory) / "all.tsv", attention)
        self.assertEqual(count, 240)


if __name__ == "__main__":
    unittest.main()
