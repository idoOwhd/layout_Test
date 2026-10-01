from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

import build_rq_adequacy_suite as suite


HERE = Path(__file__).resolve().parent


class RQAdequacySuiteTests(unittest.TestCase):
    def test_profiles_are_balanced_and_boundary_rich(self):
        values = suite.profiles()
        self.assertEqual(len(values), 64)
        self.assertEqual(Counter(value[0] for value in values),
                         {"prefill": 32, "decode": 32})
        self.assertEqual(Counter(value[2] for value in values),
                         {1: 16, 2: 16, 4: 16, 8: 16})
        lengths = {value[1] for value in values}
        self.assertTrue({15, 16, 17, 31, 32, 33, 127, 128, 129} <= lengths)

    def test_generated_cases_keep_real_world_parent_provenance(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "cases"
            subprocess.run(
                [sys.executable, str(HERE / "build_rq_adequacy_suite.py"),
                 "--output-dir", str(output)], check=True, capture_output=True, text=True)
            value = json.loads((output / "rq_adequacy_1024.json").read_text())
            cases = value["cases"]
            self.assertEqual(len(cases), 1024)
            self.assertEqual(len({case["case_id"] for case in cases}), 1024)
            self.assertEqual(Counter(case["structure"] for case in cases),
                             {structure: 128 for structure in suite.STRUCTURES})
            self.assertEqual(Counter(case["phase"] for case in cases),
                             {"prefill": 512, "decode": 512})
            self.assertEqual(Counter(case["shape"]["batch"] for case in cases),
                             {1: 256, 2: 256, 4: 256, 8: 256})
            for case in cases:
                self.assertTrue(case["parent_case_id"].startswith("prefill-"))
                self.assertTrue(case["model_id"])
                self.assertTrue(case["model_revision"])
                self.assertTrue(case["source_url"].startswith("https://"))
                self.assertTrue(case["shape_counterfactual"])
                self.assertIn("pinned real model dimensions", case["shape_source"])

    def test_cuda_source_contains_direct_missing_interventions(self):
        source = (HERE / "rq_llm_layout_bench.cu").read_text()
        for marker in (
            "weighted_multi_consumer_pipeline",
            "state_migration_trace",
            "decode_under_hbm_contention",
            "allocated_tokens",
            "seeded_markov",
            "--order-seed",
        ):
            self.assertIn(marker, source)
        self.assertNotIn("if (c.tokens % page_size) continue", source)


if __name__ == "__main__":
    unittest.main()
