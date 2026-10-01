import json
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path


HERE = Path(__file__).resolve().parent


class PersuasiveRealWorldCasesTest(unittest.TestCase):
    def test_builder_emits_128_new_balanced_contracts(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases"
            subprocess.run([
                sys.executable, str(HERE / "build_persuasive_real_world_cases.py"),
                "--output-dir", str(output),
            ], check=True, capture_output=True, text=True)
            value = json.loads((output / "persuasive_real_world_128.json").read_text())
            rows = value["cases"]
            self.assertEqual(len(rows), 128)
            self.assertEqual(len({row["tensor_spec_sha256"] for row in rows}), 128)
            self.assertEqual(set(Counter(row["structure"] for row in rows).values()), {16})
            self.assertTrue(all(row["source_url"].startswith("https://huggingface.co/")
                                for row in rows))
            self.assertTrue(all(row["parent_case_id"] for row in rows))
            summary = json.loads((output / "PERSUASIVE_CASES_SUMMARY.json").read_text())
            self.assertEqual(summary["overlap_with_original_640_tensor_contracts"], 0)

    def test_diversity_v2_is_balanced_over_structure_phase_and_batch(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "cases"
            subprocess.run([
                sys.executable, str(HERE / "build_llm_shape_diversity_v2.py"),
                "--output-dir", str(output),
            ], check=True, capture_output=True, text=True)
            value = json.loads((output / "llm_shape_diversity_256.json").read_text())
            rows = value["cases"]
            self.assertEqual(len(rows), 256)
            self.assertEqual(len({row["tensor_spec_sha256"] for row in rows}), 256)
            self.assertEqual(set(Counter(row["structure"] for row in rows).values()), {32})
            self.assertEqual(Counter(row["phase"] for row in rows),
                             {"prefill": 128, "decode": 128})
            self.assertEqual(Counter(row["shape"]["batch"] for row in rows),
                             {1: 64, 2: 64, 4: 64, 8: 64})
            self.assertTrue(all(row["single_gpu_blocked_rqs"] == ["L-RQ5"]
                                for row in rows))
            self.assertTrue(all(row["shape_counterfactual"] for row in rows))
            smoke = json.loads((output / "llm_shape_diversity_smoke_16.json").read_text())["cases"]
            self.assertEqual(len(smoke), 16)
            self.assertEqual(Counter((row["phase"], row["shape"]["batch"])
                                     for row in smoke),
                             {("prefill", 2): 8, ("decode", 4): 8})


if __name__ == "__main__":
    unittest.main()
