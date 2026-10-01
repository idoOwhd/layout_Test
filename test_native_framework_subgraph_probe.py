import json
import tempfile
import unittest
from pathlib import Path

import analyze_native_subgraph_completion as completion
import native_framework_subgraph_probe as probe
import native_kv_request_batch_probe as batch_probe


class NativeFrameworkSubgraphProbeTests(unittest.TestCase):
    def test_attention_dimensions_use_decode_append_row(self):
        case = {"phase": "decode", "structure": "gqa",
                "shape": {"query_length": 1, "num_kv_heads": 8, "head_dim": 128},
                "dimensions": {}}
        self.assertEqual(probe.attention_dimensions(case), (1, 8, 128, 128))

    def test_native_slice_flattens_request_batch_but_batch_probe_does_not(self):
        case = {"phase": "decode", "structure": "gqa",
                "shape": {"batch": 4, "query_length": 1,
                          "num_kv_heads": 8, "head_dim": 128},
                "dimensions": {}}
        self.assertEqual(probe.attention_dimensions(case), (1, 8, 128, 128))
        self.assertEqual(probe.attention_dimensions(case, include_batch=True),
                         (4, 8, 128, 128))

    def test_sparse_uses_index_cache_width(self):
        case = {"phase": "prefill", "structure": "sparse_attention",
                "shape": {"query_length": 64, "index_dim": 96, "value_head_dim": 64},
                "dimensions": {}}
        self.assertEqual(probe.attention_dimensions(case), (64, 1, 96, 64))

    def test_declared_structures_are_complete(self):
        self.assertEqual(probe.ATTENTION,
                         {"gqa", "sliding_attention", "sparse_attention", "mla"})

    def test_request_major_slots(self):
        self.assertEqual(batch_probe.slot_mapping(2, 3, "request_major_contiguous"),
                         [0, 1, 2, 3, 4, 5])

    def test_block_interleaved_slots_cover_distinct_pages(self):
        self.assertEqual(batch_probe.slot_mapping(2, 3, "block_interleaved"),
                         [0, 1, 2, 16, 17, 18])
        self.assertEqual(batch_probe.slot_mapping(4, 1, "block_interleaved"),
                         [0, 16, 32, 48])

    def test_batch_memory_estimate_increases(self):
        one = batch_probe.estimated_bytes(1, 1, 8, 128, 128)
        four = batch_probe.estimated_bytes(4, 49, 8, 128, 128)
        self.assertGreater(four, one)

    def test_balanced_batch_probe_selects_both_phases(self):
        cases = [
            {"case_id": "g-p0", "structure": "gqa", "phase": "prefill"},
            {"case_id": "g-p1", "structure": "gqa", "phase": "prefill"},
            {"case_id": "g-d", "structure": "gqa", "phase": "decode"},
            {"case_id": "m-p", "structure": "mla", "phase": "prefill"},
            {"case_id": "m-d", "structure": "mla", "phase": "decode"},
        ]
        selected = batch_probe.balanced_cases(cases, 4)
        self.assertEqual({(case["structure"], case["phase"]) for case in selected},
                         {("gqa", "prefill"), ("gqa", "decode"),
                          ("mla", "prefill"), ("mla", "decode")})

    def test_analyzer_accounts_for_missing_pairs(self):
        case = {"case_id": "c", "structure": "swiglu", "phase": "prefill",
                "target_rqs": ["L-RQ1"]}
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = root / "m.json"
            manifest.write_text(json.dumps({"cases": [case]}))
            (root / "raw").mkdir()
            old = __import__("sys").argv
            try:
                __import__("sys").argv = ["x", "--manifest", str(manifest),
                                           "--result-dir", str(root)]
                self.assertEqual(completion.main(), 0)
            finally:
                __import__("sys").argv = old
            data = json.loads((root / "NATIVE_SUBGRAPH_COVERAGE.json").read_text())
            self.assertEqual(data["expected_framework_case_pairs"], 2)
            self.assertTrue(all(row["status"] == "missing" for row in data["accounting"]))

    def test_empty_filtered_partition_is_not_a_failure(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            manifest = root / "m.json"
            output = root / "out.jsonl"
            manifest.write_text(json.dumps({"cases": [{
                "case_id": "m-p", "structure": "mamba2", "phase": "prefill",
                "shape": {}, "dimensions": {}
            }]}))
            old = __import__("sys").argv
            try:
                __import__("sys").argv = ["x", "--framework", "vllm",
                                             "--manifest", str(manifest),
                                             "--output", str(output),
                                             "--structure", "mamba2",
                                             "--phase", "decode"]
                self.assertEqual(probe.main(), 0)
                self.assertEqual(output.read_text(), "")
            finally:
                __import__("sys").argv = old


if __name__ == "__main__":
    unittest.main()
