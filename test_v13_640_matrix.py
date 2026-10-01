import json
import subprocess
import tempfile
import unittest
from pathlib import Path

import analyze_v13_640_matrix as matrix
import audit_v10_v13_completion as completion


HERE = Path(__file__).resolve().parent


class V13640MatrixTests(unittest.TestCase):
    def test_every_rq_declares_case_scope(self):
        registry = json.loads(matrix.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        self.assertEqual(len(registry["rqs"]), 10)
        for rq in registry["rqs"]:
            self.assertIn("case_scope", rq)
            self.assertTrue(rq["case_scope"]["reason"])

    def test_empty_artifacts_still_materialize_full_cartesian_product(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            subprocess.run([
                "python3", str(HERE / "analyze_v13_640_matrix.py"),
                "--raw-dir", str(root / "raw"), "--output-dir", str(root / "out")
            ], check=True, capture_output=True, text=True)
            summary = json.loads((root / "out/v13_640_matrix_summary.json").read_text())
            self.assertEqual(summary["case_count"], 640)
            self.assertEqual(summary["schema_version"], 2)
            self.assertEqual(summary["matrix_cell_count"], 640 * 6 * 10)
            self.assertFalse(summary["all_applicable_cells_natively_validated"])
            rows = (root / "out/v13_640_case_rq_matrix.jsonl").read_text().splitlines()
            self.assertEqual(len(rows), 640 * 10)

    def test_exact_pair_is_required_for_native_validation(self):
        self.assertTrue(matrix.success({"status": "success", "correct": True}))
        self.assertEqual(matrix.candidate({"layout": "HND"}), "HND")
        self.assertIsNone(matrix.candidate({"p50_ms": 1.0}))

    def test_cutlass_candidates_are_layouts_not_incidental_tiles(self):
        rows = [{"flat_ms": "2.0", "tiled_direct_ms": "1.0",
                 "tile_rows": "16", "tile_cols": "16"}]
        self.assertEqual(matrix.native_candidates("CUTLASS/CuTe", rows),
                         ["flat_row_major", "tiled_direct"])
        self.assertEqual(matrix.evidence_granularity("CUTLASS/CuTe", rows),
                         "projected_attention_score_boundary")

    def test_tvm_manifest_mapping_is_explicitly_projected(self):
        rows = [{"manifest_case_id": "c", "strategy": "row_major"}]
        self.assertEqual(matrix.evidence_granularity("TVM", rows),
                         "projected_boundary_shape")

    def test_completion_audit_recognizes_legacy_noop_alternate(self):
        legacy_noop = {"strategy": "alternate_axis12_strided",
                       "status": "success", "input_contiguous": [True]}
        legacy_changed = {"strategy": "alternate_axis12_strided",
                          "status": "success", "input_contiguous": [False]}
        current_changed = {"strategy": "alternate_strided_view",
                           "status": "success", "layout_changed": True}
        self.assertTrue(completion.is_boundary_alternate(legacy_noop))
        self.assertFalse(completion.boundary_layout_changed(legacy_noop))
        self.assertTrue(completion.boundary_layout_changed(legacy_changed))
        self.assertTrue(completion.boundary_layout_changed(current_changed))


if __name__ == "__main__":
    unittest.main()
