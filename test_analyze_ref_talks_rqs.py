import tempfile
import unittest
from pathlib import Path

from analyze_ref_talks_rqs import analyze, framework_matrix, summarize_boundary


def row(case, benchmark, strategy, ms, *, reuse=0, layout=None):
    return {
        "framework": "cuda-reference", "benchmark": benchmark,
        "case_id": case, "subgraph": "gqa", "phase": "decode",
        "strategy": strategy, "layout": layout or strategy,
        "tokens": 1024.0, "kv_heads": 8.0, "query_heads": 32.0,
        "head_dim": 128.0, "page_size": 0.0, "reuse_count": float(reuse),
        "bytes": float(1024 * 8 * 128 * 2),
        "p20_ms": ms * 0.96, "p50_ms": ms, "p80_ms": ms * 1.04,
        "max_abs_error": 0.0, "model": "synthetic", "attention_kind": "GQA",
        "evidence_level": "runtime_empirical",
    }


def case_rows(case, *, split_wins):
    rows = []
    for benchmark, values in {
        "kv_write": {"NHD": 1.0, "HND": 2.0},
        "decode_head_scan": {"NHD": 4.0, "HND": 1.0},
        "token_major_scan": {"NHD": 1.0, "HND": 4.0},
        "layout_conversion": {"NHD_to_HND": 2.0, "HND_to_NHD": 2.2},
        "decode_under_hbm_contention": {"NHD": 0.8, "HND": 2.0},
        "rope_kv_pipeline": {
            "rope_then_convert": 4.0, "convert_then_rope": 5.0,
            "fused_rope_native_HND": 2.0,
        },
    }.items():
        for strategy, ms in values.items():
            rows.append(row(case, benchmark, strategy, ms))
    multi = ({"common_NHD": 6.0, "common_HND": 7.0,
              "split_NHD_HND_with_conversion": 4.0} if split_wins else
             {"common_NHD": 5.0, "common_HND": 6.0,
              "split_NHD_HND_with_conversion": 7.0})
    for strategy, ms in multi.items():
        rows.append(row(case, "multi_consumer_pipeline", strategy, ms))
    for reuse, values in {
        1: {"common_NHD": 5.0, "producer_native_HND": 7.0,
            "NHD_then_convert_once_HND": 8.0},
        64: {"common_NHD": 100.0, "producer_native_HND": 60.0,
             "NHD_then_convert_once_HND": 62.0},
    }.items():
        for strategy, ms in values.items():
            rows.append(row(case, "conversion_reuse_pipeline", strategy, ms, reuse=reuse))
    return rows


class AnalyzerTest(unittest.TestCase):
    def test_all_rqs_get_falsifiable_records(self):
        results = analyze(case_rows("shape_a", split_wins=True) +
                          case_rows("shape_b", split_wins=False))
        self.assertEqual([f"RQ{i}" for i in range(1, 11)], [r["rq"] for r in results])
        by_rq = {r["rq"]: r for r in results}
        self.assertEqual("supported", by_rq["RQ1"]["problem_status"])
        self.assertEqual("supported", by_rq["RQ2"]["solution_status"])
        self.assertEqual("supported", by_rq["RQ3"]["problem_status"])
        self.assertEqual("supported", by_rq["RQ8"]["solution_status"])
        self.assertEqual("supported", by_rq["RQ10"]["solution_status"])

    def test_absent_gpu_data_is_not_run(self):
        results = analyze([])
        self.assertTrue(all(row["problem_status"] == "not_run" for row in results))

    def test_source_and_runtime_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "tvm_layout.csv").write_text("p50_ms\n1.0\n", encoding="utf-8")
            audit = {"rq_coverage": {"RQ1": {"frameworks": ["vLLM", "TVM"]}}}
            matrix = {row["framework"]: row for row in framework_matrix(root, audit)}
            self.assertTrue(matrix["vLLM"]["source_static"])
            self.assertFalse(matrix["vLLM"]["runtime_artifacts"])
            self.assertTrue(matrix["TVM"]["source_static"])
            self.assertEqual(["tvm_layout.csv"], matrix["TVM"]["runtime_artifacts"])

    def test_boundary_summary_keeps_subgraph_and_repair_evidence(self):
        rows = []
        for strategy, ms in {
            "native_contiguous": 2.0,
            "alternate_axis12_strided": 1.0,
            "repair_to_contiguous_each_call": 3.0,
        }.items():
            rows.append({"case_id": "moe_shape_a", "subgraph": "moe", "phase": "decode",
                         "shape": {"tokens": 8, "hidden": 4096}, "status": "success",
                         "strategy": strategy, "p20_ms": ms * .96,
                         "p50_ms": ms, "p80_ms": ms * 1.04})
        result = summarize_boundary(rows)
        self.assertEqual("measured", result["status"])
        self.assertEqual(["moe"], result["subgraphs"])
        self.assertEqual(["moe_shape_a"], result["alternate_layout_wins"])
        self.assertEqual(3.0, result["max_repair_slowdown"])


if __name__ == "__main__":
    unittest.main()
