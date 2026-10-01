import json
import tempfile
import unittest
from pathlib import Path

from extract_native_observation_results import cutlass, sglang, triton, tvm, vllm
from triton_kv_layout_bench import tsv_cases
from validate_layout_observations import (
    DEFAULT_CATALOG,
    native_for,
    performance_for,
    source_for,
)


class LayoutObservationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = json.loads(DEFAULT_CATALOG.read_text(encoding="utf-8"))["observations"]

    def observation(self, observation_id):
        return next(row for row in self.catalog if row["id"] == observation_id)

    def test_catalog_covers_v10_and_canonical_rq1_to_rq10(self):
        rqs = {row["rq"] for row in self.catalog}
        self.assertIn("L-RQ1-v10", rqs)
        self.assertTrue({f"RQ{i}" for i in range(1, 11)}.issubset(rqs))
        self.assertEqual(len(self.catalog), len({row["id"] for row in self.catalog}))

    def test_source_majority_uses_applicable_denominator(self):
        observation = self.observation("O-RQ1-TWO-LEVEL-CONTROL")
        audit = {"rq_coverage": {"RQ1": {"frameworks": [
            "vLLM", "SGLang", "TensorRT-LLM"]}}}
        result = source_for(observation, audit, {})
        self.assertEqual("supported", result["majority_status"])
        self.assertEqual(3, result["support_count"])
        self.assertEqual(4, len(result["applicable_frameworks"]))

    def test_v10_does_not_inherit_all_canonical_rq1_frameworks(self):
        observation = self.observation("O-LRQ1-EDGE-INVERSION")
        audit = {"rq_coverage": {"RQ1": {"frameworks": [
            "vLLM", "SGLang", "TensorRT-LLM", "FlashInfer"]}}}
        result = source_for(observation, audit, {})
        self.assertEqual("not_supported", result["majority_status"])
        self.assertEqual(["FlashInfer", "vLLM"], result["supporting_frameworks"])

    def test_source_vote_is_not_runtime_performance(self):
        general = {"rqs": [{"rq": "RQ3", "problem_status": "supported",
                             "solution_status": "not_run", "metrics": {},
                             "evidence_levels": ["source_static"]}]}
        result = performance_for("RQ3", general, {})
        self.assertEqual("not_run", result["solution_status"])
        self.assertIsNone(result["max_speedup"])

    def test_v10_result_key_reads_strict_hypotheses(self):
        v10 = {"hypotheses": {
            "H1.1": {"status": "supported"},
            "edge_oracle_solution": {"status": "supported", "max_speedup": 1.2},
        }, "cases": [{"case_id": "x"}]}
        result = performance_for("v10", {}, v10)
        self.assertEqual("supported", result["problem_status"])
        self.assertEqual("supported", result["solution_status"])
        self.assertEqual(1.2, result["max_speedup"])

    def test_native_majority_requires_speedup_in_distinct_frameworks(self):
        observation = self.observation("O-RQ8-NONCOMMUTATIVE-SEARCH")
        rows = [{"observation_id": observation["id"], "framework": framework,
                 "status": "supported", "speedup": 1.1}
                for framework in ("CUTLASS", "Triton", "TVM")]
        result = native_for(observation, rows)
        self.assertEqual("supported", result["status"])
        self.assertEqual(3, result["support_count"])

    def test_native_effect_must_exceed_one_percent(self):
        observation = self.observation("O-RQ8-NONCOMMUTATIVE-SEARCH")
        rows = [{"observation_id": observation["id"], "framework": framework,
                 "status": "supported", "speedup": 1.005}
                for framework in ("CUTLASS", "Triton", "TVM")]
        self.assertEqual("not_run", native_for(observation, rows)["status"])

    def test_problem_only_native_row_does_not_count_as_solution(self):
        observation = self.observation("O-RQ6-EPOCHAL-STATE")
        rows = [{"observation_id": observation["id"], "framework": "vLLM",
                 "status": "supported", "speedup": 1.2, "claim_kind": "problem"}]
        result = native_for(observation, rows)
        self.assertEqual("not_run", result["status"])
        self.assertEqual(["vLLM"], result["problem_frameworks"])

    def test_native_extractor_requires_correct_paired_counterfactual(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cutlass_softmax_boundary.csv").write_text(
                "correct,winner,winner_speedup\n1,transform_plus_flat,1.20\n",
                encoding="utf-8")
            result = cutlass(root)
            self.assertEqual("supported", result[0]["status"])
            self.assertEqual("O-RQ3-CONVERSION-INVESTMENT", result[0]["observation_id"])

    def test_cutlass_direct_winner_does_not_prove_conversion_investment(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "cutlass_softmax_boundary.csv").write_text(
                "correct,winner,winner_speedup\n1,tiled_direct,2.0\n", encoding="utf-8")
            self.assertEqual([], cutlass(root))

    def test_triton_portability_observation_needs_winner_heterogeneity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "triton_kv_layout.csv").write_text(
                "case_id,layout,p50_ms,correct\n"
                "a,NHD,1,True\na,HND,2,True\n"
                "b,NHD,2,True\nb,HND,1,True\n", encoding="utf-8")
            result = triton(root)
            self.assertEqual("supported", result[0]["status"])

    def test_vllm_two_level_control_needs_shape_dependent_winners(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "a", "layout": "LBNHC", "p50_ms": 1},
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "a", "layout": "LBHNC", "p50_ms": 2},
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "b", "layout": "LBNHC", "p50_ms": 1},
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "b", "layout": "LBHNC", "p50_ms": 3},
            ]
            (root / "vllm_native_serving.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = vllm(root)
            rq1 = next(row for row in result if row["observation_id"] == "O-RQ1-TWO-LEVEL-CONTROL")
            self.assertEqual("inconclusive", rq1["status"])

    def test_sglang_uses_only_paired_nhd_hnd_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "a", "layout": "NHD", "p50_ms": 1},
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "a", "layout": "HND", "p50_ms": 2},
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "b", "layout": "NHD", "p50_ms": 3},
                {"status": "success", "comparison_scope": "layout_only",
                 "case_id": "b", "layout": "HND", "p50_ms": 1},
                {"status": "success", "comparison_scope": "serving_policy_confounded",
                 "case_id": "a", "layout": "NHD", "p50_ms": .01},
            ]
            (root / "sglang_native_serving.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = sglang(root)
            rq1 = next(row for row in result if row["observation_id"] == "O-RQ1-TWO-LEVEL-CONTROL")
            self.assertEqual("supported", rq1["status"])
            self.assertEqual(3.0, rq1["speedup"])

    def test_tvm_extractor_separates_performance_and_feasibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                "case_id,experiment,strategy,reuse,p50_ms,correct",
                "a,consumer_scan,row_major,,1,True",
                "a,consumer_scan,tiled_16x16,,2,True",
                "b,consumer_scan,row_major,,3,True",
                "b,consumer_scan,tiled_16x16,,1,True",
                "a,conversion_reuse,keep_row_major,1,1,True",
                "a,conversion_reuse,convert_once_then_tiled,1,2,True",
                "a,conversion_reuse,convert_each_use_then_tiled,1,2,True",
                "a,conversion_reuse,keep_row_major,16,16,True",
                "a,conversion_reuse,convert_once_then_tiled,16,8,True",
                "a,conversion_reuse,convert_each_use_then_tiled,16,24,True",
                "a,fusion_order,fused_square_reduce,,1,True",
                "a,fusion_order,square_then_reduce_materialized,,2,True",
                "a,fusion_order,convert_then_square_then_tiled_reduce,,3,True",
                "b,fusion_order,fused_square_reduce,,3,True",
                "b,fusion_order,square_then_reduce_materialized,,2,True",
                "b,fusion_order,convert_then_square_then_tiled_reduce,,1,True",
            ]
            (root / "tvm_rq_observations.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
            result = tvm(root)
            by_id = {row["observation_id"]: row for row in result}
            self.assertEqual("supported", by_id["O-RQ3-CONVERSION-INVESTMENT"]["status"])
            self.assertEqual("supported", by_id["O-RQ8-NONCOMMUTATIVE-SEARCH"]["status"])
            self.assertEqual("supported", by_id["O-RQ9-RESIDUAL-METAPOLICY"]["status"])
            self.assertEqual("supported", by_id["O-RQ10-BOUNDED-REPAIR"]["status"])
            self.assertEqual("feasibility_supported", by_id["O-RQ7-TYPED-EDGE-CONTRACT"]["status"])

    def test_triton_sweep_reads_catalog_selected_attention_shapes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.tsv"
            path.write_text(
                "case_id\tmodel\tphase\tsubgraph\tattention_kind\ttokens\tquery_heads\tkv_heads\thead_dim\n"
                "mqa\tm@c\tdecode\tgqa\tMQA\t4096\t32\t1\t128\n"
                "gqa\tm@c\tprefill\tsliding_attention\tGQA\t2048\t32\t8\t128\n",
                encoding="utf-8")
            cases = tsv_cases(path)
            self.assertEqual(2, len(cases))
            self.assertEqual("MQA", cases[0]["attention_kind"])
            self.assertEqual(4, len(cases[0]["layouts"]))


if __name__ == "__main__":
    unittest.main()
