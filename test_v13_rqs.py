import json
from pathlib import Path
import tempfile
import unittest

import analyze_v13_rqs as adjudication
import analyze_v13_source_chain as source


HERE = Path(__file__).resolve().parent


class V13CoverageTests(unittest.TestCase):
    def test_registry_exactly_covers_canonical_hypotheses(self):
        canonical = source.canonical_hypotheses(source.DEFAULT_SPEC)
        registry = json.loads(source.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        registered = {hypothesis: spec
                      for rq in registry["rqs"]
                      for hypothesis, spec in rq["hypotheses"].items()}
        self.assertEqual(set(canonical), set(registered))
        self.assertEqual(len(canonical), 49)
        for hypothesis in canonical:
            self.assertEqual(canonical[hypothesis]["experiment"],
                             registered[hypothesis]["experiment"])

    def test_single_gpu_blockers_are_explicit(self):
        registry = json.loads(source.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        rqs = {row["rq"]: row for row in registry["rqs"]}
        self.assertEqual(rqs["L-RQ5"]["single_gpu_status"], "blocked_single_gpu")
        self.assertTrue(all(spec["status"] == "blocked_single_gpu"
                            for spec in rqs["L-RQ5"]["hypotheses"].values()))
        self.assertEqual(rqs["L-RQ4"]["hypotheses"]["H4.4"]["status"],
                         "blocked_requires_multiple_hardware")

    def test_all_requested_frameworks_have_fail_closed_adapters(self):
        registry = json.loads(source.DEFAULT_REGISTRY.read_text(encoding="utf-8"))
        expected = {"vLLM", "SGLang", "CUTLASS/CuTe", "Triton", "TVM", "Hexcute"}
        self.assertEqual(set(registry["framework_adapters"]), expected)
        all_rqs = {f"L-RQ{index}" for index in range(1, 11)}
        for spec in registry["framework_adapters"].values():
            classified = set(spec.get("runtime_rqs", [])) | set(spec.get("source_only_rqs", [])) | \
                set(spec.get("not_applicable_rqs", [])) | set(spec.get("missing_adapter_rqs", [])) | \
                set(spec.get("blocked_rqs", [])) | set(spec.get("architecture_blocked_rqs", []))
            self.assertEqual(classified, all_rqs)

    def test_source_chain_has_required_columns(self):
        import csv
        with source.DEFAULT_CHAIN.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            required = {"layout_rq", "stack", "evidence_role", "source_rule_fact",
                        "proxy", "ignored_variables", "fallback_repair",
                        "modern_trigger_family", "trigger_condition", "hypothesis_links"}
            self.assertTrue(required.issubset(reader.fieldnames or []))
            self.assertGreater(len(list(reader)), 200)

    def test_serving_winner_crossovers_require_three_percent_margin(self):
        rows = [
            {"status": "success", "comparison_scope": "page_size_only",
             "case_id": "short", "variant": "p1", "page_size": 1, "p50_ms": 100},
            {"status": "success", "comparison_scope": "page_size_only",
             "case_id": "short", "variant": "p16", "page_size": 16, "p50_ms": 102},
            {"status": "success", "comparison_scope": "page_size_only",
             "case_id": "long", "variant": "p1", "page_size": 1, "p50_ms": 102},
            {"status": "success", "comparison_scope": "page_size_only",
             "case_id": "long", "variant": "p16", "page_size": 16, "p50_ms": 100},
        ]
        summary = adjudication.serving_summary(rows, "page_size_only")
        self.assertEqual(summary["page_winner_crossovers"], 0)
        self.assertEqual(summary["page_winners"], [])
        self.assertLess(summary["max_case_regret"], adjudication.THRESHOLD)

    def test_serving_max_case_regret_is_not_hidden_by_aggregation(self):
        rows = [
            {"status": "success", "comparison_scope": "layout_only",
             "case_id": "short", "variant": "a", "p50_ms": 100},
            {"status": "success", "comparison_scope": "layout_only",
             "case_id": "short", "variant": "b", "p50_ms": 116},
            {"status": "success", "comparison_scope": "layout_only",
             "case_id": "long", "variant": "a", "p50_ms": 1000},
            {"status": "success", "comparison_scope": "layout_only",
             "case_id": "long", "variant": "b", "p50_ms": 1000},
        ]
        summary = adjudication.serving_summary(rows, "layout_only")
        self.assertAlmostEqual(summary["max_case_regret"], 1.16)
        self.assertLess(summary["static_regret"], 1.03)


if __name__ == "__main__":
    unittest.main()
