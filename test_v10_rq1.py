import unittest
from pathlib import Path

from analyze_v10_rq1 import adjudicate_case, compare_solution, required_fields
from build_v10_rq1_cases import DEFAULT_MANIFEST, core_cases, sensitivity_cases


def measurement(run_id, strategy, edge_us, producer_layout, producer_us,
                case_id="v10-core-r8", stage="3-edge", reuse=1):
    return {
        "run_id": run_id, "case_id": case_id, "stage": stage,
        "subgraph": "gqa_kv_decode", "source": "test", "dtype": "BF16",
        "batch": 8.0, "q_len": 1.0, "kv_len": 2048.0,
        "Hq": 32.0, "Hkv": 4.0, "page_size": 64.0,
        "reuse_count": float(reuse), "strategy": strategy,
        "layout_producer": producer_layout, "producer_us": producer_us,
        "edge_us": edge_us, "correctness_pass": 1.0, "status": "success",
    }


class V10Rq1Test(unittest.TestCase):
    def rows(self, repetitions=5):
        rows = []
        specs = {
            "NN": (10.0, "NHD", 1.0),
            "NH-copy": (8.0, "NHD", 1.0),
            "HH-native": (5.0, "HND", 2.0),
            "HN-copy": (12.0, "HND", 2.0),
            "NH-view": (7.0, "NHD", 1.0),
        }
        for repetition in range(repetitions):
            for strategy, (edge, layout, producer) in specs.items():
                rows.append(measurement(f"rep-{repetition}", strategy,
                                        edge + repetition * .01, layout, producer))
        return rows

    def test_preregistered_positive_inversion(self):
        result = adjudicate_case("v10-core-r8", self.rows())
        self.assertTrue(result["positive_rank_inversion"])
        self.assertEqual("NHD", result["producer_winner"])
        self.assertEqual("HH-native", result["edge_winner"])
        self.assertGreater(result["regret_ci95_us"][0], 0)

    def test_five_process_repetitions_are_mandatory(self):
        result = adjudicate_case("v10-core-r8", self.rows(repetitions=4))
        self.assertFalse(result["positive_rank_inversion"])

    def test_case_catalog_has_controls_real_shapes_and_all_sweeps(self):
        core = core_cases(DEFAULT_MANIFEST, 3)
        self.assertTrue(any(row["stage"] == "0-control" for row in core))
        self.assertTrue(any(row["source"].startswith("v10:Stage3") for row in core))
        self.assertTrue(any(not row["source"].startswith("v10:") for row in core))
        self.assertTrue(any(row["dtype"] == "NVFP4" for row in core))
        sweeps = sensitivity_cases()
        for token in ("S-BATCH", "S-KVLEN", "S-GQA", "S-QLEN", "S-PAGE"):
            self.assertTrue(any(token in row["source"] for row in sweeps))

    def test_hypothesis_map_names_exact_experiments(self):
        result = adjudicate_case("v10-core-r8", self.rows())
        hypotheses = compare_solution([result])
        self.assertEqual("supported", hypotheses["H1.1"]["status"])
        self.assertIn("NN/NH-copy/HH-native/HN-copy", hypotheses["H1.1"]["experiment"])
        self.assertIn("HH-native", hypotheses["H1.4"]["experiment"])

    def test_cuda_output_declares_v10_schema_and_all_strategies(self):
        source = (Path(__file__).with_name("v10_rq1_edge_bench.cu")
                  .read_text(encoding="utf-8"))
        for field in required_fields():
            self.assertIn(field, source)
        for strategy in ("NN", "NH-copy", "HH-native", "HN-copy", "NH-view"):
            self.assertIn(f'"{strategy}"', source)


if __name__ == "__main__":
    unittest.main()
