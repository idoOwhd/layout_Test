import json
import unittest
from pathlib import Path

import graph_layout_optimizer as optimizer


HERE = Path(__file__).resolve().parent


class GraphLayoutOptimizerTests(unittest.TestCase):
    def test_exact_beats_local_greedy_on_boundary_counterexample(self):
        spec = json.loads((HERE / "graph_layout_example.json").read_text())
        exact, greedy = optimizer.exact(spec), optimizer.greedy(spec)
        self.assertAlmostEqual(exact.cost_ms, 1.7)
        self.assertAlmostEqual(greedy.cost_ms, 2.3)
        self.assertEqual(exact.decisions[0]["implementation"], "producer_tiled_slightly_slower")
        self.assertEqual(greedy.decisions[0]["implementation"], "producer_row_fast")

    def test_missing_conversion_cost_is_rejected(self):
        spec = {"initial_layouts": {"x": "a"}, "ops": [{"id": "op", "inputs": ["x"], "outputs": ["y"],
                "implementations": [{"id": "b", "input_layouts": {"x": "b"}, "output_layouts": {"y": "b"}, "kernel_ms": 1}]}]}
        with self.assertRaises(KeyError): optimizer.exact(spec)


if __name__ == "__main__": unittest.main()
