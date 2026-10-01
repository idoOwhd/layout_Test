import json
import unittest
from pathlib import Path

import static_layout_analysis as sla


HERE = Path(__file__).resolve().parent


class LayoutModelTests(unittest.TestCase):
    def test_kv_layouts_are_bijective_for_complete_pages(self):
        for layout in ("NHD", "HND", "paged_NHD", "paged_HND"):
            offsets = {
                sla.kv_offset(layout, n, h, d, tokens=8, heads=4, head_dim=16, block_size=4)
                for n in range(8) for h in range(4) for d in range(16)
            }
            self.assertEqual(offsets, set(range(8 * 4 * 16)), layout)

    def test_shared_conflict_control(self):
        data = json.loads((HERE / "layout_cases.json").read_text())
        rows = sla.analyze(data)
        by_layout = {r["layout"]: r for r in rows if r["family"] == "shared_memory"}
        self.assertEqual(by_layout["row_major_stride32"]["max_bank_conflict_degree"], 32)
        self.assertEqual(by_layout["padded_stride33"]["max_bank_conflict_degree"], 1)
        self.assertEqual(by_layout["xor_swizzle"]["max_bank_conflict_degree"], 1)

    def test_all_declared_families_emit_records(self):
        data = json.loads((HERE / "layout_cases.json").read_text())
        expected = {c["family"] for c in data["cases"]}
        actual = {r["family"] for r in sla.analyze(data)}
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    unittest.main()
