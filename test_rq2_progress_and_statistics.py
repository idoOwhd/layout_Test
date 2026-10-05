import json
from pathlib import Path
import tempfile
import unittest
from check_rq2_graph_growth_progress import read_rows,progress
from audit_rq2_holdout_statistics import paired_bootstrap


class ProgressAndStatisticsTests(unittest.TestCase):
    def test_live_partial_last_record_is_not_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"a.jsonl";p.write_text('{"record_type":"graph"}\n{"record_')
            rows,partial=read_rows(p);self.assertEqual(len(rows),1);self.assertTrue(partial)
    def test_middle_record_corruption_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"a.jsonl";p.write_text('broken\n{}\n')
            with self.assertRaises(json.JSONDecodeError):read_rows(p)
    def test_empty_run_cannot_be_full_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/"design_manifest.json").write_text('{"cases":[{"case_id":"real"}]}')
            value=progress(root);self.assertFalse(value["all_declared_H2_1_full_matrix_passed"])
            self.assertTrue(all(v["state"]=="not_started" for v in value["frameworks"]))
    def test_bootstrap_preserves_exact_factor(self):
        value=paired_bootstrap([2,4,6,8,10],[1,2,3,4,5],draws=200)
        self.assertEqual(value["ratio_of_medians"],2)
        self.assertEqual(value["ratio_of_medians_paired_bootstrap_95_interval"],[2,2])
    def test_two_estimands_must_not_be_confused(self):
        value=paired_bootstrap([2,20,20],[1,1,10],draws=200)
        self.assertEqual(value["ratio_of_medians"],20)
        self.assertEqual(value["median_of_paired_ratios"],2)
    def test_invalid_or_unpaired_samples_fail_closed(self):
        for a,b in (([],[]),([1],[1,2]),([0],[1]),([float("nan")],[1]),([float("inf")],[1])):
            with self.assertRaises(ValueError):paired_bootstrap(a,b)


if __name__=="__main__":unittest.main()
