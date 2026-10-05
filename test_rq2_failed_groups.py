import json
from pathlib import Path
import tempfile
import unittest
from rq2_failed_group_plan import graph_valid,plan,expected_pairs


class FailedGroupTests(unittest.TestCase):
    def fixture(self,root):
        cases=[{"model_id":"real_geometry_fixture","phase":"decode","batch":1,"query_length":1,"kv_length":16,
                "case_id":f"g{g}_s{s}","kv_lengths":[16+g],"stage":s} for g in (0,1) for s in (0,1,2)]
        (root/"design_manifest.json").write_text(json.dumps({"cases":cases}))
        (root/"SMOKE_APPROVAL.json").write_text('{"approved_frameworks":["triton"]}')
        (root/"process_status.tsv").write_text("triton\t1\n")
        rows=[]
        for c in cases:
            if c["case_id"]=="g1_s1":continue
            rows.append({"record_type":"graph","framework":"triton","case":c,
                         "executed_graph_status":"correctness_verified_complete_controlled_graph",
                         "attempted_candidate_count":1,"measured_candidate_count":1,
                         "candidates":[{"correctness":{"correct":True},"train_samples_ms":[1.,2.],"train_median_ms":1.5}]})
        rows += [{"record_type":"growth_pair","small_case_id":a,"expanded_case_id":b,"speedup_frozen_over_free":1.,
                  "holdout":{"frozen_ms":[1.,2.],"free_ms":[1.,2.],"free_post_correctness":{"correct":True},"frozen_post_correctness":{"correct":True}}}
                 for a,b in expected_pairs(cases)]
        rows.append({"record_type":"completion"})
        (root/"triton.jsonl").write_text("\n".join(map(json.dumps,rows))+"\n")
        return rows
    def test_failed_stage_reruns_entire_sequence_not_successful_other_group(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root)
            self.assertEqual(plan(root)["case_ids_by_framework"],{"triton":["g1_s0","g1_s1","g1_s2"]})
    def test_mismatched_or_nan_median_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);rows=self.fixture(root);row=rows[0];self.assertTrue(graph_valid(row))
            row["candidates"][0]["train_samples_ms"][0]=float("nan");self.assertFalse(graph_valid(row))
    def test_active_run_not_schedulable(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);self.fixture(root);(root/"process_status.tsv").write_text("")
            with self.assertRaises(RuntimeError):plan(root)


if __name__=="__main__":unittest.main()
