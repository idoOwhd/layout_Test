import json
from pathlib import Path
import tempfile
import unittest
from check_rq2_smoke_approval import GPU_SOURCES,verify


class ApprovalTests(unittest.TestCase):
    def fixture(self,parent):
        smoke=parent/"smoke";run=parent/"run"
        for folder in (smoke,run):
            (folder/"source_snapshot").mkdir(parents=True)
            for name in GPU_SOURCES:(folder/"source_snapshot"/name).write_text(name)
            (folder/"design_manifest.json").write_text("{}")
        (smoke/"process_status.tsv").write_text("triton\t0\n")
        graph={"record_type":"graph","case":{"case_id":"real_case"},
               "executed_graph_status":"correctness_verified_complete_controlled_graph"}
        completion={"record_type":"completion","expected_graphs":1,
                    "counts":{"graphs":1,"failures":0,"incorrect_candidates":0}}
        (smoke/"triton.jsonl").write_text(json.dumps(graph)+"\n"+json.dumps(completion)+"\n")
        return smoke,run
    def test_identical_code_and_complete_smoke_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke,run=self.fixture(Path(tmp))
            self.assertEqual(verify(smoke,run,["triton"])["approved_frameworks"],["triton"])
    def test_changed_arithmetic_is_not_smoke_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke,run=self.fixture(Path(tmp))
            (run/"source_snapshot/rq2_growth_kernels.py").write_text("changed")
            with self.assertRaises(ValueError):verify(smoke,run,["triton"])
    def test_missing_framework_or_failed_process_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke,run=self.fixture(Path(tmp))
            with self.assertRaises(ValueError):verify(smoke,run,["vllm"])
            (smoke/"process_status.tsv").write_text("triton\t1\n")
            with self.assertRaises(ValueError):verify(smoke,run,["triton"])
    def test_incorrect_candidates_make_smoke_fail_closed(self):
        with tempfile.TemporaryDirectory() as tmp:
            smoke,run=self.fixture(Path(tmp))
            rows=[json.loads(x) for x in (smoke/"triton.jsonl").read_text().splitlines()]
            rows[-1]["counts"]["incorrect_candidates"]=1
            (smoke/"triton.jsonl").write_text("\n".join(map(json.dumps,rows)))
            with self.assertRaises(ValueError):verify(smoke,run,["triton"])


if __name__=="__main__":unittest.main()
