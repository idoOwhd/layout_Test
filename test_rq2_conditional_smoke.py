import json
from pathlib import Path
import tempfile
import unittest
from check_rq2_conditional_smoke import verify


class ConditionalApprovalTests(unittest.TestCase):
    def fixture(self,parent):
        smoke=parent/"smoke";run=parent/"run"
        names=("rq2_domain_anchor_bench.py","rq2_growth_bench.py","rq2_growth_search.py",
               "rq2_growth_kernels.py","rq2_growth_cutlass.cu","rq1_diverse_kernels.py",
               "rq1_flashinfer_workspace.py","audit_rq2_holdout_statistics.py")
        for folder in (smoke,run):
            (folder/"source_snapshot").mkdir(parents=True)
            for name in names:(folder/"source_snapshot"/name).write_text(name)
            (folder/"design_manifest.json").write_text("{}")
        frameworks=("pytorch","triton","cutlass","tvm","vllm","sglang")
        (smoke/"process_status.tsv").write_text("\n".join(f+"\t0" for f in frameworks)+"\n")
        for framework in frameworks:
            rows=[{"record_type":"conditional_domain_result","case":{"case_id":"real"}},
                  {"record_type":"completion","expected_graphs":1,"counts":{
                      "domain_graphs":1,"anchor_graphs":0,"anchor_semantically_inapplicable":1,"failures":0}}]
            (smoke/(framework+".jsonl")).write_text("\n".join(map(json.dumps,rows)))
        return smoke,run
    def test_matching_passing_smoke_approved(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,r=self.fixture(Path(tmp));self.assertEqual(len(verify(s,r)["frameworks"]),6)
    def test_changed_worker_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,r=self.fixture(Path(tmp));(r/"source_snapshot/rq2_domain_anchor_bench.py").write_text("new")
            with self.assertRaises(ValueError):verify(s,r)
    def test_missing_anchor_case_not_silently_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            s,r=self.fixture(Path(tmp));p=s/"sglang.jsonl"
            rows=[json.loads(l) for l in p.read_text().splitlines()]
            rows[-1]["counts"]["anchor_semantically_inapplicable"]=0
            p.write_text("\n".join(map(json.dumps,rows)))
            with self.assertRaises(ValueError):verify(s,r)


if __name__=="__main__":unittest.main()
