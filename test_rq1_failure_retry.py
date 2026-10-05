import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from analyze_rq1_diversity import assess
from rq1_flashinfer_workspace import growth_request,plan_with_workspace,WorkspaceLimitError
from rq1_failure_retry import passing_cell,select_failures,prepare,merge,dump,HERE


def overflow(name,size,remaining,alignment=16):
    return RuntimeError(f"Buffer overflow when allocating memory for {name} with size {size} "
                        f"and alignment {alignment}, but only {remaining} bytes available in AlignedAllocator")


def measurements(case_id,mode,rep):
    return [dict(case_id=case_id,consumer_mode=mode,process_repetition=rep,status="success",
        stage=stage,strategy="NHD" if stage=="producer" else "NHD/identity",correctness={"correct":True})
        for stage in ("producer","edge")]


class WorkspaceTests(unittest.TestCase):
    def test_scalar_after_vector_exactly_fills_buffer(self):
        old=128*1024**2
        info=growth_request(overflow("batch_prefill_tmp_s",262144,0),old,2*1024**3)
        self.assertEqual(info["next_bytes"],256*1024**2)
        self.assertGreater(info["minimum_required_bytes"],old)
    def test_large_vector_also_leaves_scalar_capacity(self):
        info=growth_request(overflow("batch_prefill_tmp_v",512*1024**2,128*1024**2),128*1024**2,2*1024**3)
        self.assertEqual(info["next_bytes"],1024**3)
    def test_limit_and_unrelated_errors_fail_closed(self):
        with self.assertRaises(WorkspaceLimitError):
            growth_request(overflow("batch_prefill_tmp_v",512*1024**2,128*1024**2),128*1024**2,512*1024**2)
        self.assertIsNone(growth_request(RuntimeError("CUDA out of memory"),128,1024))
        self.assertIsNone(growth_request(overflow("batch_prefill_request_indices",512,0),128,1024))
    def test_retry_rebuilds_wrapper_only_at_planning(self):
        allocations=[];plans=[]
        def factory(buffer):return {"buffer":buffer,"backend":"fa2","split":"auto","layout":"NHD"}
        def plan(wrapper):
            plans.append((wrapper["buffer"],wrapper["backend"],wrapper["split"],wrapper["layout"]))
            if wrapper["buffer"]<256:raise overflow("batch_prefill_tmp_s",16,0)
        wrapper,metadata=plan_with_workspace(factory,plan,lambda n:allocations.append(n) or n,128,128,1024)
        self.assertEqual(allocations,[256]);self.assertEqual(wrapper["buffer"],256)
        self.assertTrue(metadata["growth_outside_timed_region"])
        self.assertEqual([p[1:] for p in plans],[("fa2","auto","NHD")]*2)
        with self.assertRaisesRegex(RuntimeError,"unexpected"):
            plan_with_workspace(factory,lambda w:(_ for _ in ()).throw(RuntimeError("unexpected")),lambda n:n,128,128,1024)


class RetryTests(unittest.TestCase):
    def test_execution_failure_not_misreported_as_layout_negative(self):
        self.assertEqual(assess([{"status":"error","process_repetition":0}])["status"],"measurement_failed")
    def test_exact_failed_cells_only_and_smoke_dedup(self):
        cases=[{"case_id":"a","family":"kv_writer_swa_flashinfer"},
               {"case_id":"b","family":"kv_writer_swa_flashinfer"}]
        rows=[]
        for c in cases:
            for mode in ("vllm","sglang"):
                for rep in range(2):
                    if c["case_id"]=="a" and mode=="sglang":
                        rows.append(dict(case_id="a",consumer_mode=mode,process_repetition=rep,status="error"))
                    else:rows.extend(measurements(c["case_id"],mode,rep))
        full,failed,_=select_failures(cases,2,rows,False)
        self.assertEqual(full,[("a","sglang",0),("a","sglang",1)])
        smoke,_,_=select_failures(cases,2,rows,True)
        self.assertEqual(smoke,[("a","sglang",0)])
        self.assertEqual(len(failed),2)
    def test_duplicates_and_bad_correctness_cannot_replace_old_cells(self):
        rows=measurements("a","vllm",0)
        self.assertTrue(passing_cell(rows));self.assertFalse(passing_cell(rows+[rows[0]]))
        rows[0]["correctness"]["correct"]=False
        self.assertFalse(passing_cell(rows))
    def test_merge_read_only_originals_and_no_duplicate_cells(self):
        with tempfile.TemporaryDirectory(prefix="rq1-retry-test-") as tmp:
            old=Path(tmp)/"old";old.mkdir();(old/"raw").mkdir()
            dump(old/"manifest.json",{"cases":[{"case_id":"a","family":"kv_writer_swa_flashinfer"}],"case_count":1})
            pre={"manifest_sha256":hashlib.sha256((old/"manifest.json").read_bytes()).hexdigest(),
                "independent_process_repetitions":1,"warmup":8,"iterations":30,
                "source_sha256":{n:hashlib.sha256((HERE/n).read_bytes()).hexdigest()
                    for n in ("rq1_diverse_kernels.py","build_rq1_diversity_cases.py")}}
            dump(old/"preflight.json",pre)
            original=measurements("a","vllm",0)+[dict(case_id="a",consumer_mode="sglang",process_repetition=0,status="error")]
            path=old/"raw"/"original.jsonl"
            path.write_text("".join(json.dumps(r)+"\n" for r in original));old_bytes=path.read_bytes()
            new=Path(tmp)/"new";prepare(old,new)
            dump(new/"preflight.json",pre)
            (new/"raw"/"fixed.jsonl").write_text("".join(json.dumps(r)+"\n" for r in measurements("a","sglang",0)))
            report=merge(new)
            self.assertTrue(report["all_previous_files_unchanged"])
            self.assertEqual(path.read_bytes(),old_bytes)
            merged=[json.loads(s) for s in (new/"merged/raw/effective_measurements.jsonl").read_text().splitlines()]
            self.assertEqual(len(merged),4);self.assertTrue(all(r["status"]=="success" for r in merged))
            self.assertEqual(len({(r["case_id"],r["consumer_mode"],r["stage"]) for r in merged}),4)
            self.assertEqual(report["replaced_cells"],1)


if __name__=="__main__":unittest.main()
