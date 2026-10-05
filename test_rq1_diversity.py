#!/usr/bin/env python3
import copy
import json
from pathlib import Path
import unittest

from analyze_rq1_diversity import assess
from build_rq1_diversity_cases import build,FAMILIES,HERE
from build_rq1_source_registry import build as registry_build


def synthetic(n=5,conflict=True):
    rows=[]
    for rep in range(n):
        for layout,ms in (("row",1.),("column",1.2)):
            rows.append(dict(stage="producer",strategy=layout,process_repetition=rep,p50_ms=ms,
                             status="success",correctness={"correct":True}))
        for layout,ms in (("row",3. if conflict else 2.),("column",2. if conflict else 3.)):
            rows.append(dict(stage="edge",strategy=layout+"/identity",process_repetition=rep,p50_ms=ms,
                             status="success",correctness={"correct":True}))
    return rows


class ProtocolTests(unittest.TestCase):
    def test_selection_and_speedup(self):
        result=assess(synthetic())
        self.assertEqual(result["status"],"rq1_supported_in_candidate_space")
        self.assertEqual(result["median_held_out_speedup"],1.5)
        self.assertAlmostEqual(result["mean_held_out_regret_fraction"],1/3)
        self.assertEqual(set(result["selection_repetitions"])&set(result["held_out_repetitions"]),set())
    def test_negative_control(self):
        self.assertEqual(assess(synthetic(conflict=False))["status"],"no_layout_conflict_in_candidate_space")
    def test_smoke_does_not_prove_rq(self):
        self.assertEqual(assess(synthetic(1))["status"],"smoke_only_insufficient_process_repetitions")
    def test_duplicate_is_not_an_extra_observation(self):
        rows=synthetic();rows.append(copy.deepcopy(rows[0]))
        self.assertEqual(assess(rows)["status"],"invalid_duplicate_measurement")
    def test_contract_and_source_dedup(self):
        cases=build(HERE.parent/"real_world_shapes/real_world_shape_manifest.json",False)
        self.assertGreater(len(cases),100)
        self.assertEqual(len(cases),len({c["contract_sha256"] for c in cases}))
        self.assertEqual({c["family"] for c in cases},set(FAMILIES))
        self.assertTrue(any(c["request_batch"]>1 for c in cases))
        self.assertTrue(any(c["kv_length"]%c["page_size"] for c in cases if "kv_writer" in c["family"]))
        for c in cases:
            self.assertEqual(c["dtype"],"float16")
            self.assertRegex(c["model_revision"],r"^[0-9a-f]{40}$")
            self.assertIn(c["model_revision"],c["source_url"])
            self.assertFalse(c["complete_shape_is_observed_production_trace"])
        contexts=json.loads((HERE/"rq1_shared_source_contexts.json").read_text())
        self.assertEqual(len(contexts["records"]),173)
        self.assertEqual(len(contexts["records"]),len({c["url"] for c in contexts["records"]}))
        registry=registry_build(contexts,{},cases)
        self.assertEqual(len(registry),173)
        self.assertTrue(all(not r["native_replay_completed"] for r in registry))
    def test_reject_zero_or_transposed_output(self):
        import torch
        from rq1_diverse_edge_bench import compare,effective_storage,gdn_reference
        torch.manual_seed(2)
        ref=torch.randn(8,8)*.001
        self.assertFalse(compare(torch.zeros_like(ref),ref)["correct"])
        self.assertFalse(compare(ref.t(),ref)["correct"])
        self.assertTrue(compare(ref.clone(),ref)["correct"])
        row=torch.empty(1,16);column=torch.empty(16,1).t()
        self.assertEqual(effective_storage(row),effective_storage(column))
        q=torch.ones(1,1,1,3);k=torch.ones_like(q);v=torch.ones(1,1,1,5)
        out,state=gdn_reference(q,k,v,torch.zeros(1,1,1),torch.full((1,1,1),.5))
        self.assertEqual(state.shape,(1,1,5,3))
        self.assertTrue(torch.allclose(out.float(),torch.full_like(out.float(),3*.5/(3**.5)),atol=.001))


if __name__=="__main__":unittest.main()
