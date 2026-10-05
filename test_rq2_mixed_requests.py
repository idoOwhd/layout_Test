import json
from pathlib import Path
import unittest
from types import SimpleNamespace
import torch
from build_rq2_mixed_request_cases import build,digest
from rq2_growth_bench import (compare,reference_attention,GUARD_ELEMENTS,group_key,
                             independent_graph_reference,frontier_names)
from rq2_growth_bench import sample_indices

HERE=Path(__file__).resolve().parent


class MixedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=json.loads((HERE/"ref_talks/rq2_qwen3_real_forward_growth_design_20261001.json").read_text())
        cls.value=build(cls.original)
    def test_counts_and_no_duplicate_cases(self):
        self.assertEqual(self.value["case_count"],1020)
        self.assertEqual(self.value["nested_pair_count"],918)
        self.assertEqual(len({c["case_id"] for c in self.value["cases"]}),1020)
    def test_old_ids_preserved(self):
        self.assertEqual([c["case_id"] for c in self.value["cases"][:720]],
                         [c["case_id"] for c in self.original["cases"]])
    def test_contract_hashes_current(self):
        for c in self.value["cases"]:
            self.assertEqual(c["graph_contract_sha256"],digest({k:v for k,v in c.items() if k!="graph_contract_sha256"}))
    def test_no_changed_or_impossible_request_shapes(self):
        for c in self.value["cases"][720:]:
            self.assertGreater(c["batch"],1)
            self.assertEqual(len(c["kv_lengths"]),c["batch"])
            self.assertTrue(all(c["query_length"]<=n<=c["kv_length"] for n in c["kv_lengths"]))
            self.assertEqual(max(c["kv_lengths"]),c["kv_length"])
            self.assertGreater(len(set(c["kv_lengths"])),1)
    def test_uniform_and_mixed_are_different_groups(self):
        base=next(c for c in self.value["cases"] if c["batch"]==4 and c["phase"]=="decode")
        mixed=next(c for c in self.value["cases"] if c["case_id"]==base["case_id"]+"_mixed")
        self.assertNotEqual(group_key(base),group_key(mixed))
    def test_causal_alignment_uneven_lengths(self):
        # Tiny sizes are CPU unit tests of the independent oracle, NEVER GPU
        # performance cases standing in for real model dimensions.
        q=torch.zeros(2,2,2,8,dtype=torch.float16)
        k=torch.zeros(2,5,1,8,dtype=torch.float16)
        v=torch.arange(5).view(1,5,1,1).expand(2,5,1,8).half()
        out=reference_attention(q,k,v,torch.tensor([5,3]))
        self.assertTrue(torch.equal(out[:, :, 0, 0],torch.tensor([[1.5,2.],[.5,1.]],dtype=torch.float16)))
    def test_negative_zero_nan_and_slot_changes_rejected(self):
        ref=torch.ones(128,dtype=torch.float16)
        self.assertFalse(compare(torch.zeros_like(ref),ref)["correct"])
        self.assertFalse(compare(torch.full_like(ref,float("nan")),ref)["correct"])
        value=ref.clone();value[-1]=0
        self.assertFalse(compare(value,ref)["correct"])
    def test_guard_keeps_native_256_byte_alignment(self):
        self.assertEqual(GUARD_ELEMENTS*2%256,0)
    def test_large_tensor_sample_index_never_rounds_out_of_bounds(self):
        for length in (2**24,2**24+1,2**26,2**29+3):
            ids=sample_indices(length,8192,"cpu")
            self.assertEqual(int(ids[0]),0);self.assertEqual(int(ids[-1]),length-1)
            self.assertTrue(bool((ids>=0).all() and (ids<length).all()))
    def test_embedded_configs_match_pinned_primary_sha(self):
        import hashlib
        for c in self.value["cases"]:
            self.assertEqual(hashlib.sha256(c["pinned_config_json"].encode()).hexdigest(),c["config_sha256"])
    def test_independent_whole_graph_frontier_and_residual_topology(self):
        # Pure CPU algebra test, not a model-size substitution in GPU evidence.
        b,q,kv,h,hq,hk,d,inter=2,3,3,4,2,1,8,6
        group=SimpleNamespace(b=b,q=q,kv=kv,h=h,hq=hq,hk=hk,d=d,i=inter,
            config={"rms_norm_eps":1e-6},x=torch.arange(b*q*h).view(b*q,h).half()/24+.1,
            request_ids=torch.arange(b)[:,None],write_positions=torch.arange(q).repeat(b,1),
            lengths=torch.full((b,),kv),positions=torch.arange(q).repeat(b),
            table=torch.cat((torch.ones(kv,d//2),torch.zeros(kv,d//2)),-1).half())
        group.layers=[]
        for _ in range(8):
            group.layers.append({"wn":torch.ones(h).half(),"wp":torch.ones(h).half(),
                "wqn":torch.ones(d).half(),"wkn":torch.ones(d).half(),
                "wqkv":torch.zeros(h,(hq+2*hk)*d).half(),"wo":torch.zeros(hq*d,h).half(),
                "wgu":torch.zeros(h,2*inter).half(),"wd":torch.zeros(inter,h).half(),
                "kprev":torch.zeros(b,kv,hk,d).half(),"vprev":torch.zeros(b,kv,hk,d).half()})
        for stage in range(10):
            layers={7:2,8:4,9:8}.get(stage,1)
            value=independent_graph_reference(group,{"stage":stage,"decoder_blocks":layers})
            frontier={f"L{layers-1}."+n for n in frontier_names(stage)}
            kv_names={f"L{i}.kv_{kind}_update" for i in range(layers) for kind in ("k","v")} if stage>=3 else set()
            self.assertEqual(set(value),frontier|kv_names)
            residual_key=next(k for k in frontier if "residual" in k)
            self.assertTrue(torch.equal(value[residual_key],group.x))
            self.assertTrue(all(torch.count_nonzero(t)==0 for k,t in value.items() if k!=residual_key))


if __name__=="__main__":unittest.main()
