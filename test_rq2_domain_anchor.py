import json
from pathlib import Path
import unittest
import torch
from rq2_domain_anchor_bench import (operator_path,intervention_axes,factorial_assignments,
                                   representation_assignments,RepresentationAdapter)
from rq2_growth_bench import GUARD_ELEMENTS,compare
from rq2_growth_search import layout_axes


class DomainAnchorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cases=json.loads((Path(__file__).parent/"ref_talks/rq2_qwen3_real_forward_growth_mixed_design_20261002_v3.json").read_text())["cases"]
    def case(self,stage,batch=4):
        return next(c for c in self.cases if c["stage"]==stage and c["batch"]==batch and c["phase"]=="decode")
    def test_real_graph_path_uses_residual_branch_not_block_count(self):
        c=self.case(9);p=operator_path(c,"L0.qkv_projection","L7.attention")
        self.assertEqual(p[0],"L0.qkv_projection");self.assertEqual(p[-1],"L7.attention")
        self.assertIn("L0.post_attention_norm",p)
        self.assertNotIn("L0.down_projection",p)  # shorter residual path exists
    def test_no_backward_causal_path_fabricated(self):
        self.assertIsNone(operator_path(self.case(9),"L7.attention","L0.qkv_projection"))
    def test_missing_anchor_is_semantic_inapplicability(self):
        for stage in range(3):self.assertIsNone(intervention_axes(self.case(stage))[0])
    def test_single_row_same_axis_is_not_two_factors(self):
        self.assertIsNone(intervention_axes(self.case(6,batch=1))[0])
        self.assertIsNotNone(intervention_axes(self.case(9,batch=1))[0])
    def test_complete_matched_two_by_two_preserves_other_choices(self):
        c=self.case(9);axes=layout_axes(c);bits=[1]*len(axes);factors,_=intervention_axes(c)
        rows=factorial_assignments(bits,axes,factors["upstream_axis"],factors["anchor_axis"])
        self.assertEqual({(r["upstream"],r["anchor"]) for r in rows},{(0,0),(1,0),(0,1),(1,1)})
        altered={axes.index(factors["upstream_axis"]),axes.index(factors["anchor_axis"])}
        for r in rows:self.assertTrue(all(r["bits"][i]==bits[i] for i in range(len(bits)) if i not in altered))
    def test_materializations_are_real_one_or_two_representation_choices(self):
        c=self.case(6);rows=representation_assignments(c,[0]*len(layout_axes(c)))
        self.assertEqual(len(rows),4)
        for row in rows:self.assertEqual(row["explicit_output_representation_count"],
                                         1+int(row["source_column"]!=row["destination_column"]))
    def test_one_row_layout_aliases_do_not_create_fake_domain_variants(self):
        c=self.case(6,batch=1);rows=representation_assignments(c,[0]*len(layout_axes(c)))
        self.assertEqual(len(rows),1);self.assertEqual(rows[0]["explicit_output_representation_count"],1)
    def test_copy_checked_before_later_layer_reuses_output_backing(self):
        # Tiny CPU correctness regression, NEVER a GPU performance case.
        class CPUPrimitive:
            def gemm(self,x,w,out):out.copy_((x.float()@w.float()).half())
        x=torch.ones(2,3).half();w=torch.ones(3,4).half();out=torch.empty(2,4).half()
        a=RepresentationAdapter(CPUPrimitive());a.target_weight_ptr=w.data_ptr()
        a.raw=torch.full((8+2*GUARD_ELEMENTS,),777,dtype=torch.float16)
        a.temp=a.raw[GUARD_ELEMENTS:-GUARD_ELEMENTS].view(4,2).t();a.check_copy=True
        a.gemm(x,w,out);self.assertEqual(a.copy_checks,[True])
        self.assertTrue(compare(out,(x.float()@w.float()).half())["correct"])
        out.zero_()  # simulate L7 replacing L0 contents in the SAME backing
        self.assertEqual(a.copy_checks,[True])
        self.assertTrue(bool((a.raw[:GUARD_ELEMENTS]==777).all() and (a.raw[-GUARD_ELEMENTS:]==777).all()))


if __name__=="__main__":unittest.main()
