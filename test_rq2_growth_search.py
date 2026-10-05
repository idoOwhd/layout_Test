"""Regression checks: set-aware freezing, aliases, provenance, no fake results."""
import unittest
from rq2_growth_search import candidates, frozen_eligible, layout_axes, stable_seed, near_optimal,summarize_pair
from audit_rq2_holdout_statistics import paired_bootstrap


class SearchTests(unittest.TestCase):
    def case(self,stage=6,b=4,q=1,layers=1):
        return {"stage":stage,"batch":b,"query_length":q,"decoder_blocks":layers}
    def test_one_token_aliases(self):
        self.assertEqual(layout_axes(self.case(b=1)),["L0.kv_cache"])
    def test_multiple_requests_keep_axes(self):
        self.assertEqual(len(layout_axes(self.case())),5)
    def test_exact_space(self):
        values,scope=candidates(layout_axes(self.case()))
        self.assertEqual(len(values),32);self.assertEqual(scope,"exact_declared_space")
    def test_frozen_set_not_single_winner(self):
        self.assertTrue(frozen_eligible((1,0,1),["a","b","c"],["a","b"],[(0,1),(1,0)]))
        self.assertFalse(frozen_eligible((0,0,1),["a","b","c"],["a","b"],[(0,1),(1,0)]))
    def test_every_old_near_plan_has_measured_extension(self):
        axes=[str(i) for i in range(10)]
        old={"axes":axes[:5],"near_plans":[(0,0,0,0,0),(1,0,1,0,1)]}
        values,scope=candidates(axes,old,budget=32)
        self.assertIn("not_oracle",scope)
        for v in old["near_plans"]:
            self.assertTrue(any(x[:5]==v for x in values))
    def test_semantic_old_axis_removal_rejected(self):
        with self.assertRaises(ValueError):
            candidates([str(i) for i in range(6)],{"axes":["old"],"near_plans":[(1,)]})
    def test_stage_independent_seed(self):
        key=("model","decode",4,1,512)
        self.assertEqual(stable_seed(key,"L0.wqkv"),stable_seed(key,"L0.wqkv"))
        self.assertNotEqual(stable_seed(key,"L0.wqkv"),stable_seed(key,"L1.wqkv"))
    def test_incorrect_plan_cannot_enter_near_set(self):
        rows=[{"bits":[0],"train_median_ms":2,"correctness":{"correct":True}},
              {"bits":[1],"train_median_ms":1,"correctness":{"correct":False}}]
        self.assertEqual(near_optimal(rows,.03),[(0,)])
    def test_point_and_interval_target_the_same_estimand(self):
        case={**self.case(stage=3,b=1),"case_id":"larger"}
        rows=[{"bits":[0],"train_median_ms":2,"correctness":{"correct":True}},
              {"bits":[1],"train_median_ms":1,"correctness":{"correct":True}}]
        previous={"axes":layout_axes(case),"near_plans":[(0,)],"case_id":"small"}
        holdout={"frozen_ms":[2,20,20],"free_ms":[1,1,10]}
        value=summarize_pair(case,rows,previous,.03,holdout)
        expected=paired_bootstrap(holdout["frozen_ms"],holdout["free_ms"])
        self.assertEqual(value["speedup_frozen_over_free"],20)
        self.assertEqual(value["median_of_paired_ratios"],2)
        self.assertEqual(value["paired_ratio_bootstrap_95_interval"],expected["ratio_of_medians_paired_bootstrap_95_interval"])


if __name__=="__main__":unittest.main()
