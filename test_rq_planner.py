import math
import unittest

from rq_planner import (
    Consumer,
    Epoch,
    Repair,
    Representation,
    common_layout_plan,
    fixed_epoch_policy,
    joint_representation_plan,
    partition_layout_plan,
    per_request_oracle,
    run_epoch_controller,
    shortest_repair_path,
)


def reps():
    return {
        "NHD": Representation("NHD", ("token", "head", "dim"), persistent=True),
        "HND": Representation("HND", ("head", "token", "dim"), persistent=True),
    }


class PlannerTest(unittest.TestCase):
    def test_legality_prunes_cheap_illegal_repair(self):
        repairs = [
            Repair("illegal_view", "NHD", "HND", .01, legal=False),
            Repair("transpose", "NHD", "HND", .4, temporary_bytes=1024),
        ]
        path = shortest_repair_path("NHD", {"HND"}, repairs)
        self.assertIsNotNone(path)
        self.assertEqual([x.name for x in path.repairs], ["transpose"])

    def test_memory_budget_prunes_materialization(self):
        consumers = [Consumer("decode", frozenset({"HND"}), {"HND": 1.0}, 10)]
        repairs = [Repair("transpose", "NHD", "HND", .3, temporary_bytes=2048, persistent_bytes=2048)]
        plan = joint_representation_plan(reps(), consumers, repairs, base_bytes=4096, max_extra_bytes=1024)
        self.assertEqual(plan.storage, ("HND",))

    def test_conversion_is_paid_once_for_reuse(self):
        consumers = [Consumer("decode", frozenset({"NHD", "HND"}), {"NHD": 2.0, "HND": 1.0}, 10)]
        repairs = [
            Repair("n2h", "NHD", "HND", 3.0, persistent_bytes=4096),
            Repair("h2n", "HND", "NHD", 3.0, persistent_bytes=4096),
        ]
        plan = joint_representation_plan(reps(), consumers, repairs, base_bytes=4096)
        self.assertAlmostEqual(plan.total_ms, 10.0)
        self.assertIn("HND", plan.storage)

    def test_partition_positive_and_negative_regimes(self):
        groups = {
            "write": [Consumer("writer", frozenset({"NHD", "HND"}), {"NHD": 1.0, "HND": 3.0}, 10)],
            "read": [Consumer("reader", frozenset({"NHD", "HND"}), {"NHD": 3.0, "HND": 1.0}, 10)],
        }
        common = common_layout_plan(reps(), [*groups["write"], *groups["read"]])
        split = partition_layout_plan(groups, reps(), pool_overhead_ms=1, pool_overhead_bytes=1, base_bytes=1)
        self.assertLess(split.total_ms, common.total_ms)
        expensive = partition_layout_plan(groups, reps(), pool_overhead_ms=30, pool_overhead_bytes=1, base_bytes=1)
        self.assertGreater(expensive.total_ms, common.total_ms)

    def test_empty_intersection_can_be_repaired(self):
        consumers = [
            Consumer("target", frozenset({"NHD"}), {"NHD": 1.0}, 1),
            Consumer("draft", frozenset({"HND"}), {"HND": 1.0}, 1),
        ]
        self.assertIsNone(common_layout_plan(reps(), consumers))
        repairs = [Repair("n2h", "NHD", "HND", .2, persistent_bytes=1024)]
        plan = joint_representation_plan(reps(), consumers, repairs, base_bytes=1024)
        self.assertIsNotNone(plan)
        self.assertTrue(plan.repairs)

    def test_epoch_hysteresis_avoids_per_request_thrashing(self):
        epochs = [
            Epoch("short_A", 2, {"NHD": 1.0, "HND": .9}),
            Epoch("long_B", 100, {"NHD": 2.0, "HND": 1.0}),
        ]
        switch = {("NHD", "HND"): 10.0, ("HND", "NHD"): 10.0}
        epochal = run_epoch_controller(epochs, ["NHD", "HND"], initial="NHD", switch_cost_ms=switch)
        fixed = fixed_epoch_policy(epochs, "NHD")
        per_request = per_request_oracle(epochs, ["NHD", "HND"], initial="NHD", switch_cost_ms=switch)
        self.assertEqual(epochal.switch_count, 1)
        self.assertLess(epochal.total_ms, fixed.total_ms)
        self.assertLess(epochal.total_ms, per_request.total_ms)


if __name__ == "__main__":
    unittest.main()
