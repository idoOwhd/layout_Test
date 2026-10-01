import unittest
from pathlib import Path
import sys

import torch

HERE = Path(__file__).resolve().parent
BASELINE_ROOT = HERE.parent
for path in (HERE, BASELINE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from llm_boundary_layout_sweep import alternate_stride, strategy_error_status
from real_world_workloads import (correctness_verdict, make_workload, rmsnorm,
                                  tensor_error_metrics)


class RepairedWorkloadTests(unittest.TestCase):
    def test_batched_decode_cache_inputs_preserve_request_axis(self):
        cases = [
            {"structure": "gqa", "phase": "decode", "tensor_seed": 3,
             "shape": {"batch": 2, "query_length": 1, "kv_length": 4,
                       "x": [2, 1, 8], "rms_weight": [8],
                       "num_query_heads": 2, "num_kv_heads": 1, "head_dim": 4,
                       "w_q": [8, 8], "w_k": [8, 4], "w_v": [8, 4],
                       "w_o": [8, 8]}},
            {"structure": "mla", "phase": "decode", "tensor_seed": 4,
             "shape": {"batch": 2, "query_length": 1, "kv_length": 4,
                       "x": [2, 1, 8], "rms_weight": [8], "num_heads": 2,
                       "qk_head_dim": 4, "value_head_dim": 2,
                       "kv_lora_rank": 3, "w_q": [8, 8], "w_kv_down": [8, 3],
                       "w_kv_up": [3, 12], "w_o": [4, 8]}},
            {"structure": "sparse_attention", "phase": "decode", "tensor_seed": 5,
             "shape": {"batch": 2, "query_length": 1, "kv_length": 4,
                       "x": [2, 1, 8], "rms_weight": [8], "hidden_size": 8,
                       "num_heads": 2, "head_dim": 4, "value_head_dim": 4,
                       "index_dim": 3, "selected_tokens": 2,
                       "w_index_q": [8, 3], "w_index_k": [8, 3],
                       "w_q": [8, 8], "w_k": [8, 8], "w_v": [8, 8],
                       "w_o": [8, 8]}},
        ]
        for case in cases:
            with self.subTest(structure=case["structure"]):
                workload = make_workload(case, "cpu")
                self.assertTrue(all(tensor.shape[0] == 2 for tensor in workload.inputs))
                output = workload.model(*workload.inputs)
                self.assertEqual(tuple(output.shape), (2, 1, 8))

    def test_decode_shaped_tensor_gets_real_padded_stride(self):
        source = torch.arange(16, dtype=torch.float32).reshape(1, 1, 16)
        alternate, kind = alternate_stride(source)
        self.assertEqual(kind, "padded_last_dim_stride2")
        self.assertTrue(torch.equal(source, alternate))
        self.assertNotEqual(source.stride(), alternate.stride())
        self.assertFalse(alternate.is_contiguous())

    def test_scale_aware_correctness_rejects_relative_mismatch(self):
        reference = torch.tensor([1e-3, 100.0])
        actual = torch.tensor([2e-3, 100.0])
        metrics = tensor_error_metrics(actual, reference, atol=1e-5, rtol=1e-2)
        self.assertFalse(metrics["correct"])
        self.assertGreater(metrics["max_normalized_error"], 1)

    def test_long_recurrence_uses_abs_and_rmse_not_near_zero_relative_error(self):
        case = {"structure": "mamba2", "shape": {"query_length": 2048}}
        metrics = {"correct": False, "nonfinite_count": 0,
                   "max_abs_error": .25, "rmse": .03}
        passed, rule = correctness_verdict(case, metrics)
        self.assertTrue(passed)
        self.assertIn("recurrent_prefill", rule)

    def test_long_routed_moe_uses_abs_and_rmse_bounds(self):
        case = {"structure": "moe", "shape": {"query_length": 2048}}
        metrics = {"correct": False, "nonfinite_count": 0,
                   "max_abs_error": .25, "rmse": .03}
        passed, rule = correctness_verdict(case, metrics)
        self.assertTrue(passed)
        self.assertIn("routed_moe_prefill", rule)

    def test_sparse_prefill_uses_abs_and_rmse_bounds(self):
        case = {"structure": "sparse_attention", "shape": {"query_length": 2048}}
        metrics = {"correct": False, "nonfinite_count": 0,
                   "max_abs_error": .04, "rmse": .002}
        passed, rule = correctness_verdict(case, metrics)
        self.assertTrue(passed)
        self.assertIn("sparse_prefill", rule)

    def test_routed_moe_has_fixed_capacity_dispatch(self):
        case = {"structure": "moe", "tensor_seed": 1,
                "shape": {"x": [1, 4, 8], "rms_weight": [8],
                          "num_experts": 4, "experts_per_token": 2,
                          "hidden_size": 8, "intermediate_size": 16,
                          "w_router": [8, 4]}}
        workload = make_workload(case, "cpu")
        output = workload.model(*workload.inputs)
        self.assertEqual(tuple(output.shape), (1, 4, 8))
        self.assertEqual(tuple(workload.model.dispatch_token.shape)[0], 4)
        self.assertEqual(int(workload.model.dispatch_valid.sum()), 8)

    def test_moe_expert_weights_use_logical_fan_in_not_expert_count(self):
        case = {"structure": "moe", "tensor_seed": 7,
                "shape": {"x": [1, 8, 64], "rms_weight": [64],
                          "num_experts": 4, "experts_per_token": 2,
                          "hidden_size": 64, "intermediate_size": 128,
                          "w_router": [64, 4]}}
        workload = make_workload(case, "cpu")
        # Correct fan-in scales are 1/sqrt(64)=0.125 and
        # 1/sqrt(128)~=0.088.  The old bug used 1/sqrt(experts)=0.5.
        self.assertLess(float(workload.model.w_gate.float().std()), .16)
        self.assertLess(float(workload.model.w_down.float().std()), .12)
        self.assertTrue(torch.isfinite(workload.model(*workload.inputs)).all())

    def test_higher_order_layout_failure_is_explicitly_unsupported(self):
        class UncapturedHigherOrderOpError(RuntimeError):
            pass

        status = strategy_error_status(
            UncapturedHigherOrderOpError("HigherOrderOperator body not captured"))
        self.assertEqual(status, "unsupported_alternate_layout")

    def test_mamba_scan_matches_literal_recurrence(self):
        case = {"structure": "mamba2", "tensor_seed": 2,
                "shape": {"x": [1, 4, 8], "rms_weight": [8],
                          "hidden_size": 8, "state_size": 4,
                          "state": [1, 8, 4], "w_in": [8, 16],
                          "w_dt": [8, 8], "w_b": [8, 4],
                          "w_c": [8, 4], "w_out": [8, 8]}}
        workload = make_workload(case, "cpu")
        model, (x, initial_state) = workload.model, workload.inputs
        actual, final_state = model(x, initial_state)
        residual = x
        y = rmsnorm(x, model.rms_weight)
        u, z = (y @ model.w_in).chunk(2, -1)
        dt = torch.nn.functional.softplus(y @ model.w_dt).float().clamp_max(1.0)
        bv, cv = (y @ model.w_b).float(), (y @ model.w_c).float()
        state = initial_state
        outputs = []
        for token in range(y.shape[1]):
            state = (torch.exp(dt[:, token, :, None] * -model.a.abs()[None]) * state
                     + dt[:, token, :, None] * bv[:, token, None, :]
                     * u[:, token, :, None].float())
            outputs.append(((state * cv[:, token, None, :]).sum(-1)
                            + model.d * u[:, token].float())
                           * torch.nn.functional.silu(z[:, token].float()))
        expected = residual + torch.stack(outputs, 1).to(x.dtype) @ model.w_out
        self.assertTrue(torch.allclose(actual, expected, atol=1e-3, rtol=1e-3))
        self.assertTrue(torch.allclose(final_state, state, atol=1e-4, rtol=1e-4))


if __name__ == "__main__":
    unittest.main()
