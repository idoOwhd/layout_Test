import argparse
import json
import unittest
from unittest.mock import patch

import analyze_layout_results as analysis
import analyze_persuasive_real_world_rqs as persuasive
import native_offline_layout_bench as offline
import native_serving_layout_bench as serving


class NativeServingTests(unittest.TestCase):
    def test_vllm_layout_override_is_native_environment(self):
        args = argparse.Namespace(framework="vllm", model="/model", memory_fraction=.7, server_arg=[])
        with patch.dict("os.environ", {"VLLM_KV_CACHE_LAYOUT": "OLD"}):
            command, env = serving.server_command(args, {"kv_layout": "LBHNC"}, 32000)
        self.assertEqual(env["VLLM_KV_CACHE_LAYOUT"], "LBHNC")
        self.assertIn("vllm.entrypoints.openai.api_server", command)

    def test_sglang_policy_uses_framework_server(self):
        args = argparse.Namespace(framework="sglang", model="/model", memory_fraction=.7, server_arg=[])
        command, _ = serving.server_command(args, {"backend": "flashinfer", "page_size": 16}, 31000)
        self.assertIn("sglang.launch_server", command)
        self.assertEqual(command[-4:], ["--attention-backend", "flashinfer", "--page-size", "16"])

    def test_sglang_hnd_layout_uses_native_environment(self):
        args = argparse.Namespace(framework="sglang", model="/model", memory_fraction=.7, server_arg=[])
        with patch.dict("os.environ", {"SGLANG_USE_HND_KVCACHE": "0"}):
            _, env = serving.server_command(
                args, {"backend": None, "page_size": None, "kv_layout": "HND"}, 31000)
        self.assertEqual(env["SGLANG_USE_HND_KVCACHE"], "1")

    def test_exact_token_prompts_are_unique_against_prefix_cache(self):
        args = argparse.Namespace(served_model_name=None, model="/model", request_timeout=1,
                                  prompt_token_id=42)
        response = ({"usage": {"prompt_tokens": 8, "completion_tokens": 2}}, 1.0)
        with patch.object(serving, "request_json", return_value=response) as request:
            result = serving.run_requests(args, 31000, serving.Case(8, 2, 2, 1))
        prompts = [call.args[1]["prompt"] for call in request.call_args_list]
        self.assertEqual([len(prompt) for prompt in prompts], [8, 8, 8])
        self.assertEqual(len({prompt[0] for prompt in prompts}), 3)
        self.assertEqual(result["requested_prompt_tokens_per_request"], 8)
        self.assertEqual(result["prompt_representation"], "exact_token_ids_unique_first_token")

    def test_confounded_row_cannot_produce_layout_regret(self):
        row = {"framework": "sglang", "case_id": "c", "layout": "a", "layout_role": "native",
               "p50_ms": 1, "status": "success", "comparison_scope": "serving_policy_confounded", "_file": "x.jsonl"}
        normalized = analysis.normalize(row, {})
        self.assertEqual(normalized["comparison_scope"], "serving_policy_confounded")

    def test_offline_variants_separate_layout_from_page_size(self):
        self.assertEqual(
            offline.variant_config("vllm", "LBHNC")["comparison_scope"],
            "layout_only")
        self.assertEqual(
            offline.variant_config("vllm", "block32")["comparison_scope"],
            "page_size_only")
        self.assertEqual(
            offline.variant_config("sglang", "flashinfer_p16")["page_size"], 16)

    def test_offline_token_batches_keep_exact_shape_and_avoid_same_prefix(self):
        case = offline.Case(128, 1, 3, 1)
        batch = offline.token_batch(case, iteration=2, base_token=42)
        self.assertEqual([len(prompt) for prompt in batch], [128, 128, 128])
        self.assertEqual(len({prompt[0] for prompt in batch}), 3)

    def test_offline_concurrency_is_enforced_as_bounded_waves(self):
        sizes = []
        def generate(batch, output_tokens):
            sizes.append((len(batch), output_tokens))
            return [object()] * len(batch)
        result = offline.execute_case(generate, [[1]] * 10, output_tokens=3, concurrency=4)
        self.assertEqual(sizes, [(4, 3), (4, 3), (2, 3)])
        self.assertEqual(len(result), 3)

    def test_native_metrics_pairs_single_and_multi_request(self):
        rows = [
            {"status": "success", "p50_ms": 9.0, "variant": "NHD",
             "comparison_scope": "layout_only", "case_id": "c1",
             "requested_prompt_tokens_per_request": 128,
             "requested_output_tokens_per_request": 1, "request_count": 3,
             "effective_max_simultaneous_batch": 1},
            {"status": "success", "p50_ms": 3.0, "variant": "NHD",
             "comparison_scope": "layout_only", "case_id": "c3",
             "requested_prompt_tokens_per_request": 128,
             "requested_output_tokens_per_request": 1, "request_count": 3,
             "effective_max_simultaneous_batch": 3},
        ]
        metrics = persuasive.native_metrics(rows)
        self.assertEqual(metrics["single_multi_request_pairs"], 1)
        self.assertEqual(metrics["median_multi_request_wall_speedup"], 3.0)

    def test_coverage_keeps_all_six_frameworks_for_visual_cases(self):
        # The production manifests contain 640 LLM + 83 visual cases.
        llm = json.loads((analysis.HERE.parent / "real_world_shapes" / "real_world_shape_manifest.json").read_text())
        vision = json.loads((analysis.HERE.parent / "vision_shapes" / "vision_common_subgraph_manifest.json").read_text())
        self.assertEqual((len(llm) + len(vision["cases"])) * len(analysis.TARGET_FRAMEWORKS), 4338)


if __name__ == "__main__":
    unittest.main()
