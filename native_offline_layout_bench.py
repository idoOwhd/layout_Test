#!/usr/bin/env python3
"""Measure one vLLM/SGLang layout policy through the native offline API.

One process handles exactly one variant.  This is required because both
frameworks read layout environment variables during import/engine startup.
The caller launches a fresh process for every policy and concatenates JSONL.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import traceback
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Case:
    input_tokens: int
    output_tokens: int
    prompts: int
    concurrency: int

    @property
    def case_id(self) -> str:
        return (f"offline_t{self.input_tokens}_o{self.output_tokens}_"
                f"n{self.prompts}_c{self.concurrency}")


def parse_case(text: str) -> Case:
    try:
        values = [int(value) for value in text.split(":" )]
        if len(values) != 4 or min(values) <= 0:
            raise ValueError
        return Case(*values)
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "case must be input_tokens:output_tokens:prompts:concurrency") from error


def variant_config(framework: str, variant: str) -> dict:
    configs = {
        "sglang": {
            "auto_nhd": {"kv_layout": "NHD", "page_size": 1,
                         "backend": None, "comparison_scope": "layout_only"},
            "auto_hnd": {"kv_layout": "HND", "page_size": 1,
                         "backend": None, "comparison_scope": "layout_only"},
            "flashinfer_p1": {"kv_layout": "NHD", "page_size": 1,
                              "backend": "flashinfer", "comparison_scope": "page_size_only"},
            "flashinfer_p16": {"kv_layout": "NHD", "page_size": 16,
                               "backend": "flashinfer", "comparison_scope": "page_size_only"},
        },
        "vllm": {
            "LBNHC": {"kv_layout": "LBNHC", "block_size": 16,
                      "comparison_scope": "layout_only"},
            "LBHNC": {"kv_layout": "LBHNC", "block_size": 16,
                      "comparison_scope": "layout_only"},
            "block16": {"kv_layout": "LBNHC", "block_size": 16,
                        "comparison_scope": "page_size_only"},
            "block32": {"kv_layout": "LBNHC", "block_size": 32,
                        "comparison_scope": "page_size_only"},
        },
    }
    try:
        return {"framework": framework, "variant": variant,
                **configs[framework][variant]}
    except KeyError as error:
        known = ", ".join(sorted(configs.get(framework, {})))
        raise ValueError(f"unknown {framework} variant {variant!r}; known: {known}") from error


def token_batch(case: Case, iteration: int, base_token: int) -> list[list[int]]:
    # Each request remains exact length but differs in its first token. Prefix
    # caching is disabled in vLLM and flushed between SGLang measurements.
    return [[base_token + 1 + ((iteration * case.prompts + index) % 64)]
            + [base_token] * (case.input_tokens - 1)
            for index in range(case.prompts)]


def synchronize() -> None:
    import torch
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def execute_case(generate, batch: list[list[int]], output_tokens: int,
                 concurrency: int):
    """Run deterministic waves whose simultaneous batch never exceeds concurrency."""
    outputs = []
    for start in range(0, len(batch), concurrency):
        outputs.append(generate(batch[start:start + concurrency], output_tokens))
    return outputs


def vllm_runner(args: argparse.Namespace, config: dict):
    os.environ["VLLM_KV_CACHE_LAYOUT"] = config["kv_layout"]
    from vllm import LLM, SamplingParams

    llm = LLM(model=args.model, gpu_memory_utilization=args.memory_fraction,
              block_size=config["block_size"], max_model_len=args.max_model_len,
              enforce_eager=not args.enable_cuda_graph,
              enable_prefix_caching=False, disable_log_stats=True)

    def generate(batch: list[list[int]], output_tokens: int):
        prompts = [{"prompt_token_ids": prompt} for prompt in batch]
        params = SamplingParams(max_tokens=output_tokens, min_tokens=output_tokens,
                                temperature=0, ignore_eos=True)
        return llm.generate(prompts, params, use_tqdm=False)

    def close() -> None:
        # vLLM 0.29 exposes shutdown on the EngineCore client rather than on
        # the compatibility LLMEngine wrapper.
        llm.llm_engine.engine_core.shutdown()

    return generate, close


def sglang_runner(args: argparse.Namespace, config: dict):
    if config["kv_layout"] == "HND":
        os.environ["SGLANG_USE_HND_KVCACHE"] = "1"
    else:
        os.environ.pop("SGLANG_USE_HND_KVCACHE", None)
    import sglang as sgl

    kwargs = {"model_path": args.model,
              "mem_fraction_static": args.memory_fraction,
              "page_size": config["page_size"],
              "disable_cuda_graph": not args.enable_cuda_graph,
              "disable_radix_cache": True,
              "log_level": "error"}
    if config.get("backend"):
        kwargs["attention_backend"] = config["backend"]
    engine = sgl.Engine(**kwargs)

    def generate(batch: list[list[int]], output_tokens: int):
        return engine.generate(
            input_ids=batch,
            sampling_params={"max_new_tokens": output_tokens,
                             "temperature": 0, "ignore_eos": True})

    return generate, engine.shutdown


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("vllm", "sglang"), required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--memory-fraction", type=float, default=.60)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--prompt-token-id", type=int, default=42)
    parser.add_argument("--enable-cuda-graph", action="store_true")
    args = parser.parse_args()
    args.case = args.case or [Case(128, 1, 3, 1), Case(1024, 4, 3, 1)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    config = variant_config(args.framework, args.variant)
    rows: list[dict] = []
    close = None
    try:
        factory = vllm_runner if args.framework == "vllm" else sglang_runner
        generate, close = factory(args, config)
        # Warm the selected native execution path without polluting a case.
        generate([[args.prompt_token_id] * 32], 1)
        for case in args.case:
            samples = []
            for repetition in range(args.repetitions):
                batch = token_batch(case, repetition, args.prompt_token_id)
                synchronize()
                begin = time.perf_counter()
                execute_case(generate, batch, case.output_tokens, case.concurrency)
                synchronize()
                samples.append((time.perf_counter() - begin) * 1e3)
            ordered = sorted(samples)
            p50 = statistics.median(ordered)
            mean = statistics.fmean(ordered)
            rows.append({
                **config, "case_id": case.case_id, "status": "success",
                "layout": config["kv_layout"],
                "page_size": config.get("page_size", config.get("block_size")),
                "p50_ms": p50, "mean_ms": mean, "samples_ms": samples,
                "wall_seconds": p50 / 1e3,
                "prompt_tokens": case.input_tokens * case.prompts,
                "completion_tokens": case.output_tokens * case.prompts,
                "output_tokens_per_second": case.output_tokens * case.prompts / (p50 / 1e3),
                "requested_prompt_tokens_per_request": case.input_tokens,
                "requested_output_tokens_per_request": case.output_tokens,
                "request_count": case.prompts,
                "requested_concurrency": case.concurrency,
                "effective_max_simultaneous_batch": min(case.concurrency, case.prompts),
                "execution_schedule": "sequential_waves_of_simultaneous_batch",
                "execution_mode": "native_offline_engine",
                "cuda_graph": args.enable_cuda_graph,
                "prompt_representation": "exact_token_ids_unique_first_token",
            })
    except BaseException as error:
        rows.append({**config, "case_id": "engine_or_generate", "status": "failed",
                     "error": repr(error), "traceback": traceback.format_exc(),
                     "execution_mode": "native_offline_engine"})
    finally:
        if close is not None:
            try:
                close()
            except BaseException as error:
                rows.append({**config, "case_id": "shutdown", "status": "failed",
                             "error": repr(error), "traceback": traceback.format_exc(),
                             "execution_mode": "native_offline_engine"})
        args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                       for row in rows), encoding="utf-8")
    success = sum(row["status"] == "success" for row in rows)
    print(json.dumps({"framework": args.framework, "variant": args.variant,
                      "rows": len(rows), "success": success,
                      "output": str(args.output)}, ensure_ascii=False))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
