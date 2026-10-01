#!/usr/bin/env python3
"""Run native SGLang/vLLM serving policies without substituting another engine.

This is an opt-in end-to-end adapter.  It starts the framework's own OpenAI
server, waits for readiness, sends an identical request set, records the
framework log that proves (or rejects) the requested policy, and then tears the
server down.  Backend-changing SGLang rows are explicitly labelled confounded;
they are not presented as a layout-only causal comparison.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        return f"serving_t{self.input_tokens}_o{self.output_tokens}_n{self.prompts}_c{self.concurrency}"


def percentiles(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {}

    def pick(q: float) -> float:
        return ordered[round((len(ordered) - 1) * q)]

    return {"p20_ms": pick(.2), "p50_ms": pick(.5), "p80_ms": pick(.8)}


def variants(framework: str) -> list[dict]:
    if framework == "sglang":
        return [
            {"id": "auto_nhd", "backend": None, "page_size": None, "kv_layout": "NHD",
             "layout_role": "native", "comparison_scope": "layout_only"},
            {"id": "auto_hnd", "backend": None, "page_size": None, "kv_layout": "HND",
             "layout_role": "alternative", "comparison_scope": "layout_only"},
            {"id": "flashinfer_p1", "backend": "flashinfer", "page_size": 1, "kv_layout": "NHD",
             "layout_role": "alternative", "comparison_scope": "page_size_only"},
            {"id": "flashinfer_p16", "backend": "flashinfer", "page_size": 16, "kv_layout": "NHD",
             "layout_role": "alternative", "comparison_scope": "page_size_only"},
            {"id": "triton_p1", "backend": "triton", "page_size": 1, "kv_layout": "NHD",
             "layout_role": "alternative", "comparison_scope": "serving_policy_confounded"},
        ]
    return [
        {"id": "auto", "kv_layout": None, "block_size": None, "layout_role": "native"},
        {"id": "LBNHC", "kv_layout": "LBNHC", "block_size": None, "layout_role": "alternative"},
        {"id": "LBHNC", "kv_layout": "LBHNC", "block_size": None, "layout_role": "alternative"},
        {"id": "BLNHC", "kv_layout": "BLNHC", "block_size": None, "layout_role": "alternative"},
        {"id": "BLHNC", "kv_layout": "BLHNC", "block_size": None, "layout_role": "alternative"},
        {"id": "block8", "kv_layout": None, "block_size": 8, "layout_role": "alternative",
         "comparison_scope": "page_size_only"},
        {"id": "block16", "kv_layout": None, "block_size": 16, "layout_role": "alternative",
         "comparison_scope": "page_size_only"},
        {"id": "block32", "kv_layout": None, "block_size": 32, "layout_role": "alternative",
         "comparison_scope": "page_size_only"},
    ]


def server_command(args: argparse.Namespace, variant: dict, port: int) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    if args.framework == "sglang":
        command = [sys.executable, "-m", "sglang.launch_server", "--model-path", args.model,
                   "--host", "127.0.0.1", "--port", str(port), "--mem-fraction-static", str(args.memory_fraction)]
        if variant.get("kv_layout") == "HND":
            env["SGLANG_USE_HND_KVCACHE"] = "1"
        else:
            env.pop("SGLANG_USE_HND_KVCACHE", None)
        if variant["backend"]:
            command += ["--attention-backend", variant["backend"]]
        if variant["page_size"]:
            command += ["--page-size", str(variant["page_size"])]
    else:
        command = [sys.executable, "-m", "vllm.entrypoints.openai.api_server", "--model", args.model,
                   "--host", "127.0.0.1", "--port", str(port), "--gpu-memory-utilization", str(args.memory_fraction)]
        if variant["kv_layout"]:
            env["VLLM_KV_CACHE_LAYOUT"] = variant["kv_layout"]
        else:
            env.pop("VLLM_KV_CACHE_LAYOUT", None)
        if variant.get("block_size"):
            command += ["--block-size", str(variant["block_size"])]
    command += args.server_arg
    return command, env


def request_json(url: str, payload: dict, timeout: float) -> tuple[dict, float]:
    body = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, (time.perf_counter() - start) * 1e3


def wait_ready(port: int, process: subprocess.Popen, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as response:
                if response.status == 200:
                    return True
        except (OSError, urllib.error.URLError):
            time.sleep(1)
    return False


def stop(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=20)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def parse_observed_policy(text: str, framework: str) -> dict:
    if framework == "vllm":
        match = re.search(r"Using\s+([A-Z]+)\s+KV cache layout", text)
        return {"observed_kv_layout": match.group(1) if match else None,
                "observation_rule": "vLLM server log: Using <layout> KV cache layout"}
    backend = re.search(r"(?:attention backend|attention_backend)\D+([a-z0-9_]+)", text, re.I)
    return {"observed_attention_backend": backend.group(1).lower() if backend else None,
            "observed_kv_layout": "NHD" if '"NHD"' in text or " NHD" in text else None,
            "observation_rule": "SGLang server log only; source-declared NHD is not promoted to runtime-observed"}


def run_requests(args: argparse.Namespace, port: int, case: Case) -> dict:
    def payload(sequence: int) -> dict:
        # Both framework OpenAI completion schemas accept token-id prompts.
        # Vary the first token to prevent radix/prefix-cache reuse from turning
        # a prefill case into an accidental cache-hit benchmark.
        first = args.prompt_token_id + 1 + sequence
        prompt = [first] + [args.prompt_token_id] * (case.input_tokens - 1)
        return {"model": args.served_model_name or args.model, "prompt": prompt,
                "max_tokens": case.output_tokens, "temperature": 0, "stream": False}

    request_json(f"http://127.0.0.1:{port}/v1/completions", payload(0), args.request_timeout)
    start = time.perf_counter()
    latencies, prompt_tokens, completion_tokens = [], 0, 0
    with ThreadPoolExecutor(max_workers=case.concurrency) as pool:
        futures = [pool.submit(request_json, f"http://127.0.0.1:{port}/v1/completions",
                               payload(index + 1), args.request_timeout)
                   for index in range(case.prompts)]
        for future in as_completed(futures):
            response, latency = future.result()
            usage = response.get("usage", {})
            prompt_tokens += int(usage.get("prompt_tokens", 0))
            completion_tokens += int(usage.get("completion_tokens", 0))
            latencies.append(latency)
    wall = time.perf_counter() - start
    return {**percentiles(latencies), "mean_ms": statistics.fmean(latencies),
            "request_count": len(latencies), "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens, "wall_seconds": wall,
            "output_tokens_per_second": completion_tokens / wall if wall else None,
            "requested_prompt_tokens_per_request": case.input_tokens,
            "requested_output_tokens_per_request": case.output_tokens,
            "prompt_representation": "exact_token_ids_unique_first_token"}


def parse_case(text: str) -> Case:
    try:
        values = [int(value) for value in text.split(":")]
        if len(values) != 4 or min(values) <= 0:
            raise ValueError
        return Case(*values)
    except ValueError as error:
        raise argparse.ArgumentTypeError("case must be input_tokens:output_tokens:prompts:concurrency") from error


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=("sglang", "vllm"), required=True)
    parser.add_argument("--model", required=True, help="Local model path or already-cached model id")
    parser.add_argument("--served-model-name")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--port", type=int, default=31000)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--memory-fraction", type=float, default=.75)
    parser.add_argument("--prompt-token-id", type=int, default=42,
                        help="Ordinary in-vocabulary token repeated to construct exact-length prompts")
    parser.add_argument("--case", action="append", type=parse_case)
    parser.add_argument("--server-arg", action="append", default=[])
    parser.add_argument(
        "--variant", action="append", default=[],
        help="Run only the named policy variant; repeat to select several. "
             "The default remains every variant.")
    args = parser.parse_args()
    if not args.case:
        args.case = [Case(128, 32, 8, 1), Case(1024, 32, 8, 4)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    selected_variants = variants(args.framework)
    if args.variant:
        known = {variant["id"] for variant in selected_variants}
        unknown = sorted(set(args.variant) - known)
        if unknown:
            parser.error(f"unknown {args.framework} variant(s): {', '.join(unknown)}; "
                         f"known: {', '.join(sorted(known))}")
        selected_variants = [variant for variant in selected_variants
                             if variant["id"] in set(args.variant)]
    for index, variant in enumerate(selected_variants):
        port = args.port + index
        log_path = args.log_dir / f"{args.framework}_{variant['id']}.log"
        command, env = server_command(args, variant, port)
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       text=True, start_new_session=True)
        try:
            ready = wait_ready(port, process, args.startup_timeout)
            if not ready:
                text = log_path.read_text(encoding="utf-8", errors="replace")
                rows.append({"framework": args.framework, "case_id": "server_start", "status": "server_failed",
                             "variant": variant["id"], "requested_policy": variant, "command": command, "log": str(log_path),
                             "exit_code": process.poll(), "log_tail": text[-4000:]})
                continue
            for case in args.case:
                base = {"framework": args.framework, "case_id": case.case_id, "variant": variant["id"],
                        "layout": variant.get("kv_layout") or f"backend={variant.get('backend') or 'auto'},page={variant.get('page_size') or 'auto'}",
                        "page_size": variant.get("page_size") or variant.get("block_size"),
                        "layout_role": variant["layout_role"], "model": args.model, "status": "success",
                        "comparison_scope": variant.get("comparison_scope", "layout_only" if args.framework == "vllm" else "serving_policy_confounded"),
                        "controlled_dimensions": "identical model, requests, process isolation, memory fraction",
                        "log": str(log_path)}
                try:
                    rows.append({**base, **run_requests(args, port, case)})
                except Exception as error:
                    rows.append({**base, "status": "request_failed", "error": repr(error)})
        finally:
            stop(process)
            text = log_path.read_text(encoding="utf-8", errors="replace")
            observed = parse_observed_policy(text, args.framework)
            for row in rows:
                if row.get("variant") == variant["id"]:
                    row.update(observed)
                    if args.framework == "vllm" and variant["kv_layout"] is None and observed["observed_kv_layout"]:
                        row["layout"] = observed["observed_kv_layout"]
    args.output.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"framework": args.framework, "rows": len(rows), "output": str(args.output)}))
    return 0 if any(row.get("status") == "success" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
