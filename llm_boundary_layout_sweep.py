#!/usr/bin/env python3
"""Boundary-layout alternatives for every representative LLM subgraph.

This complements the attention-specific CUDA mechanism suite.  It preserves
logical shapes and values while changing physical strides of input/state/cache
tensors, then times both direct consumption and an explicit repair back to the
native contiguous contract.  Results are PyTorch-runtime evidence, not evidence
for frameworks that were not executed.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
import sys

import torch

# Running this file directly only adds layout_research to sys.path, while the
# shared workload definitions live in its parent directory.
BASELINE_ROOT = Path(__file__).resolve().parent.parent
if str(BASELINE_ROOT) not in sys.path:
    sys.path.insert(0, str(BASELINE_ROOT))

from real_world_workloads import (
    correctness_tolerances,
    correctness_verdict,
    estimate_boundary_input_bytes,
    estimate_case_bytes,
    load_cases,
    make_workload,
    tensor_error_metrics,
)


def alternate_stride(tensor: torch.Tensor) -> tuple[torch.Tensor, str]:
    """Return equal values/shape with a guaranteed different legal stride."""
    if tensor.ndim < 3 or tensor.shape[1] <= 1 or tensor.shape[2] <= 1:
        if tensor.ndim == 0 or tensor.numel() == 0 or tensor.shape[-1] == 0:
            return tensor, "no_legal_stride_transform"
        # A padded last dimension is a real zero-copy consumer view and works
        # for decode tensors whose sequence axis is one.  copy_ initializes
        # the backing allocation once, outside the timed region.
        storage_shape = list(tensor.shape)
        storage_shape[-1] *= 2
        storage = torch.empty(storage_shape, dtype=tensor.dtype, device=tensor.device)
        view = storage[..., ::2]
        view.copy_(tensor)
        return view, "padded_last_dim_stride2"
    order = list(range(tensor.ndim))
    order[1], order[2] = order[2], order[1]
    # The second permutation is a view restoring the logical axis order.  Its
    # storage keeps the alternate physical ordering and is intentionally not
    # contiguous in the original contract.
    return tensor.permute(order).contiguous().permute(order), "axis1_axis2_physical_swap"


def percentiles(fn, warmup: int, iterations: int) -> dict:
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
    torch.cuda.synchronize()
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iterations)]
    with torch.inference_mode():
        for start, end in zip(starts, ends):
            start.record()
            fn()
            end.record()
    torch.cuda.synchronize()
    values = sorted(float(a.elapsed_time(b)) for a, b in zip(starts, ends))
    pick = lambda q: values[round((len(values) - 1) * q)]
    return {"p20_ms": pick(.2), "p50_ms": pick(.5), "p80_ms": pick(.8),
            "iterations": iterations}


def append(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        handle.flush()


def strategy_error_status(error: Exception) -> str:
    """Classify a layout-specific failure without hiding other strategies."""
    text = f"{type(error).__name__}: {error}".lower()
    if ("uncapturedhigherorderop" in text or "higherorderoperator" in text
            or "unsupported" in text or "not supported" in text):
        return "unsupported_alternate_layout"
    if "out of memory" in text or "cuda_error_out_of_memory" in text:
        return "oom_runtime"
    return "strategy_error"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-device-index", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--memory-fraction", type=float, default=.72)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--case-id", action="append")
    parser.add_argument(
        "--extended-rq-counterfactuals", action="store_true",
        help="directly time reuse=1/4/16 for native, strided, repair-once and repair-each",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible.split(",")[0] != str(args.physical_device_index):
        raise RuntimeError(
            f"logical cuda:0 is not physical GPU {args.physical_device_index}: CUDA_VISIBLE_DEVICES={visible!r}")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    cases = load_cases(args.manifest)
    if args.case_id:
        selected = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in selected]
    if args.limit:
        cases = cases[:args.limit]
    # Mamba's associative_scan is captured through TorchDynamo even though
    # this driver invokes the model eagerly.  The default cache limit (8) is
    # too small for a multi-shape suite and manifests as a misleading
    # UncapturedHigherOrderOpError after earlier cases have filled the cache.
    torch._dynamo.config.cache_size_limit = max(
        torch._dynamo.config.cache_size_limit, len(cases) + 32)
    if hasattr(torch._dynamo.config, "accumulated_cache_size_limit"):
        torch._dynamo.config.accumulated_cache_size_limit = max(
            torch._dynamo.config.accumulated_cache_size_limit,
            len(cases) * 2 + 64)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    for index, case in enumerate(cases, 1):
        base = {
            "framework": "PyTorch",
            "evidence_level": "runtime_empirical",
            "experiment": "boundary_layout_sweep",
            "case_id": case["case_id"], "model_id": case["model_id"],
            "model_revision": case["model_revision"], "phase": case["phase"],
            "subgraph": case["structure"], "shape": case["shape"],
            "device_name": props.name, "compute_capability": f"{props.major}.{props.minor}",
            "physical_device_index": args.physical_device_index,
            "torch_version": torch.__version__, "python_version": platform.python_version(),
        }
        estimate = estimate_case_bytes(case)
        alternate_input_bytes = estimate_boundary_input_bytes(case)
        estimated_peak = estimate + alternate_input_bytes
        if estimated_peak > props.total_memory * args.memory_fraction:
            append(args.output, {**base, "status": "oom_preflight",
                                 "estimated_persistent_bytes": estimate,
                                 "estimated_alternate_input_bytes": alternate_input_bytes,
                                 "estimated_peak_bytes": estimated_peak,
                                 "reason": "persistent tensors plus alternate graph inputs exceed memory budget"})
            continue
        workload = None
        alternate = None
        try:
            workload = make_workload(case, "cuda")
            native = workload.inputs
            transformed = tuple(alternate_stride(tensor) for tensor in native)
            alternate = tuple(item[0] for item in transformed)
            transform_kinds = [item[1] for item in transformed]
            changed = [index for index, (before, after) in enumerate(zip(native, alternate))
                       if before.stride() != after.stride()]
            if not changed:
                append(args.output, {**base, "status": "not_applicable_no_layout_change",
                                     "strategy": "alternate_boundary",
                                     "transform_kinds": transform_kinds,
                                     "layout_changed": False,
                                     "changed_tensor_indices": []})
                continue
            with torch.inference_mode():
                reference = workload.model(*native)
            atol, rtol = correctness_tolerances(case)
            variants = [
                ("native_contiguous", native, lambda: workload.model(*native), 0),
                ("alternate_strided_view", alternate,
                 lambda: workload.model(*alternate), 0),
                ("repair_to_contiguous_each_call", alternate,
                 lambda: workload.model(*(tensor.contiguous() for tensor in alternate)),
                 sum(tensor.numel() * tensor.element_size() for tensor in alternate if not tensor.is_contiguous())),
            ]
            for strategy, tensors, fn, repaired_bytes in variants:
                row = {
                    **base, "strategy": strategy,
                    "layout": ("framework_native_contiguous" if strategy == "native_contiguous"
                               else "explicit_contiguous_repair_each_call"
                               if strategy == "repair_to_contiguous_each_call"
                               else "logical_shape_preserved_strided_view"),
                    "input_shapes": [list(t.shape) for t in tensors],
                    "input_strides": [list(t.stride()) for t in tensors],
                    "input_contiguous": [t.is_contiguous() for t in tensors],
                    "transform_kinds": transform_kinds,
                    "layout_changed": strategy != "native_contiguous",
                    "changed_tensor_indices": changed if strategy != "native_contiguous" else [],
                    "explicit_repair_bytes": repaired_bytes,
                    "correctness_atol": atol, "correctness_rtol": rtol,
                }
                try:
                    with torch.inference_mode():
                        actual = fn()
                    correctness = tensor_error_metrics(actual, reference, atol=atol, rtol=rtol)
                    passed, correctness_rule = correctness_verdict(case, correctness)
                    row.update(correctness)
                    row.update(
                        elementwise_correct=correctness["correct"],
                        correct=passed,
                        correctness_passed=passed,
                        correctness_rule=correctness_rule,
                        correctness_rule_origin=
                            "frozen_after_representative_smoke_before_full_640",
                    )
                    if not passed:
                        row.update(status="numerical_mismatch")
                    else:
                        row.update(status="success",
                                   **percentiles(fn, args.warmup, args.iterations))
                except Exception as error:
                    # A direct alternate layout can be unsupported by a graph
                    # primitive (notably higher-order scans).  Persist that
                    # fact for this strategy, then continue to the explicit
                    # repair strategy instead of dropping all three rows.
                    row.update(status=strategy_error_status(error),
                               error=repr(error)[:4000])
                append(args.output, row)
            if args.extended_rq_counterfactuals:
                repair_bytes = sum(
                    tensor.numel() * tensor.element_size()
                    for tensor in alternate if not tensor.is_contiguous())

                def repeated_native(reuse: int):
                    result = None
                    for _ in range(reuse):
                        result = workload.model(*native)
                    return result

                def repeated_alternate(reuse: int):
                    result = None
                    for _ in range(reuse):
                        result = workload.model(*alternate)
                    return result

                def repair_once_then_reuse(reuse: int):
                    repaired = tuple(tensor.contiguous() for tensor in alternate)
                    result = None
                    for _ in range(reuse):
                        result = workload.model(*repaired)
                    return result

                def repair_every_use(reuse: int):
                    result = None
                    for _ in range(reuse):
                        result = workload.model(
                            *(tensor.contiguous() for tensor in alternate))
                    return result

                for reuse in (1, 4, 16):
                    reuse_variants = (
                        ("native_reuse", lambda reuse=reuse: repeated_native(reuse), 0),
                        ("alternate_direct_reuse", lambda reuse=reuse: repeated_alternate(reuse), 0),
                        ("repair_once_then_reuse", lambda reuse=reuse: repair_once_then_reuse(reuse),
                         repair_bytes),
                        ("repair_every_use", lambda reuse=reuse: repair_every_use(reuse),
                         repair_bytes * reuse),
                    )
                    for strategy, fn, materialized_bytes in reuse_variants:
                        row = {
                            **base,
                            "experiment": "boundary_reuse_counterfactual",
                            "strategy": strategy,
                            "reuse_count": reuse,
                            "layout": ("framework_native_contiguous" if strategy == "native_reuse"
                                       else "logical_shape_preserved_strided_view"
                                       if strategy == "alternate_direct_reuse"
                                       else "explicit_contiguous_repair"),
                            "transform_kinds": transform_kinds,
                            "changed_tensor_indices": changed,
                            "explicit_repair_bytes": materialized_bytes,
                            "conversion_placement": ("none" if materialized_bytes == 0
                                                     else "once_per_pipeline"
                                                     if strategy == "repair_once_then_reuse"
                                                     else "once_per_consumer"),
                            "correctness_atol": atol,
                            "correctness_rtol": rtol,
                        }
                        try:
                            with torch.inference_mode():
                                actual = fn()
                            correctness = tensor_error_metrics(actual, reference, atol=atol, rtol=rtol)
                            passed, correctness_rule = correctness_verdict(case, correctness)
                            row.update(correctness)
                            row.update(correct=passed, correctness_passed=passed,
                                       correctness_rule=correctness_rule)
                            if passed:
                                row.update(status="success",
                                           **percentiles(fn, args.warmup, args.iterations))
                            else:
                                row["status"] = "numerical_mismatch"
                        except Exception as error:
                            row.update(status=strategy_error_status(error),
                                       error=repr(error)[:4000])
                        append(args.output, row)
            statuses = []
            for strategy in ("native_contiguous", "alternate_strided_view",
                             "repair_to_contiguous_each_call"):
                # The rows are already durable; this concise progress message
                # deliberately avoids re-reading the growing result file.
                statuses.append(strategy)
            print(f"[{index}/{len(cases)}] {case['case_id']} strategies_recorded=3", flush=True)
        except Exception as error:
            status = strategy_error_status(error)
            if status == "unsupported_alternate_layout":
                status = "workload_unsupported"
            append(args.output, {**base, "status": status, "error": repr(error)[:4000]})
            print(f"[{index}/{len(cases)}] {case['case_id']} error: {error!r}", flush=True)
        finally:
            del alternate, workload
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
