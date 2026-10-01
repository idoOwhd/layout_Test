#!/usr/bin/env python3
"""TVM-native multi-shape experiments for RQ3/RQ7/RQ8/RQ9/RQ10.

Direct rows are timed TVM CUDA kernels. ``measured_cost_model`` rows only add
measured primitive medians and are never presented as a directly timed fused
pipeline.  This distinction is intentional and carried into the extractor.
"""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
from pathlib import Path

import numpy as np
import tvm
from tvm import te


CUDA_TARGET = {"kind": "cuda", "arch": os.environ.get("TVM_CUDA_ARCH", "sm_86")}


def load_manifest(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    return value["cases"] if isinstance(value, dict) else value


def projected_boundary_shape(case: dict, tile_rows: int, tile_cols: int) -> tuple[int, int]:
    """Project an LLM boundary to a padded matrix without hiding padding."""
    shape = case["shape"]
    rows = int(shape.get("batch", 1)) * int(shape.get("query_length", 1))
    cols = int(shape.get("hidden_size", shape.get("head_dim", tile_cols)))
    rows = max(tile_rows, ((rows + tile_rows - 1) // tile_rows) * tile_rows)
    cols = max(tile_cols, ((cols + tile_cols - 1) // tile_cols) * tile_cols)
    return rows, cols


def parse_shape(text: str) -> tuple[int, int]:
    try:
        rows, cols = (int(x) for x in text.lower().split("x"))
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("shape must be ROWSxCOLS") from exc
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, cols


def logical_offset(layout: str, row, col, rows: int, cols: int, tr: int, tc: int):
    if layout == "row_major":
        return row * cols + col
    return (((row // tr) * (cols // tc) + col // tc) * tr + row % tr) * tc + col % tc


def physical(logical: np.ndarray, layout: str, tr: int, tc: int) -> np.ndarray:
    if layout == "row_major":
        return np.ascontiguousarray(logical).reshape(-1)
    rows, cols = logical.shape
    return np.ascontiguousarray(
        logical.reshape(rows // tr, tr, cols // tc, tc).transpose(0, 2, 1, 3)
    ).reshape(-1)


def compile_1d(source, result, factor: int = 128):
    prim = te.create_prim_func([source, result])
    schedule = tvm.s_tir.Schedule(prim)
    block = schedule.get_sblock(result.op.name)
    spatial = schedule.get_loops(block)[0]
    outer, inner = schedule.split(spatial, factors=[None, factor])
    schedule.bind(outer, "blockIdx.x")
    schedule.bind(inner, "threadIdx.x")
    return tvm.compile(schedule.mod, target=CUDA_TARGET)


def build_reduce(layout: str, rows: int, cols: int, tr: int, tc: int,
                 square: bool = False, axis: str = "row"):
    source = te.placeholder((rows * cols,), "float16", name="A")
    extent = cols if axis == "row" else rows
    output_extent = rows if axis == "row" else cols
    k = te.reduce_axis((0, extent), "k")
    def value(r, c):
        item = source[logical_offset(layout, r, c, rows, cols, tr, tc)]
        return (item * item).astype("float32") if square else item.astype("float32")
    result = te.compute(
        (output_extent,),
        lambda outer: te.sum(value(outer, k) if axis == "row" else value(k, outer), axis=k),
        name="B")
    return compile_1d(source, result)


def build_unary(size: int, square: bool):
    source = te.placeholder((size,), "float16", name="A")
    result = te.compute((size,), lambda i: source[i] * source[i] if square else source[i], name="B")
    return compile_1d(source, result, 256)


def build_row_to_tiled(rows: int, cols: int, tr: int, tc: int):
    source = te.placeholder((rows * cols,), "float16", name="A")
    co_count = cols // tc
    def load(index):
        ci = index % tc
        q1 = index // tc
        ri = q1 % tr
        q2 = q1 // tr
        co = q2 % co_count
        ro = q2 // co_count
        return source[(ro * tr + ri) * cols + co * tc + ci]
    result = te.compute((rows * cols,), load, name="B")
    return compile_1d(source, result, 256)


def build_multi_consumer(layout: str, rows: int, cols: int, tr: int, tc: int,
                         row_fanout: int, column_fanout: int):
    """One TVM module containing both row- and column-oriented consumers."""
    source = te.placeholder((rows * cols,), "float16", name="A")
    row_k = te.reduce_axis((0, cols), "row_k")
    col_k = te.reduce_axis((0, rows), "col_k")
    row_out = te.compute(
        (row_fanout, rows),
        lambda _consumer, row: te.sum(
            source[logical_offset(layout, row, row_k, rows, cols, tr, tc)].astype("float32"),
            axis=row_k), name="row_consumers")
    col_out = te.compute(
        (column_fanout, cols),
        lambda _consumer, col: te.sum(
            source[logical_offset(layout, col_k, col, rows, cols, tr, tc)].astype("float32"),
            axis=col_k), name="column_consumers")
    prim = te.create_prim_func([source, row_out, col_out])
    schedule = tvm.s_tir.Schedule(prim)
    for name in ("row_consumers", "column_consumers"):
        block = schedule.get_sblock(name)
        loops = schedule.get_loops(block)
        spatial = schedule.fuse(loops[0], loops[1])
        outer, inner = schedule.split(spatial, factors=[None, 128])
        schedule.bind(outer, "blockIdx.x")
        schedule.bind(inner, "threadIdx.x")
    return tvm.compile(schedule.mod, target=CUDA_TARGET)


def timing(module, dev, args: list, number: int, repeat: int) -> dict[str, float]:
    module(*args)
    dev.sync()
    runtime_module = module.jit() if hasattr(module, "jit") else module
    values = sorted(float(x * 1000) for x in
                    runtime_module.time_evaluator("main", dev, number=number, repeat=repeat)(*args).results)
    def pick(q: float) -> float:
        return values[round((len(values) - 1) * q)]
    return {"p20_ms": pick(.2), "p50_ms": pick(.5), "p80_ms": pick(.8),
            "min_ms": min(values), "max_ms": max(values)}


def direct_row(case_id: str, experiment: str, strategy: str, timing_row: dict,
               correct: bool, max_error: float, rows: int, cols: int) -> dict:
    return {"framework": "TVM", "case_id": case_id, "experiment": experiment,
            "strategy": strategy, "evidence_level": "native_runtime_counterfactual",
            "rows": rows, "cols": cols, "correct": correct, "max_abs_error": max_error,
            **timing_row}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dump-dir", type=Path)
    parser.add_argument("--shape", action="append", type=parse_shape)
    parser.add_argument("--manifest", type=Path,
                        help="run unique padded boundary shapes and map results back to every manifest case")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tile-rows", type=int, default=16)
    parser.add_argument("--tile-cols", type=int, default=16)
    parser.add_argument("--number", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=7)
    args = parser.parse_args()
    tr, tc = args.tile_rows, args.tile_cols
    manifest_by_shape: dict[tuple[int, int], list[dict]] = {}
    if args.manifest:
        manifest_cases = load_manifest(args.manifest)
        if args.limit:
            manifest_cases = manifest_cases[:args.limit]
        for case in manifest_cases:
            manifest_by_shape.setdefault(projected_boundary_shape(case, tr, tc), []).append(case)
        shapes = sorted(manifest_by_shape)
    else:
        shapes = args.shape or [(512, 512), (1024, 4096), (4096, 1024), (4096, 4096)]
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    init_code = int(driver.cuInit(0))
    if init_code != 0:
        raise RuntimeError(f"CUDA driver initialization failed: cuInit={init_code}")
    dev = tvm.cuda(0)
    if not dev.exist:
        raise RuntimeError("TVM CUDA device unavailable")
    all_rows: list[dict] = []
    for rows, cols in shapes:
        shape_row_start = len(all_rows)
        if rows % tr or cols % tc:
            raise ValueError(f"{rows}x{cols} is not divisible by tile {tr}x{tc}")
        case_id = f"llm_boundary_{rows}x{cols}"
        rng = np.random.default_rng(rows * 100003 + cols)
        logical = rng.normal(0, .25, size=(rows, cols)).astype("float16")
        row_host = physical(logical, "row_major", tr, tc)
        tiled_host = physical(logical, "tiled_16x16", tr, tc)
        row_dev = tvm.runtime.tensor(row_host, dev)
        tiled_dev = tvm.runtime.tensor(tiled_host, dev)
        temp_dev = tvm.runtime.empty((rows * cols,), "float16", dev)
        square_reference = (logical * logical).astype("float32").sum(1)

        reduce_times = {"row": {}, "column": {}}
        modules = {}
        for axis, reference in (("row", logical.astype("float32").sum(1)),
                                ("column", logical.astype("float32").sum(0))):
            output_extent = rows if axis == "row" else cols
            output = tvm.runtime.empty((output_extent,), "float32", dev)
            for layout, input_dev in (("row_major", row_dev), ("tiled_16x16", tiled_dev)):
                module = build_reduce(layout, rows, cols, tr, tc, axis=axis)
                modules[f"reduce_{axis}_{layout}"] = module
                measured = timing(module, dev, [input_dev, output], args.number, args.repeat)
                module(input_dev, output); dev.sync()
                error = float(np.max(np.abs(output.numpy() - reference)))
                reduce_times[axis][layout] = measured["p50_ms"]
                all_rows.append(direct_row(f"{case_id}_{axis}_consumer", "consumer_scan",
                                           layout, measured,
                                           error < max(1e-2, (cols if axis == 'row' else rows) * 2e-4),
                                           error, rows, cols))

        # RQ2: actual multi-consumer modules, not a weighted sum of primitive
        # medians.  Row and column reducers impose opposing layout locality.
        for row_fanout, column_fanout in ((1, 4), (1, 1), (4, 1)):
            for layout, input_dev in (("row_major", row_dev),
                                      ("tiled_16x16", tiled_dev)):
                module = build_multi_consumer(layout, rows, cols, tr, tc,
                                              row_fanout, column_fanout)
                modules[f"multi_{row_fanout}_{column_fanout}_{layout}"] = module
                row_output = tvm.runtime.empty((row_fanout, rows), "float32", dev)
                col_output = tvm.runtime.empty((column_fanout, cols), "float32", dev)
                measured = timing(module, dev, [input_dev, row_output, col_output],
                                  args.number, args.repeat)
                module(input_dev, row_output, col_output); dev.sync()
                row_error = float(np.max(np.abs(
                    row_output.numpy()[0] - logical.astype("float32").sum(1))))
                col_error = float(np.max(np.abs(
                    col_output.numpy()[0] - logical.astype("float32").sum(0))))
                tolerance = max(1e-2, max(rows, cols) * 2e-4)
                all_rows.append({
                    **direct_row(case_id, "weighted_multi_consumer_pipeline", layout,
                                 measured, max(row_error, col_error) < tolerance,
                                 max(row_error, col_error), rows, cols),
                    "fanout_row": row_fanout,
                    "fanout_column": column_fanout,
                    "evidence_level": "native_runtime_multi_consumer_module",
                    "producer_materialization_timed": False,
                })

        convert = build_row_to_tiled(rows, cols, tr, tc)
        modules["row_to_tiled"] = convert
        convert_t = timing(convert, dev, [row_dev, temp_dev], args.number, args.repeat)
        convert(row_dev, temp_dev); dev.sync()
        conversion_error = float(np.max(np.abs(temp_dev.numpy() - tiled_host)))
        all_rows.append(direct_row(case_id, "primitive", "row_to_tiled", convert_t,
                                   conversion_error == 0.0, conversion_error, rows, cols))

        square = build_unary(rows * cols, square=True)
        modules["square"] = square
        square_t = timing(square, dev, [row_dev, temp_dev], args.number, args.repeat)
        all_rows.append(direct_row(case_id, "primitive", "square_materialize", square_t,
                                   True, 0.0, rows, cols))

        fused = build_reduce("row_major", rows, cols, tr, tc, square=True)
        modules["fused_square_reduce"] = fused
        fused_out = tvm.runtime.empty((rows,), "float32", dev)
        fused_t = timing(fused, dev, [row_dev, fused_out], args.number, args.repeat)
        fused(row_dev, fused_out); dev.sync()
        fused_error = float(np.max(np.abs(fused_out.numpy() - square_reference)))
        all_rows.append(direct_row(case_id, "fusion_order", "fused_square_reduce", fused_t,
                                   fused_error < max(1e-2, cols * 2e-4), fused_error, rows, cols))

        # These totals compose medians from directly timed native TVM kernels.
        # They deliberately retain their measured_cost_model label.
        for reuse in (1, 4, 16, 64):
            strategies = {
                "keep_row_major": reuse * reduce_times["column"]["row_major"],
                "convert_once_then_tiled": convert_t["p50_ms"] + reuse * reduce_times["column"]["tiled_16x16"],
                "convert_each_use_then_tiled": reuse * (convert_t["p50_ms"] + reduce_times["column"]["tiled_16x16"]),
            }
            for strategy, total in strategies.items():
                all_rows.append({"framework": "TVM", "case_id": case_id,
                    "experiment": "conversion_reuse", "strategy": strategy, "reuse": reuse,
                    "p50_ms": total, "correct": conversion_error == 0.0,
                    "evidence_level": "native_measured_cost_model", "rows": rows, "cols": cols})

        separate = square_t["p50_ms"] + reduce_times["row"]["row_major"]
        convert_first = convert_t["p50_ms"] + square_t["p50_ms"] + reduce_times["row"]["tiled_16x16"]
        for strategy, total in (
            ("square_then_reduce_materialized", separate),
            ("convert_then_square_then_tiled_reduce", convert_first),
        ):
            all_rows.append({"framework": "TVM", "case_id": case_id,
                "experiment": "fusion_order", "strategy": strategy, "p50_ms": total,
                "correct": conversion_error == 0.0, "evidence_level": "native_measured_cost_model",
                "rows": rows, "cols": cols})

        if args.dump_dir:
            args.dump_dir.mkdir(parents=True, exist_ok=True)
            for name, module in modules.items():
                runtime_module = module.jit() if hasattr(module, "jit") else module
                imports = list(runtime_module.imports)
                source = imports[0].inspect_source() if imports else ""
                (args.dump_dir / f"{case_id}_{name}.ptx").write_text(source, encoding="utf-8")

        if manifest_by_shape:
            templates = all_rows[shape_row_start:]
            del all_rows[shape_row_start:]
            for source_case in manifest_by_shape[(rows, cols)]:
                original_rows = (int(source_case["shape"].get("batch", 1)) *
                                 int(source_case["shape"].get("query_length", 1)))
                original_cols = int(source_case["shape"].get(
                    "hidden_size", source_case["shape"].get("head_dim", tc)))
                for template in templates:
                    all_rows.append({**template,
                        "manifest_case_id": source_case["case_id"],
                        "model_id": source_case["model_id"],
                        "structure": source_case["structure"],
                        "phase": source_case["phase"],
                        "logical_rows": original_rows, "logical_cols": original_cols,
                        "padded_rows": rows, "padded_cols": cols,
                        "measurement_reused_for_equal_projected_shape": len(
                            manifest_by_shape[(rows, cols)]) > 1})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in all_rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps({"output": str(args.output), "shapes": shapes, "rows": len(all_rows),
                      "manifest_cases": sum(map(len, manifest_by_shape.values()))}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
