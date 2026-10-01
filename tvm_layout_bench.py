#!/usr/bin/env python3
"""TVM explicit IndexMap-style storage experiment for a row reduction."""

from __future__ import annotations

import argparse
import ctypes
import csv
import json
import os
import platform
from pathlib import Path

import numpy as np
import tvm
from tvm import te


def offset(layout, row, col, rows, cols, tr, tc):
    if layout == "row_major":
        return row * cols + col
    # Physical [Ro, Co, ri, ci], the same mapping used by the CUDA boundary test.
    return (((row // tr) * (cols // tc) + col // tc) * tr + row % tr) * tc + col % tc


def build(layout, rows, cols, tr, tc):
    a = te.placeholder((rows * cols,), "float16", name="A")
    k = te.reduce_axis((0, cols), "k")
    b = te.compute((rows,), lambda r: te.sum(a[offset(layout, r, k, rows, cols, tr, tc)].astype("float32"), axis=k), name="B")
    s = tvm.s_tir.Schedule(te.create_prim_func([a, b]))
    block = s.get_sblock("B")
    bo, ti = s.split(s.get_loops(block)[0], factors=[None, 128])
    s.bind(bo, "blockIdx.x")
    s.bind(ti, "threadIdx.x")
    return tvm.compile(s.mod, target={"kind": "cuda", "arch": os.environ.get("TVM_CUDA_ARCH", "sm_86")})


def physical(logical, layout, tr, tc):
    if layout == "row_major": return np.ascontiguousarray(logical).reshape(-1)
    r, c = logical.shape
    return np.ascontiguousarray(logical.reshape(r // tr, tr, c // tc, tc).transpose(0, 2, 1, 3)).reshape(-1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--rows", type=int, default=8192)
    p.add_argument("--cols", type=int, default=4096)
    p.add_argument("--tile-rows", type=int, default=16)
    p.add_argument("--tile-cols", type=int, default=16)
    p.add_argument("--number", type=int, default=20)
    p.add_argument("--repeat", type=int, default=5)
    p.add_argument("--dump-dir", type=Path)
    a = p.parse_args()
    if a.rows % a.tile_rows or a.cols % a.tile_cols: raise ValueError("shape must divide tile")
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    init_code = int(driver.cuInit(0))
    if init_code != 0: raise RuntimeError(f"CUDA driver initialization failed: cuInit={init_code}")
    dev = tvm.cuda(0)
    if not dev.exist: raise RuntimeError("TVM CUDA device unavailable")
    rng = np.random.default_rng(7)
    logical = rng.normal(size=(a.rows, a.cols)).astype("float16")
    reference = logical.astype("float32").sum(1)
    rows = []
    for layout in ("row_major", "tiled_16x16"):
        module = build(layout, a.rows, a.cols, a.tile_rows, a.tile_cols)
        if a.dump_dir:
            a.dump_dir.mkdir(parents=True, exist_ok=True)
            runtime_module = module.jit() if hasattr(module, "jit") else module
            imports = list(runtime_module.imports)
            imported = imports[0].inspect_source() if imports else ""
            (a.dump_dir / f"tvm_{layout}.ptx").write_text(imported, encoding="utf-8")
        x = tvm.runtime.tensor(physical(logical, layout, a.tile_rows, a.tile_cols), dev)
        y = tvm.runtime.empty((a.rows,), "float32", dev)
        module(x, y); dev.sync()
        error = float(np.max(np.abs(y.numpy() - reference)))
        runtime_module = module.jit() if hasattr(module, "jit") else module
        timing = runtime_module.time_evaluator("main", dev, number=a.number, repeat=a.repeat)(x, y)
        samples = sorted(float(v * 1000) for v in timing.results)
        rows.append({"framework": "tvm", "case_id": "norm_softmax_boundary", "layout": layout,
                     "rows": a.rows, "cols": a.cols, "tile_rows": a.tile_rows, "tile_cols": a.tile_cols,
                     "p50_ms": samples[len(samples) // 2], "min_ms": min(samples),
                     "max_abs_error": error, "correct": error < 1.0,
                     "tvm_version": tvm.__version__, "python_version": platform.python_version(),
                     "cuda_device": dev.index})
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__": main()
