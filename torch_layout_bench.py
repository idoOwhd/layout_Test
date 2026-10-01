#!/usr/bin/env python3
"""Framework-level paired layouts for SwiGLU and diffusion Conv2d."""

from __future__ import annotations

import argparse
import csv
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


def q(values, p):
    values = sorted(values)
    return values[round((len(values) - 1) * p)]


def bench(fn, warmup, iterations):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); values = []
    for _ in range(iterations):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True)
        a.record(); fn(); b.record(); b.synchronize(); values.append(float(a.elapsed_time(b)))
    return {"p20_ms": q(values, .2), "p50_ms": q(values, .5), "p80_ms": q(values, .8)}


def run_swiglu(m, h, i, warmup, iterations):
    # Smaller defaults can be selected in --quick.  The full shapes correspond
    # to a Llama-3-class FFN and need roughly 0.35 GiB of weights.
    x = torch.randn((m, h), device="cuda", dtype=torch.float16)
    wg = torch.randn((h, i), device="cuda", dtype=torch.float16)
    wu = torch.randn((h, i), device="cuda", dtype=torch.float16)
    wp = torch.cat((wg, wu), dim=1).contiguous()
    separate = lambda: F.silu(x @ wg) * (x @ wu)
    def packed():
        gate, up = (x @ wp).chunk(2, dim=-1)
        return F.silu(gate) * up
    reference = separate(); actual = packed(); torch.cuda.synchronize()
    error = float((reference - actual).abs().max().item())
    rows = []
    for layout, fn in (("separate_gate_up", separate), ("packed_gate_up", packed)):
        rows.append({"case_id": f"swiglu_m{m}_h{h}_i{i}", "family": "swiglu",
                     "framework": "pytorch", "layout": layout, "m": m, "hidden": h,
                     "intermediate": i, "max_abs_error_to_paired": error,
                     "x_stride": list(x.stride()),
                     "weight_stride": list((wp if layout == "packed_gate_up" else wg).stride()),
                     **bench(fn, warmup, iterations)})
    return rows


def run_conv(n, c, h, w, out_c, warmup, iterations):
    torch.backends.cudnn.benchmark = True
    x = torch.randn((n, c, h, w), device="cuda", dtype=torch.float16)
    weight = torch.randn((out_c, c, 3, 3), device="cuda", dtype=torch.float16)
    x_cl = x.contiguous(memory_format=torch.channels_last)
    w_cl = weight.contiguous(memory_format=torch.channels_last)
    fns = {
        "NCHW": lambda: F.conv2d(x, weight, padding=1),
        "NHWC_channels_last": lambda: F.conv2d(x_cl, w_cl, padding=1),
    }
    ref = fns["NCHW"](); actual = fns["NHWC_channels_last"](); torch.cuda.synchronize()
    error = float((ref - actual).abs().max().item())
    rows = []
    for layout, fn in fns.items():
        rows.append({"case_id": f"diffusion_conv_n{n}_c{c}_h{h}_w{w}_o{out_c}",
                     "family": "diffusion_conv", "framework": "pytorch_cudnn",
                     "layout": layout, "n": n, "channels": c, "height": h, "width": w,
                     "out_channels": out_c, "max_abs_error_to_paired": error,
                     "input_stride": list((x if layout == "NCHW" else x_cl).stride()),
                     "weight_stride": list((weight if layout == "NCHW" else w_cl).stride()),
                     **bench(fn, warmup, iterations)})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--iterations", type=int, default=50)
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    if a.quick:
        rows = run_swiglu(32, 1024, 3072, a.warmup, a.iterations)
        rows += run_conv(1, 128, 32, 32, 128, a.warmup, a.iterations)
    else:
        rows = run_swiglu(32, 4096, 14336, a.warmup, a.iterations)
        rows += run_swiglu(2048, 4096, 14336, a.warmup, a.iterations)
        rows += run_conv(1, 320, 64, 64, 320, a.warmup, a.iterations)
    props = torch.cuda.get_device_properties(0)
    for row in rows:
        row.update(device_name=props.name, compute_capability=f"{props.major}.{props.minor}",
                   torch_version=torch.__version__, cuda_runtime=torch.version.cuda,
                   python_version=platform.python_version())
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    print(f"wrote {len(rows)} PyTorch layout measurements to {a.output}")


if __name__ == "__main__": main()
