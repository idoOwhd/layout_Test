#!/usr/bin/env python3
"""Reference paired-layout benchmarks for the modern vision shape manifest.

This runner establishes identical numerical contracts for common image/video
motifs.  It is a PyTorch reference, not a substitute for native CUTLASS,
Triton, TVM or Hexcute adapters; that distinction is explicit in every row.
"""

from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path

import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
DEFAULT_MANIFEST = HERE.parent / "vision_shapes" / "vision_common_subgraph_manifest.json"


def quantile(values, p):
    values = sorted(values); return values[round((len(values) - 1) * p)]


def bench(fn, warmup, iterations):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); values = []
    for _ in range(iterations):
        a, b = torch.cuda.Event(True), torch.cuda.Event(True); a.record(); fn(); b.record(); b.synchronize(); values.append(float(a.elapsed_time(b)))
    return {"p20_ms": quantile(values, .2), "p50_ms": quantile(values, .5), "p80_ms": quantile(values, .8), "iterations": iterations}


def tensor_desc(x):
    return {"shape": list(x.shape), "stride": list(x.stride()), "dtype": str(x.dtype),
            "contiguous": x.is_contiguous(), "channels_last": x.ndim == 4 and x.is_contiguous(memory_format=torch.channels_last),
            "channels_last_3d": x.ndim == 5 and x.is_contiguous(memory_format=torch.channels_last_3d),
            "storage_offset": x.storage_offset(), "data_ptr_alignment_128": x.data_ptr() % 128}


def timing_rows(case, variants, warmup, iterations, reference_name):
    outputs = {}; rows = []
    for name, (fn, tensors) in variants.items():
        outputs[name] = fn(); torch.cuda.synchronize()
    reference = outputs[reference_name]
    for name, (fn, tensors) in variants.items():
        error = float((outputs[name].float() - reference.float()).abs().max().item())
        rows.append({"case_id": case["case_id"], "domain": case["domain"], "motif": case["motif"],
                     "framework": "pytorch_reference", "layout": name, "layout_role": "native" if name == reference_name else "alternative",
                     "correct": error < 0.5, "max_abs_error": error, "tensor_layouts": {k: tensor_desc(v) for k, v in tensors.items()},
                     **bench(fn, warmup, iterations)})
    return rows


def token_views(shape):
    b, s, d = shape["batch"], shape["tokens"], shape["hidden"]
    logical = torch.randn((b, s, d), device="cuda", dtype=torch.float16)
    channel = logical.transpose(1, 2).contiguous()
    return logical, channel


def run_ffn(case, warmup, iterations):
    s = case["shape"]; x, xc = token_views(s); d = s["hidden"]; inter = s["intermediate"]
    w1 = torch.randn((d, inter), device="cuda", dtype=torch.float16) / math.sqrt(d)
    w2 = torch.randn((inter, d), device="cuda", dtype=torch.float16) / math.sqrt(inter)
    f = lambda z: z + F.gelu(z @ w1) @ w2
    return timing_rows(case, {"token_major": (lambda: f(x), {"x": x, "w1": w1}),
                              "channel_major_persistent": (lambda: f(xc.transpose(1, 2)), {"x_physical": xc, "w1": w1})}, warmup, iterations, "token_major")


def run_modulation(case, warmup, iterations):
    s = case["shape"]; x, xc = token_views(s); d = s["hidden"]
    scale = torch.randn((s["batch"], 1, d), device="cuda", dtype=torch.float16); shift = torch.randn_like(scale); gate = torch.sigmoid(torch.randn_like(scale))
    f = lambda z: (F.layer_norm(z, (d,)) * (1 + scale) + shift) * gate
    return timing_rows(case, {"token_major": (lambda: f(x), {"x": x, "scale": scale}),
                              "channel_major_persistent": (lambda: f(xc.transpose(1, 2)), {"x_physical": xc, "scale": scale})}, warmup, iterations, "token_major")


def qkv(s):
    b, n, h, kh, d = s["batch"], s["tokens"], s["q_heads"], s.get("kv_heads", s["q_heads"]), s["head_dim"]
    q = torch.randn((b, n, h, d), device="cuda", dtype=torch.float16)
    k = torch.randn((b, n, kh, d), device="cuda", dtype=torch.float16); v = torch.randn_like(k)
    return q, k, v


def run_attention(case, warmup, iterations, cross=False):
    s = case["shape"]; q, k, v = qkv(s)
    if cross:
        text = s["text_tokens"]; k = k[:, :text].contiguous(); v = v[:, :text].contiguous()
    qh, kh, vh = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(), v.transpose(1, 2).contiguous()
    use_gqa = s.get("kv_heads", s["q_heads"]) != s["q_heads"]
    f = lambda a, b, c: F.scaled_dot_product_attention(a, b, c, enable_gqa=use_gqa)
    return timing_rows(case, {"BSHD_token_major": (lambda: f(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)), {"q": q, "k": k, "v": v}),
                              "BHSD_head_major": (lambda: f(qh, kh, vh), {"q": qh, "k": kh, "v": vh})}, warmup, iterations, "BSHD_token_major")


def run_rope_or_qknorm(case, warmup, iterations):
    s = case["shape"]; q, k, _ = qkv(s); qh, kh = q.transpose(1, 2).contiguous(), k.transpose(1, 2).contiguous(); d = s["head_dim"]
    if case["motif"] == "qk_norm":
        one = lambda z: F.normalize(z.float(), dim=-1).half()
    else:
        cos = torch.randn((d // 2,), device="cuda", dtype=torch.float16); sin = torch.randn_like(cos)
        def one(z):
            left, right = z[..., :d // 2], z[..., d // 2:d]
            return torch.cat((left * cos - right * sin, right * cos + left * sin), -1)
    logical = lambda a, b: torch.cat((one(a).reshape(-1), one(b).reshape(-1)))
    return timing_rows(case, {"BSHD_token_head": (lambda: logical(q, k), {"q": q, "k": k}),
                              "BHSD_head_token": (lambda: logical(qh.transpose(1, 2), kh.transpose(1, 2)), {"q": qh, "k": kh})}, warmup, iterations, "BSHD_token_head")


def conv_variants(case):
    s = case["shape"]
    if "frames" in s:
        patch = s.get("patch", [1, 1, 1]); x = torch.randn((s["batch"], s["channels"], s["frames"], s["height"], s["width"]), device="cuda", dtype=torch.float16)
        out_c = min(int(s.get("hidden", s.get("out_channels", 128))), 512); w = torch.randn((out_c, s["channels"], *patch), device="cuda", dtype=torch.float16)
        xc = x.contiguous(memory_format=torch.channels_last_3d); wc = w.contiguous(memory_format=torch.channels_last_3d)
        return {"NCTHW": (lambda: F.conv3d(x, w, stride=patch), {"x": x, "w": w}),
                "NDHWC_channels_last_3d": (lambda: F.conv3d(xc, wc, stride=patch), {"x": xc, "w": wc})}, "NCTHW"
    patch = s.get("patch", [1, 1]); x = torch.randn((s["batch"], s["channels"], s["height"], s["width"]), device="cuda", dtype=torch.float16)
    out_c = min(int(s.get("hidden", s.get("out_channels", 128))), 512); w = torch.randn((out_c, s["channels"], *patch), device="cuda", dtype=torch.float16)
    xc = x.contiguous(memory_format=torch.channels_last); wc = w.contiguous(memory_format=torch.channels_last)
    return {"NCHW": (lambda: F.conv2d(x, w, stride=patch), {"x": x, "w": w}),
            "NHWC_channels_last": (lambda: F.conv2d(xc, wc, stride=patch), {"x": xc, "w": wc})}, "NCHW"


def run_structured(case, warmup, iterations):
    s = case["shape"]; q, k, v = qkv(s); block = s["block_tokens"]; n = s["tokens"] // block
    q = q[:, :n * block]; k = k[:, :n * block]; v = v[:, :n * block]
    qb = q.view(s["batch"], n, block, s["q_heads"], s["head_dim"]).permute(0, 1, 3, 2, 4).contiguous()
    kb = k.view_as(q).view(s["batch"], n, block, s["q_heads"], s["head_dim"]).permute(0, 1, 3, 2, 4).contiguous(); vb = v.view_as(q).view(s["batch"], n, block, s["q_heads"], s["head_dim"]).permute(0, 1, 3, 2, 4).contiguous()
    dense_view = lambda: F.scaled_dot_product_attention(q.view(s["batch"] * n, block, s["q_heads"], s["head_dim"]).transpose(1, 2), k.view(s["batch"] * n, block, s["q_heads"], s["head_dim"]).transpose(1, 2), v.view(s["batch"] * n, block, s["q_heads"], s["head_dim"]).transpose(1, 2))
    blocked = lambda: F.scaled_dot_product_attention(qb.flatten(0, 1), kb.flatten(0, 1), vb.flatten(0, 1))
    return timing_rows(case, {"token_major_block_views": (dense_view, {"q": q}), "block_major_persistent": (blocked, {"q": qb})}, warmup, iterations, "token_major_block_views")


def estimate_bytes(case):
    s = case["shape"]; b = s.get("batch", 1); d = s.get("hidden", 0); n = s.get("tokens", 0)
    total = b * n * d * 2 * 8
    if case["motif"] in {"ffn", "dit_block"}: total += d * s.get("intermediate", d * 4) * 2 * 2
    if "frames" in s: total += b * s["channels"] * s["frames"] * s["height"] * s["width"] * 2 * 4
    elif "height" in s: total += b * s["channels"] * s["height"] * s["width"] * 2 * 4
    return total


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--motif", action="append"); p.add_argument("--limit", type=int); p.add_argument("--warmup", type=int, default=5); p.add_argument("--iterations", type=int, default=20); p.add_argument("--memory-fraction", type=float, default=.65); p.add_argument("--max-dense-pairs", type=int, default=500_000_000); p.add_argument("--quick", action="store_true"); a = p.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError("CUDA unavailable")
    cases = json.loads(a.manifest.read_text(encoding="utf-8"))["cases"]
    if a.motif: cases = [x for x in cases if x["motif"] in set(a.motif)]
    if a.quick:
        kept = {}; selected = []
        for case in cases:
            if kept.get(case["motif"], 0) < 1: selected.append(case); kept[case["motif"]] = 1
        cases = selected
    if a.limit: cases = cases[:a.limit]
    props = torch.cuda.get_device_properties(0); allowed = props.total_memory * a.memory_fraction; rows = []
    for case in cases:
        base = {"case_id": case["case_id"], "domain": case["domain"], "motif": case["motif"], "shape": case["shape"], "framework": "pytorch_reference", "device_name": props.name, "compute_capability": f"{props.major}.{props.minor}", "torch_version": torch.__version__, "python_version": platform.python_version()}
        if estimate_bytes(case) > allowed:
            rows.append({**base, "status": "oom_preflight", "estimated_bytes": estimate_bytes(case), "memory_limit_bytes": int(allowed)}); continue
        if case["motif"] in {"token_mixer", "dit_block", "dual_stream"} and case["shape"].get("tokens", 0) ** 2 * case["shape"].get("q_heads", 1) > a.max_dense_pairs:
            rows.append({**base, "status": "compute_preflight", "reason": "dense attention pair count exceeds A10 batch-run guard; use structured_attention or explicitly lower the manifest shape"}); continue
        try:
            motif = case["motif"]
            if motif == "ffn": current = run_ffn(case, a.warmup, a.iterations)
            elif motif == "timestep_modulation": current = run_modulation(case, a.warmup, a.iterations)
            elif motif in {"token_mixer", "text_conditioning", "dual_stream"}: current = run_attention(case, a.warmup, a.iterations, motif == "text_conditioning")
            elif motif in {"rope", "qk_norm"}: current = run_rope_or_qknorm(case, a.warmup, a.iterations)
            elif motif in {"patchify", "latent_codec", "video_3d"}: variants, ref = conv_variants(case); current = timing_rows(case, variants, a.warmup, a.iterations, ref)
            elif motif == "structured_attention": current = run_structured(case, a.warmup, a.iterations)
            else:
                rows.append({**base, "status": "unsupported_reference", "reason": "complete/hybrid DiT adapter requires model-specific projection contracts"}); continue
            rows.extend({**base, **row, "status": "success"} for row in current)
        except Exception as error:
            rows.append({**base, "status": "error", "error": repr(error)[:4000]})
        finally:
            torch.cuda.empty_cache()
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"cases": len(cases), "rows": len(rows), "success": sum(r["status"] == "success" for r in rows), "output": str(a.output)}))
    return 0


if __name__ == "__main__": raise SystemExit(main())
