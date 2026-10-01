#!/usr/bin/env python3
"""Fail-fast memory preflight before optional whole-engine serving tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--reserve-gib", type=float, default=4.0,
                        help="Runtime/KV/workspace reserve in addition to 2.5x weight bytes")
    args = parser.parse_args()
    import torch
    weights = sum(path.stat().st_size for path in args.model.glob("*.safetensors"))
    if not weights:
        raise SystemExit(f"no safetensors weights in {args.model}")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA unavailable")
    free, total = torch.cuda.mem_get_info(0)
    required = int(weights * 2.5 + args.reserve_gib * 2**30)
    payload = {"model": str(args.model.resolve()), "weight_bytes": weights,
               "free_gpu_bytes": free, "total_gpu_bytes": total,
               "estimated_required_bytes": required, "fits": required <= free}
    print(json.dumps(payload, ensure_ascii=False))
    return 0 if payload["fits"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
