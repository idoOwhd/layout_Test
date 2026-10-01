#!/usr/bin/env python3
"""Record installed framework versions and whether CUDA is usable."""

from __future__ import annotations

import importlib
import json
import os
import platform
from pathlib import Path


MODULES = {"torch": "torch", "triton": "triton", "sglang": "sglang", "vllm": "vllm",
           "tvm": "tvm", "tilelang": "tilelang", "hidet_hexcute": "hidet"}


def main():
    out = {"python": platform.python_version(), "executable": os.sys.executable,
           "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "modules": {}}
    for name, module_name in MODULES.items():
        try:
            module = importlib.import_module(module_name)
            out["modules"][name] = {"available": True, "version": getattr(module, "__version__", None),
                                     "path": getattr(module, "__file__", None)}
        except Exception as e:
            out["modules"][name] = {"available": False, "error": repr(e)}
    try:
        import torch
        out["cuda"] = {"available": torch.cuda.is_available(), "runtime": torch.version.cuda}
        if torch.cuda.is_available():
            p = torch.cuda.get_device_properties(0)
            out["cuda"].update(name=p.name, capability=f"{p.major}.{p.minor}", total_memory=p.total_memory)
    except Exception as e:
        out["cuda"] = {"available": False, "error": repr(e)}
    output = Path(os.environ["LAYOUT_PROBE_OUTPUT"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__": main()
