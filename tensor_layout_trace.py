"""Stable JSON descriptions for graph-boundary PyTorch tensor layouts."""

from __future__ import annotations

import hashlib
import json


def describe_tensor(value) -> dict:
    row = {"shape": list(value.shape), "stride": list(value.stride()), "dtype": str(value.dtype),
           "device": str(value.device), "storage_offset": int(value.storage_offset()),
           "contiguous": bool(value.is_contiguous())}
    if value.ndim == 4:
        import torch
        row["channels_last"] = bool(value.is_contiguous(memory_format=torch.channels_last))
    if value.ndim == 5:
        import torch
        row["channels_last_3d"] = bool(value.is_contiguous(memory_format=torch.channels_last_3d))
    if value.device.type == "cuda":
        row["data_ptr_alignment_16"] = int(value.data_ptr() % 16)
        row["data_ptr_alignment_128"] = int(value.data_ptr() % 128)
    return row


def describe_workload(workload) -> dict:
    result = {"inputs": {f"input_{i}": describe_tensor(x) for i, x in enumerate(workload.inputs)},
              "buffers": {name: describe_tensor(value) for name, value in workload.model.named_buffers()}}
    result["signature"] = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    result["scope"] = "graph_boundary_only"
    result["internal_layout_note"] = "Compiler/kernel internal layouts require IR/PTX/SASS evidence; they are not inferred from boundary strides."
    return result
