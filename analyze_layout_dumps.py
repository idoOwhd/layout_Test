#!/usr/bin/env python3
"""Normalize load/store and layout evidence from compiler dump directories."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


PTX_MEMORY = re.compile(r"\b(ld|st)\.(global|shared|local)(?:\.[a-z0-9_]+)*", re.I)
PTX_VECTOR = re.compile(r"\.(v[248])\.", re.I)
SASS_MEMORY = re.compile(r"\b(LDG|STG|LDS|STS)(?:\.[A-Z0-9.]+)?", re.I)
TRITON_ENCODINGS = {
    "blocked": re.compile(r"(?:#ttg\.)?blocked", re.I),
    "shared": re.compile(r"(?:#ttg\.)?(?:swizzled_)?shared", re.I),
    "dot_operand": re.compile(r"(?:#ttg\.)?dot_op", re.I),
    "slice": re.compile(r"(?:#ttg\.)?slice", re.I),
    "linear_layout": re.compile(r"LinearLayout|linear_layout", re.I),
    "convert_layout": re.compile(r"convert_layout|ConvertLayout", re.I),
}


def parse(path: Path, root: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    ptx_ops = Counter(f"{a.lower()}.{b.lower()}" for a, b in PTX_MEMORY.findall(text))
    sass_ops = Counter(x.upper() for x in SASS_MEMORY.findall(text))
    vectors = Counter(x.lower() for x in PTX_VECTOR.findall(text))
    encodings = {name: len(regex.findall(text)) for name, regex in TRITON_ENCODINGS.items()}
    return {"file": str(path.relative_to(root)), "suffix": path.suffix.lower(), "bytes": path.stat().st_size,
            "ptx_memory_ops": dict(ptx_ops), "sass_memory_ops": dict(sass_ops), "ptx_vector_widths": dict(vectors),
            "layout_encoding_mentions": encodings,
            "has_tvm_index_map": bool(re.search(r"IndexMap|transform_layout|axis_separator", text, re.I)),
            "has_cute_layout": bool(re.search(r"Layout<|make_layout|Swizzle<|TiledMMA|TiledCopy", text)),
            "evidence_status": "observed" if ptx_ops or sass_ops or any(encodings.values()) else "no_known_pattern"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--dump-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); root = a.dump_dir.resolve()
    suffixes = {".ptx", ".sass", ".ttgir", ".mlir", ".ll", ".tir", ".cu", ".cuh"}
    paths = sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in suffixes)
    rows = [parse(path, root) for path in paths]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    print(json.dumps({"files": len(rows), "with_layout_or_memory_evidence": sum(r["evidence_status"] == "observed" for r in rows), "output": str(a.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
