#!/usr/bin/env python3
"""Download and verify the catalog-pinned small Qwen checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
EXPECTED = {"hidden_size": 896, "num_attention_heads": 14, "num_key_value_heads": 2}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--no-download", action="store_true")
    args = parser.parse_args()
    config_path = args.output / "config.json"
    weights = list(args.output.glob("*.safetensors")) if args.output.exists() else []
    if not config_path.exists() or not weights:
        if args.no_download:
            raise SystemExit(f"incomplete local model: {args.output}")
        args.output.mkdir(parents=True, exist_ok=True)
        snapshot_download(repo_id=MODEL_ID, revision=REVISION, local_dir=args.output)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    bad = {key: {"actual": config.get(key), "expected": value}
           for key, value in EXPECTED.items() if config.get(key) != value}
    if bad:
        raise SystemExit(f"checkpoint shape mismatch: {bad}")
    print(json.dumps({"model_id": MODEL_ID, "revision": REVISION,
                      "path": str(args.output.resolve()), "verified": EXPECTED}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
