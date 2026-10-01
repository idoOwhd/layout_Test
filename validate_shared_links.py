#!/usr/bin/env python3
"""Validate that supplied ChatGPT shares expose embedded conversation data."""

from __future__ import annotations

import argparse
import json
import re
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent


def fetch(source: dict, timeout: float) -> dict:
    request = urllib.request.Request(source["url"], headers={"User-Agent": "Mozilla/5.0 layout-research-link-audit/1.0"})
    row = {"id": source["id"], "url": source["url"], "checked_at": datetime.now(timezone.utc).isoformat()}
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
            row.update(http_status=response.status, final_url=response.url, bytes=len(body.encode()),
                       title=(re.search(r"<title>(.*?)</title>", body, re.I | re.S).group(1) if re.search(r"<title>(.*?)</title>", body, re.I | re.S) else None))
        share_id = source["url"].rstrip("/").rsplit("/", 1)[-1]
        marker_hits = {marker: marker.lower() in body.lower() for marker in source.get("required_markers", [])}
        row.update(marker_hits=marker_hits, has_router_payload="streamController.enqueue" in body,
                   has_share_id=share_id in body)
        row["accessible"] = bool(row["http_status"] == 200 and row["bytes"] > 100_000 and row["has_router_payload"] and row["has_share_id"] and all(marker_hits.values()))
    except Exception as error:
        row.update(accessible=False, error=repr(error))
    return row


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sources", type=Path, default=HERE / "shared_sources.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--timeout", type=float, default=60)
    p.add_argument("--strict", action="store_true")
    a = p.parse_args()
    sources = json.loads(a.sources.read_text(encoding="utf-8"))["sources"]
    rows = [fetch(source, a.timeout) for source in sources]
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps({"all_accessible": all(r["accessible"] for r in rows), "sources": rows}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"accessible": sum(r["accessible"] for r in rows), "total": len(rows), "output": str(a.output)}))
    return 1 if a.strict and not all(r["accessible"] for r in rows) else 0


if __name__ == "__main__":
    raise SystemExit(main())
