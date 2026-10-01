#!/usr/bin/env python3
"""Extract auditable layout-decision snippets from optional local checkouts."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import urllib.parse
import urllib.request
from pathlib import Path


HERE = Path(__file__).resolve().parent


def commit(root: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], check=True,
                              text=True, capture_output=True).stdout.strip()
    except Exception:
        return None


def lines_for(path: Path, patterns: list[str], context: int) -> list[dict]:
    if path.is_dir():
        candidates = [p for p in path.rglob("*") if p.is_file() and p.stat().st_size < 4_000_000]
    elif path.is_file():
        candidates = [path]
    else:
        return []
    regex = re.compile("|".join(re.escape(x) for x in patterns), re.IGNORECASE)
    hits = []
    for candidate in candidates:
        try:
            text = candidate.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for i, line in enumerate(text):
            if regex.search(line):
                lo, hi = max(0, i - context), min(len(text), i + context + 1)
                hits.append({"file": str(candidate), "line": i + 1,
                             "snippet": "\n".join(f"{j + 1}: {text[j]}" for j in range(lo, hi))})
                if len(hits) >= 40:
                    return hits
    return hits


def github_json(url: str) -> dict:
    headers = {"Accept": "application/vnd.github+json", "User-Agent": "layout-research-evidence/1.0"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return json.load(response)


def github_text(url: str) -> str:
    headers = {"User-Agent": "layout-research-evidence/1.0"}
    if token := os.environ.get("GITHUB_TOKEN"):
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def hits_in_text(text: str, patterns: list[str], context: int, file_name: str) -> list[dict]:
    lines = text.splitlines()
    regex = re.compile("|".join(re.escape(x) for x in patterns), re.IGNORECASE)
    hits = []
    for index, line in enumerate(lines):
        if regex.search(line):
            lo, hi = max(0, index - context), min(len(lines), index + context + 1)
            hits.append({"file": file_name, "line": index + 1,
                         "snippet": "\n".join(f"{number + 1}: {lines[number]}" for number in range(lo, hi))})
            if len(hits) >= 40:
                break
    return hits


def remote_evidence(framework: dict, context: int) -> dict:
    repo = framework["remote_repo"]
    meta = github_json(f"https://api.github.com/repos/{repo}")
    branch = meta["default_branch"]
    revision = github_json(f"https://api.github.com/repos/{repo}/commits/{urllib.parse.quote(branch)}")["sha"]
    tree = None
    hits, missing = [], []
    for spec in framework["evidence"]:
        paths = [spec["path"]]
        if Path(spec["path"]).suffix == "":
            if tree is None:
                tree = github_json(f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1")
            prefix = spec["path"].rstrip("/") + "/"
            paths = [row["path"] for row in tree.get("tree", [])
                     if row.get("type") == "blob" and row["path"].startswith(prefix)
                     and Path(row["path"]).suffix in {".py", ".cc", ".cpp", ".h", ".hpp"}][:80]
        found = []
        for path in paths:
            raw = f"https://raw.githubusercontent.com/{repo}/{revision}/{path}"
            try:
                found.extend(hits_in_text(github_text(raw), spec["patterns"], context, path))
            except Exception:
                continue
            if len(found) >= 40:
                break
        if not found:
            missing.append(spec["path"])
        hits.extend(found[:40])
    return {"status": "success" if hits else "no_pattern_match", "commit": revision,
            "default_branch": branch, "remote_repo": repo,
            "immutable_root": f"https://github.com/{repo}/tree/{revision}",
            "missing_or_unmatched": missing, "hits": hits, "evidence_mode": "remote_pinned"}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", type=Path, default=HERE / "framework_decisions.json")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--context", type=int, default=2)
    p.add_argument("--remote-fallback", action="store_true")
    args = p.parse_args()
    matrix = json.loads(args.matrix.read_text(encoding="utf-8"))
    rows = []
    for fw in matrix["frameworks"]:
        root_text = os.environ.get(fw["source_root_env"], "")
        root = Path(root_text).resolve() if root_text else None
        base = {"framework": fw["name"], "kind": fw["kind"],
                "decision_scope": fw["decision_scope"], "source_root_env": fw["source_root_env"],
                "source_root": str(root) if root else None, "commit": commit(root) if root else None,
                "urls": fw["urls"]}
        if not root or not root.exists():
            if args.remote_fallback:
                try:
                    rows.append({**base, **remote_evidence(fw, args.context)})
                except Exception as error:
                    rows.append({**base, "status": "remote_source_error", "error": repr(error), "hits": []})
            else:
                rows.append({**base, "status": "source_checkout_unavailable", "hits": []})
            continue
        hits = []
        missing = []
        for spec in fw["evidence"]:
            path = root / spec["path"]
            found = lines_for(path, spec["patterns"], args.context)
            if not found:
                missing.append(spec["path"])
            hits.extend(found)
        rows.append({**base, "status": "success" if hits else "no_pattern_match", "evidence_mode": "local_checkout",
                     "missing_or_unmatched": missing, "hits": hits})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote source evidence for {len(rows)} frameworks to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
