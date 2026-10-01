#!/usr/bin/env python3
"""Audit the ref_talks evidence chain and optionally verify immutable sources.

This script establishes *source-level existence*, not empirical performance.
It keeps that distinction explicit in every output record.  RQ1--RQ6 are read
from the canonical v14 reverse-coverage table; RQ7--RQ10 and the compiler/kernel
mechanisms are read from the v4 forensic CSV.
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import re
import urllib.request
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
REF = HERE / "ref_talks"
CANONICAL = REF / "layout_framework_compare_continued_v14_canonical_evidence_RQ1_RQ6_audited.md"
LEDGER = REF / "layout_summary_v4_evidence_solution_roles.csv"

PREFIX_FRAMEWORK = {
    "V": "vLLM",
    "S": "SGLang",
    "T": "TensorRT-LLM",
    "F": "FlashInfer",
}


def rq_tokens(text: str) -> set[str]:
    return {f"RQ{number}" for number in re.findall(r"RQ(10|[1-9])", text or "")}


def parse_v14_rules(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    marker = "# 4. Reverse coverage audit"
    if marker not in text:
        raise ValueError(f"missing reverse-coverage table in {path}")
    section = text.split(marker, 1)[1].split("## 4.1", 1)[0]
    rows = []
    for line in section.splitlines():
        if not line.startswith("|") or "---" in line or "Evidence ID" in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if len(cells) != 3 or not re.fullmatch(r"[FSTV]-[ABL]\d+", cells[0]):
            continue
        evidence_id, coverage, hypotheses = cells
        rows.append({
            "id": evidence_id,
            "framework": PREFIX_FRAMEWORK[evidence_id[0]],
            "rqs": sorted(rq_tokens(coverage)),
            "hypotheses": hypotheses,
            "evidence_level": "source_static",
            "source": str(path),
        })
    return rows


def parse_v4_ledger(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", newline="") as handle:
        for source in csv.DictReader(handle):
            rows.append({
                "id": source["id"],
                "framework": source["framework"],
                "semantic_class": source["semantic_class"],
                "subgraph": source["subgraph"],
                "rqs": sorted(rq_tokens(source["rq"])),
                "direct_fact": source["direct_fact"],
                "problem_map": source["problem_map"],
                "solution_component": source["solution_component"],
                "counterevidence_group": source["counterevidence_group"],
                "repo": source["repo"],
                "sha": source["sha"],
                "path": source["path"],
                "audited_range": source["audited_range"],
                "immutable_url": source["immutable_url"],
                "evidence_level": "source_static",
                "source": str(path),
            })
    return rows


def raw_url(url: str) -> str:
    match = re.match(r"https://github\.com/([^/]+/[^/]+)/blob/([^/]+)/(.*?)(?:#L\d+(?:-L\d+)?)?$", url)
    if not match:
        return url.split("#", 1)[0]
    repo, sha, path = match.groups()
    return f"https://raw.githubusercontent.com/{repo}/{sha}/{path}"


def verify_row(row: dict, timeout: float) -> dict:
    url = row.get("immutable_url", "")
    result = {"id": row["id"], "url": url, "status": "not_checked"}
    if not url:
        return {**result, "status": "missing_url"}
    if row.get("sha") and row["sha"] not in url:
        return {**result, "status": "sha_not_in_url"}
    try:
        request = urllib.request.Request(raw_url(url), headers={"User-Agent": "layout-rq-audit/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read()
        line_count = payload.count(b"\n") + 1
        start = int((row.get("audited_range") or "1").split("-", 1)[0])
        return {**result, "status": "immutable_range_verified" if line_count >= start else "range_out_of_file",
                "bytes": len(payload), "line_count": line_count}
    except Exception as error:  # Preserve individual failures; never invent success.
        return {**result, "status": "fetch_failed", "error": repr(error)}


def build_report(v14: list[dict], v4: list[dict], verification: list[dict]) -> dict:
    by_rq: dict[str, dict[str, set[str]]] = {
        f"RQ{i}": {"frameworks": set(), "evidence_ids": set(), "subgraphs": set()} for i in range(1, 11)
    }
    for row in [*v14, *v4]:
        for rq in row["rqs"]:
            by_rq[rq]["frameworks"].add(row["framework"])
            by_rq[rq]["evidence_ids"].add(row["id"])
            if row.get("subgraph"):
                by_rq[rq]["subgraphs"].add(row["subgraph"])
    normalized = {
        rq: {key: sorted(values) for key, values in entry.items()}
        for rq, entry in by_rq.items()
    }
    malformed = []
    for row in v4:
        required = ("id", "framework", "repo", "sha", "path", "audited_range", "immutable_url", "direct_fact")
        missing = [key for key in required if not row.get(key)]
        if missing or (row["sha"] not in row["immutable_url"]):
            malformed.append({"id": row["id"], "missing": missing,
                              "sha_in_url": row["sha"] in row["immutable_url"]})
    verified = sum(item["status"] == "immutable_range_verified" for item in verification)
    return {
        "schema_version": 1,
        "claim": "The audited source contains the decision rule/mechanism; GPU performance hypotheses remain open.",
        "guardrail": "source_static evidence must never be labeled runtime_empirical",
        "v14_rule_count": len(v14),
        "v4_evidence_count": len(v4),
        "rq_coverage": normalized,
        "malformed_evidence": malformed,
        "url_verification": {
            "attempted": len(verification),
            "verified": verified,
            "rows": verification,
        },
        "source_chain_pass": len(v14) == 65 and len(v4) == 50 and not malformed,
    }


def markdown(report: dict) -> str:
    lines = [
        "# ref_talks source-evidence audit",
        "",
        f"- v14 stable rules: {report['v14_rule_count']} (expected 65)",
        f"- v4 forensic evidence rows: {report['v4_evidence_count']} (expected 50)",
        f"- schema/provenance check: {'PASS' if report['source_chain_pass'] else 'FAIL'}",
        f"- immutable URLs verified in this run: {report['url_verification']['verified']}/{report['url_verification']['attempted']}",
        "",
        "> This audit proves source-level mechanism existence only. It does not prove an RQ hypothesis or a speedup.",
        "",
        "| RQ | Frameworks with source evidence | Evidence count | Subgraph tags |",
        "|---|---|---:|---|",
    ]
    for rq, row in report["rq_coverage"].items():
        lines.append(f"| {rq} | {', '.join(row['frameworks']) or '-'} | {len(row['evidence_ids'])} | {', '.join(row['subgraphs']) or '-'} |")
    if report["malformed_evidence"]:
        lines += ["", "## Malformed evidence", "", "```json", json.dumps(report["malformed_evidence"], indent=2), "```"]
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-urls", action="store_true")
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    v14 = parse_v14_rules(CANONICAL)
    v4 = parse_v4_ledger(LEDGER)
    verification = []
    if args.verify_urls:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(verify_row, row, args.timeout) for row in v4]
            for future in as_completed(futures):
                verification.append(future.result())
        verification.sort(key=lambda row: row["id"])
    report = build_report(v14, v4, verification)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report_path = args.report or args.output.with_suffix(".md")
    report_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "report": str(report_path),
                      "source_chain_pass": report["source_chain_pass"]}))
    return 0 if report["source_chain_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
