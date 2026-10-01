#!/usr/bin/env python3
"""Audit v13 source observations without confusing them with performance proof."""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DEFAULT_SPEC = HERE / "ref_talks/layout_summary_v13_complete_problem_trial_audit_RQ1_RQ10.md"
DEFAULT_CHAIN = HERE / "ref_talks/layout_summary_v13_complete_rule_proxy_fallback_trigger_hypothesis_chain.csv"
DEFAULT_REGISTRY = HERE / "v13_experiment_registry.json"


def canonical_hypotheses(path: Path) -> dict[str, dict]:
    result: dict[str, dict] = {}
    current_rq = None
    row = re.compile(r"^\| (H\d+(?:\.\d+|\.NEG)) \| (.*?) \| (.*?) \| (RQ\d+-E\d+) \|")
    for line in path.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"^# \d+\. (L-RQ\d+) — (.+)$", line)
        if heading:
            current_rq = heading.group(1)
        match = row.match(line)
        if match and current_rq:
            result[match.group(1)] = {
                "rq": current_rq, "hypothesis": match.group(2),
                "falsifier": match.group(3), "experiment": match.group(4),
            }
    return result


def split_hypotheses(value: str) -> set[str]:
    return {part.strip() for part in value.split(",") if part.strip()}


def markdown(summary: dict) -> str:
    lines = [
        "# v13 source-observation audit",
        "",
        "> This report answers whether the source-level mechanism/omitted-variable observation occurs across applicable stacks. It does not claim a speedup; performance is adjudicated separately.",
        "",
        "## Completeness",
        "",
        f"- Canonical hypotheses: **{summary['canonical_hypothesis_count']}**.",
        f"- Registry hypotheses: **{summary['registry_hypothesis_count']}**.",
        f"- Missing from registry: **{', '.join(summary['missing_from_registry']) or 'none'}**.",
        f"- Extra in registry: **{', '.join(summary['extra_in_registry']) or 'none'}**.",
        f"- Experiment-ID mismatches: **{len(summary['experiment_id_mismatches'])}**.",
        "",
        "## Per-hypothesis prevalence",
        "",
        "The denominator is the set of stacks with at least one `DIRECT_MECHANISM` row for that RQ. Therefore a majority means the specific observation is linked by direct evidence in more than half of the RQ-applicable direct stacks, not merely in more than half of rows.",
        "",
        "| Hypothesis | RQ | Direct stacks linked / applicable | Majority | Current-source rows | Runtime experiment |",
        "|---|---|---:|---|---:|---|",
    ]
    for row in summary["hypotheses"]:
        lines.append(
            f"| {row['id']} | {row['rq']} | {row['support_count']}/{row['applicable_count']} "
            f"({', '.join(row['supporting_stacks']) or 'none'}) | **{row['majority_status']}** | "
            f"{row['current_source_rows']} | {row['experiment']} |"
        )
    lines += ["", "## Per-RQ observation summary", "",
              "| RQ | Direct applicable stacks | Hypotheses with direct-stack majority | Important observations not yet majority-backed |",
              "|---|---|---:|---|"]
    for row in summary["rqs"]:
        lines.append(f"| {row['rq']} | {', '.join(row['applicable_stacks'])} | "
                     f"{row['majority_hypotheses']}/{row['hypothesis_count']} | "
                     f"{', '.join(row['non_majority_hypotheses']) or 'none'} |")
    lines += ["", "## Interpretation guardrails", "",
              "- `supported` here means source-prevalent, not empirically faster.",
              "- A non-majority result does not falsify a hypothesis; it means the current source ledger does not establish broad prevalence.",
              "- Adjacent/counterevidence rows are retained in the machine-readable output but are excluded from the direct-stack majority numerator and denominator.",
              "- RQ5 performance remains blocked on a single GPU even though its mechanism is source-prevalent.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--chain", type=Path, default=DEFAULT_CHAIN)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    canonical = canonical_hypotheses(args.spec)
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    registered = {}
    for rq in registry["rqs"]:
        for hypothesis, spec in rq["hypotheses"].items():
            registered[hypothesis] = {"rq": rq["rq"], **spec}

    with args.chain.open(encoding="utf-8", newline="") as handle:
        chain = list(csv.DictReader(handle))
    direct_by_rq: dict[str, set[str]] = defaultdict(set)
    for row in chain:
        if row["evidence_role"] == "DIRECT_MECHANISM":
            direct_by_rq[row["layout_rq"]].add(row["stack"])

    hypotheses = []
    for hypothesis, definition in sorted(canonical.items(), key=lambda item: (
            int(re.search(r"\d+", item[1]["rq"]).group()), item[0])):
        rq = definition["rq"]
        relevant = [row for row in chain if row["layout_rq"] == rq
                    and hypothesis in split_hypotheses(row["hypothesis_links"])]
        direct = [row for row in relevant if row["evidence_role"] == "DIRECT_MECHANISM"]
        support = sorted({row["stack"] for row in direct})
        applicable = sorted(direct_by_rq[rq])
        threshold = len(applicable) // 2 + 1 if applicable else None
        hypotheses.append({
            "id": hypothesis, **definition,
            "support_count": len(support), "applicable_count": len(applicable),
            "majority_threshold": threshold,
            "majority_status": "supported" if threshold and len(support) >= threshold else "not_supported",
            "supporting_stacks": support, "applicable_stacks": applicable,
            "direct_rows": len(direct), "adjacent_rows": len(relevant) - len(direct),
            "current_source_rows": sum(row["source_status"] == "CURRENT_SOURCE_SUPPLEMENT" for row in relevant),
            "source_observations": [row["source_rule_fact"] for row in relevant],
        })

    rq_rows = []
    for index in range(1, 11):
        rq = f"L-RQ{index}"
        selected = [row for row in hypotheses if row["rq"] == rq]
        rq_rows.append({
            "rq": rq, "applicable_stacks": sorted(direct_by_rq[rq]),
            "hypothesis_count": len(selected),
            "majority_hypotheses": sum(row["majority_status"] == "supported" for row in selected),
            "non_majority_hypotheses": [row["id"] for row in selected if row["majority_status"] != "supported"],
        })

    mismatches = []
    for hypothesis in sorted(set(canonical) & set(registered)):
        if canonical[hypothesis]["experiment"] != registered[hypothesis].get("experiment"):
            mismatches.append({"hypothesis": hypothesis,
                               "canonical": canonical[hypothesis]["experiment"],
                               "registry": registered[hypothesis].get("experiment")})
    summary = {
        "schema_version": 1,
        "canonical_hypothesis_count": len(canonical),
        "registry_hypothesis_count": len(registered),
        "missing_from_registry": sorted(set(canonical) - set(registered)),
        "extra_in_registry": sorted(set(registered) - set(canonical)),
        "experiment_id_mismatches": mismatches,
        "hypotheses": hypotheses, "rqs": rq_rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "v13_source_observations.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "V13_SOURCE_OBSERVATIONS.md").write_text(markdown(summary), encoding="utf-8")
    print(json.dumps({"hypotheses": len(hypotheses),
                      "missing": summary["missing_from_registry"],
                      "mismatches": len(mismatches),
                      "report": str(args.output_dir / "V13_SOURCE_OBSERVATIONS.md")}, ensure_ascii=False))
    return 1 if summary["missing_from_registry"] or mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
