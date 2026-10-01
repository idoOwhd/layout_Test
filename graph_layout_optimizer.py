#!/usr/bin/env python3
"""Small exact graph-level layout optimizer and local-greedy comparator.

Each operator implementation declares required input layouts, produced output
layouts and a measured/predicted kernel cost.  The solver includes conversions
and therefore exposes when a locally fastest kernel is not the fastest path.
It is exact for the supplied finite implementation set; large graphs should
use the same objective with beam search or an ILP solver.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Candidate:
    cost_ms: float
    layouts: tuple[tuple[str, str], ...]
    decisions: tuple[dict, ...]


def as_state(layouts: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted(layouts.items()))


def conversion_cost(spec: dict, tensor: str, source: str, target: str) -> float:
    if source == target: return 0.0
    table = spec.get("conversion_cost_ms", {})
    keys = (f"{tensor}:{source}->{target}", f"*:{source}->{target}")
    for key in keys:
        if key in table: return float(table[key])
    raise KeyError(f"missing conversion cost for {tensor}: {source}->{target}")


def apply(spec: dict, op: dict, candidate: Candidate, implementation: dict) -> Candidate:
    layouts = dict(candidate.layouts); conversions = []; extra = 0.0
    for tensor in op["inputs"]:
        required = implementation["input_layouts"][tensor]; current = layouts[tensor]
        cost = conversion_cost(spec, tensor, current, required)
        if cost: conversions.append({"tensor": tensor, "from": current, "to": required, "cost_ms": cost})
        layouts[tensor] = required; extra += cost
    for tensor in op["outputs"]: layouts[tensor] = implementation["output_layouts"][tensor]
    decision = {"op": op["id"], "implementation": implementation["id"], "kernel_ms": float(implementation["kernel_ms"]),
                "conversions": conversions, "incremental_ms": float(implementation["kernel_ms"]) + extra,
                "output_layouts": implementation["output_layouts"]}
    return Candidate(candidate.cost_ms + decision["incremental_ms"], as_state(layouts), candidate.decisions + (decision,))


def exact(spec: dict, max_states: int = 1_000_000) -> Candidate:
    states = {as_state(spec["initial_layouts"]): Candidate(0.0, as_state(spec["initial_layouts"]), ())}
    for op in spec["ops"]:
        next_states = {}
        for candidate in states.values():
            for implementation in op["implementations"]:
                updated = apply(spec, op, candidate, implementation); old = next_states.get(updated.layouts)
                if old is None or updated.cost_ms < old.cost_ms: next_states[updated.layouts] = updated
        if len(next_states) > max_states: raise RuntimeError(f"state count {len(next_states)} exceeds --max-states={max_states}")
        states = next_states
    return min(states.values(), key=lambda x: x.cost_ms)


def greedy(spec: dict) -> Candidate:
    current = Candidate(0.0, as_state(spec["initial_layouts"]), ())
    for op in spec["ops"]:
        current = min((apply(spec, op, current, impl) for impl in op["implementations"]), key=lambda x: x.cost_ms)
    return current


def render(candidate: Candidate) -> dict:
    return {"total_ms": candidate.cost_ms, "final_layouts": dict(candidate.layouts), "decisions": list(candidate.decisions)}


def main() -> int:
    p = argparse.ArgumentParser(); p.add_argument("spec", type=Path); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-states", type=int, default=1_000_000); a = p.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8")); optimal, local = exact(spec, a.max_states), greedy(spec)
    result = {"name": spec.get("name", a.spec.stem), "objective": "kernel + explicit layout conversion latency",
              "exact": render(optimal), "local_greedy": render(local),
              "exact_speedup_over_greedy": local.cost_ms / optimal.cost_ms,
              "hypothesis_supported_by_cost_instance": optimal.cost_ms < local.cost_ms,
              "claim_guardrail": "This is a cost-instance mechanism proof, not a GPU speedup claim. Replace costs with paired measurements."}
    a.output.parent.mkdir(parents=True, exist_ok=True); a.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
