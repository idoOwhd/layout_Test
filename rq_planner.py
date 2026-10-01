#!/usr/bin/env python3
"""Small, auditable planner for the ref_talks RQ validation suite.

This module deliberately implements finite exact search rather than a learned
model.  It is used in two ways:

* CPU unit tests exercise legality, edge contracts, repairs, scope selection
  and epochal switching without requiring a GPU.
* ``analyze_ref_talks_rqs.py`` replaces all costs with measurements produced by
  ``rq_llm_layout_bench.cu``.  A synthetic cost is never promoted to empirical
  evidence.

The state space is intentionally small.  The scientific question is whether
the framework-local/default plan has measurable regret, not whether this
prototype is a production-scale optimizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import heapq
import math
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True)
class Representation:
    """Typed edge representation used by producers, consumers and repairs."""

    name: str
    axes: tuple[str, ...]
    dtype: str = "fp16"
    page_size: int = 0
    packing: str = "plain"
    scale_format: str = "none"
    placement: str = "cuda"
    persistent: bool = False

    def same_storage_contract(self, other: "Representation") -> bool:
        return (
            self.axes == other.axes
            and self.dtype == other.dtype
            and self.page_size == other.page_size
            and self.packing == other.packing
            and self.scale_format == other.scale_format
            and self.placement == other.placement
        )


@dataclass(frozen=True)
class Consumer:
    name: str
    accepted: frozenset[str]
    cost_ms: Mapping[str, float]
    weight: float = 1.0

    def cost(self, representation: str) -> float:
        if representation not in self.accepted:
            return math.inf
        return float(self.cost_ms[representation]) * self.weight


@dataclass(frozen=True)
class Repair:
    name: str
    source: str
    target: str
    cost_ms: float
    temporary_bytes: int = 0
    persistent_bytes: int = 0
    kind: str = "convert"
    legal: bool = True


@dataclass(frozen=True)
class RepairPath:
    total_ms: float
    final_representation: str
    repairs: tuple[Repair, ...]


@dataclass(frozen=True)
class JointPlan:
    total_ms: float
    storage: tuple[str, ...]
    consumer_representations: tuple[tuple[str, str], ...]
    repairs: tuple[Repair, ...]
    memory_bytes: int
    detail: Mapping[str, object] = field(default_factory=dict)


def shortest_repair_path(
    start: str,
    accepted: Iterable[str],
    repairs: Sequence[Repair],
    *,
    max_temporary_bytes: int = 2**63 - 1,
) -> RepairPath | None:
    """Return the least-cost legal repair path with a hard memory constraint."""

    targets = set(accepted)
    if start in targets:
        return RepairPath(0.0, start, ())
    adjacency: dict[str, list[Repair]] = {}
    for repair in repairs:
        if repair.legal and repair.temporary_bytes <= max_temporary_bytes:
            adjacency.setdefault(repair.source, []).append(repair)
    queue: list[tuple[float, int, str, tuple[Repair, ...]]] = [(0.0, 0, start, ())]
    best: dict[str, float] = {start: 0.0}
    serial = 1
    while queue:
        cost, _, node, path = heapq.heappop(queue)
        if cost != best.get(node):
            continue
        if node in targets:
            return RepairPath(cost, node, path)
        for edge in adjacency.get(node, ()):
            new_cost = cost + edge.cost_ms
            if new_cost < best.get(edge.target, math.inf):
                best[edge.target] = new_cost
                heapq.heappush(queue, (new_cost, serial, edge.target, path + (edge,)))
                serial += 1
    return None


def common_layout_plan(
    representations: Mapping[str, Representation],
    consumers: Sequence[Consumer],
) -> JointPlan | None:
    """Best single layout accepted directly by every consumer."""

    common = set(representations)
    for consumer in consumers:
        common.intersection_update(consumer.accepted)
    if not common:
        return None
    layout = min(common, key=lambda name: sum(c.cost(name) for c in consumers))
    return JointPlan(
        total_ms=sum(c.cost(layout) for c in consumers),
        storage=(layout,),
        consumer_representations=tuple((c.name, layout) for c in consumers),
        repairs=(),
        memory_bytes=0,
        detail={"policy": "common_layout"},
    )


def joint_representation_plan(
    representations: Mapping[str, Representation],
    consumers: Sequence[Consumer],
    repairs: Sequence[Repair],
    *,
    base_bytes: int,
    max_extra_bytes: int = 2**63 - 1,
    allow_duplicate_persistent: bool = True,
) -> JointPlan | None:
    """Exact finite search over one base representation plus repair edges.

    A repair may be paid once and shared by all consumers that use its target.
    This models a materialized/duplicated persistent representation.  Direct
    views are represented by zero-byte repair edges.  The function first
    filters illegal and over-budget edges; performance ranking never sees an
    illegal candidate.
    """

    best_plan: JointPlan | None = None
    for base in representations:
        choices: list[list[tuple[str, float, tuple[Repair, ...]]]] = []
        feasible = True
        for consumer in consumers:
            per_consumer: list[tuple[str, float, tuple[Repair, ...]]] = []
            for target in consumer.accepted:
                path = shortest_repair_path(base, {target}, repairs, max_temporary_bytes=max_extra_bytes)
                if path is not None:
                    per_consumer.append((target, consumer.cost(target), path.repairs))
            if not per_consumer:
                feasible = False
                break
            choices.append(per_consumer)
        if not feasible:
            continue

        # The suites have at most a handful of consumers.  Enumerating their
        # finite alternatives keeps the result exact and auditable.
        states: list[tuple[float, dict[str, str], dict[tuple[str, str, str], Repair]]] = [(0.0, {}, {})]
        for consumer, alternatives in zip(consumers, choices):
            next_states = []
            for cost, assignment, used in states:
                for target, consume_cost, path in alternatives:
                    new_used = dict(used)
                    for edge in path:
                        new_used[(edge.name, edge.source, edge.target)] = edge
                    next_states.append((cost + consume_cost, {**assignment, consumer.name: target}, new_used))
            states = next_states
        for consume_cost, assignment, used in states:
            unique_repairs = tuple(used.values())
            materialized_targets = {edge.target for edge in unique_repairs if edge.persistent_bytes > 0}
            if materialized_targets and not allow_duplicate_persistent:
                continue
            extra_memory = sum(edge.persistent_bytes for edge in unique_repairs)
            peak_temporary = max((edge.temporary_bytes for edge in unique_repairs), default=0)
            if max(extra_memory, peak_temporary) > max_extra_bytes:
                continue
            total = consume_cost + sum(edge.cost_ms for edge in unique_repairs)
            storage = tuple(sorted({base, *materialized_targets}))
            plan = JointPlan(
                total_ms=total,
                storage=storage,
                consumer_representations=tuple(sorted(assignment.items())),
                repairs=unique_repairs,
                memory_bytes=base_bytes + extra_memory,
                detail={"policy": "joint_repair", "base": base},
            )
            if best_plan is None or plan.total_ms < best_plan.total_ms:
                best_plan = plan
    return best_plan


def partition_layout_plan(
    groups: Mapping[str, Sequence[Consumer]],
    representations: Mapping[str, Representation],
    *,
    pool_overhead_ms: float,
    pool_overhead_bytes: int,
    base_bytes: int,
) -> JointPlan | None:
    """Choose one direct layout per group and charge split-pool overhead."""

    storage: list[str] = []
    assignments: list[tuple[str, str]] = []
    total = 0.0
    for group, consumers in groups.items():
        plan = common_layout_plan(representations, consumers)
        if plan is None:
            return None
        layout = plan.storage[0]
        storage.append(layout)
        total += plan.total_ms
        assignments.extend((f"{group}/{name}", rep) for name, rep in plan.consumer_representations)
    extra_pools = max(0, len(groups) - 1)
    total += extra_pools * pool_overhead_ms
    return JointPlan(
        total_ms=total,
        storage=tuple(storage),
        consumer_representations=tuple(assignments),
        repairs=(),
        memory_bytes=base_bytes + extra_pools * pool_overhead_bytes,
        detail={"policy": "partitioned", "groups": tuple(groups), "extra_pools": extra_pools},
    )


@dataclass(frozen=True)
class Epoch:
    name: str
    request_count: int
    cost_per_request_ms: Mapping[str, float]


@dataclass(frozen=True)
class EpochDecision:
    epoch: str
    configuration: str
    service_ms: float
    switch_ms: float


@dataclass(frozen=True)
class EpochResult:
    total_ms: float
    decisions: tuple[EpochDecision, ...]
    switch_count: int


def run_epoch_controller(
    epochs: Sequence[Epoch],
    configurations: Sequence[str],
    *,
    initial: str,
    switch_cost_ms: Mapping[tuple[str, str], float],
    hysteresis_ratio: float = 1.0,
) -> EpochResult:
    """Switch only when expected epoch savings exceed cost × hysteresis."""

    current = initial
    decisions: list[EpochDecision] = []
    switches = 0
    total = 0.0
    for epoch in epochs:
        current_service = epoch.request_count * float(epoch.cost_per_request_ms[current])
        candidate = min(configurations, key=lambda c: epoch.cost_per_request_ms[c])
        candidate_service = epoch.request_count * float(epoch.cost_per_request_ms[candidate])
        switch = 0.0 if candidate == current else float(switch_cost_ms[(current, candidate)])
        if candidate != current and current_service - candidate_service > switch * hysteresis_ratio:
            current = candidate
            service = candidate_service
            switches += 1
        else:
            switch = 0.0
            service = current_service
        total += service + switch
        decisions.append(EpochDecision(epoch.name, current, service, switch))
    return EpochResult(total, tuple(decisions), switches)


def fixed_epoch_policy(epochs: Sequence[Epoch], configuration: str) -> EpochResult:
    decisions = tuple(
        EpochDecision(e.name, configuration, e.request_count * float(e.cost_per_request_ms[configuration]), 0.0)
        for e in epochs
    )
    return EpochResult(sum(d.service_ms for d in decisions), decisions, 0)


def per_request_oracle(
    epochs: Sequence[Epoch],
    configurations: Sequence[str],
    *,
    initial: str,
    switch_cost_ms: Mapping[tuple[str, str], float],
) -> EpochResult:
    """Deliberately aggressive baseline that can expose switching thrash.

    Each request independently chooses its fastest configuration and pays a
    switch before every request when that differs from the current state.
    """

    current = initial
    total = 0.0
    switches = 0
    decisions: list[EpochDecision] = []
    for epoch in epochs:
        candidate = min(configurations, key=lambda c: epoch.cost_per_request_ms[c])
        switch = 0.0
        service = epoch.request_count * float(epoch.cost_per_request_ms[candidate])
        if candidate != current:
            switch = epoch.request_count * float(switch_cost_ms[(current, candidate)])
            switches += epoch.request_count
            current = candidate
        total += service + switch
        decisions.append(EpochDecision(epoch.name, current, service, switch))
    return EpochResult(total, tuple(decisions), switches)
