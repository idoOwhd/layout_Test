"""CPU-only contracts for a finite, explicitly bounded RQ2 layout search.

No performance result is produced here. Exactness is relative to the declared
binary address-mapping space, never to all legal GPU implementations.
"""
from __future__ import annotations
import hashlib
import itertools
import random
import statistics
from audit_rq2_holdout_statistics import paired_bootstrap


def layout_axes(case):
    axes = []
    stage = case["stage"]
    for layer in range(case["decoder_blocks"]):
        cutoff = 6 if stage >= 6 else stage
        # A one-row matrix has identical row/column address mappings. Do not
        # manufacture two layouts by changing an irrelevant size-one stride.
        if case["batch"] * case["query_length"] > 1:
            axes.append(f"L{layer}.qkv_projection")
        if cutoff >= 3:
            axes.append(f"L{layer}.kv_cache")
        if cutoff >= 4 and case["batch"] * case["query_length"] > 1:
            axes.append(f"L{layer}.output_projection")
        if cutoff >= 5 and case["batch"] * case["query_length"] > 1:
            axes.append(f"L{layer}.gate_up_projection")
        if cutoff >= 6 and case["batch"] * case["query_length"] > 1:
            axes.append(f"L{layer}.down_projection")
    return axes


def candidates(axes, previous=None, exact_bits=5, budget=48, seed=0):
    """Include measured extensions of EVERY retained near-optimal old plan.

    Larger searches are a bounded measured set, NOT an exact oracle. Free and
    frozen minima are evaluated over the SAME final candidate set. For large
    expansions, each old plan gets at least an all-zero extension; remaining
    budget supplies free proposals. This is a best-measured frozen extension,
    NOT a proven optimal extension of every old plan. Absence of an extension
    cannot masquerade as invalidity. Budget is a proposal target, not a hard
    cap that silently discards old near-optimal plans.
    """
    if len(axes) <= exact_bits:
        return list(itertools.product((0, 1), repeat=len(axes))), "exact_declared_space"
    rng = random.Random(seed)
    values = {tuple([0]*len(axes)), tuple([1]*len(axes))}
    for i in range(len(axes)):
        values.add(tuple(int(j == i) for j in range(len(axes))))
    if previous:
        old_axes = previous["axes"]
        if not set(old_axes).issubset(axes):
            raise ValueError("graph growth removed an old layout decision")
        for old in previous["near_plans"]:
            frozen = dict(zip(old_axes, old))
            values.add(tuple(frozen.get(k, 0) for k in axes))
    while len(values) < min(budget, 2**len(axes)):
        values.add(tuple(rng.randrange(2) for _ in axes))
    return sorted(values), "bounded_measured_candidates_not_oracle"


def near_optimal(rows, epsilon):
    correct = [r for r in rows if r["correctness"]["correct"]]
    if not correct:
        raise ValueError("no correctness-qualified candidate")
    optimum = min(r["train_median_ms"] for r in correct)
    return [tuple(r["bits"]) for r in correct if r["train_median_ms"] <= (1+epsilon)*optimum]


def frozen_eligible(bits, axes, old_axes, old_near):
    index = [axes.index(k) for k in old_axes]
    return tuple(bits[i] for i in index) in set(map(tuple, old_near))


def summarize_pair(case, rows, previous, epsilon, holdout):
    qualified = [r for r in rows if r["correctness"]["correct"]]
    free = min(qualified, key=lambda r: r["train_median_ms"])
    frozen_rows = [r for r in qualified if frozen_eligible(
        r["bits"], layout_axes(case), previous["axes"], previous["near_plans"])]
    if not frozen_rows:
        raise ValueError("frozen extension missing: search is incomplete")
    frozen = min(frozen_rows, key=lambda r: r["train_median_ms"])
    # holdout is independently timed, interleaved AB/BA, never reused to select
    # either plan. Formula-derived ratios keep their measured inputs explicit.
    numerator = statistics.median(holdout["frozen_ms"])
    denominator = statistics.median(holdout["free_ms"])
    # The CI must target the same ratio-of-medians estimand as the point
    # estimate. Resample paired process/repeat indices, never separate arms.
    stat=paired_bootstrap(holdout["frozen_ms"],holdout["free_ms"])
    new_near=near_optimal(rows,epsilon)
    indices=[layout_axes(case).index(k) for k in previous["axes"]]
    projected_new={tuple(bits[i] for i in indices) for bits in new_near}
    old_set=set(map(tuple,previous["near_plans"]))
    surviving=projected_new & old_set
    minimum_changes=min(sum(a != b for a,b in zip(old,new))
                        for old in old_set for new in projected_new)
    return {"small_case_id": previous["case_id"], "expanded_case_id": case["case_id"],
            "epsilon": epsilon, "old_near_set_size": len(previous["near_plans"]),
            "old_axes": previous["axes"], "expanded_axes": layout_axes(case),
            "free_plan": free["bits"], "frozen_plan": frozen["bits"],
            "old_near_plans": previous["near_plans"], "frozen_candidate_count": len(frozen_rows),
            "free_candidate_count": len(qualified),
            "ER_set": numerator/denominator-1,
            "speedup_frozen_over_free": numerator/denominator,
            "measured_numerator_frozen_median_ms": numerator,
            "measured_denominator_free_median_ms": denominator,
            "paired_ratio_bootstrap_95_interval": stat["ratio_of_medians_paired_bootstrap_95_interval"],
            "paired_ratio_interval_estimand": "median(frozen_ms)/median(free_ms), paired index bootstrap",
            "median_of_paired_ratios": stat["median_of_paired_ratios"],
            "median_of_paired_ratios_bootstrap_95_interval": stat["median_of_paired_ratios_bootstrap_95_interval"],
            "old_near_set_survival_fraction":len(surviving)/len(old_set),
            "minimum_old_axis_changes_to_expanded_near_set":minimum_changes,
            "influence_scope":"assignment survival across expansion, NOT causal compiler propagation distance",
            "holdout": holdout, "same_plan": free["bits"] == frozen["bits"],
            "claim_scope": "declared_layout_space_and_actual_adapter_only"}


def stable_seed(group, name):
    # Stage/candidate/framework deliberately excluded: old math gets the same
    # values under graph expansion and across framework processes.
    text = str(group)+"/"+name
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**31)
