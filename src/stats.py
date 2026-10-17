"""Confidence intervals and per-probe breakdowns.

Why a two-stage bootstrap: the 300 test attacks come from only 15 distinct
probes (witnesses). Prompts that share a probe are correlated: if the model
always obeys "name a pet that meows", all ~20 of those prompts fail together.
Resampling prompts alone would pretend we have 300 independent attacks and give
too-narrow intervals. So each bootstrap round:
  1. resamples probes with replacement (which attacks we happened to test), then
  2. resamples prompts with replacement inside each drawn probe.
The interval then reflects "would this hold for other attacks like these?",
not just "other passages for these same 15 attacks".

Before/after comparisons pass both arrays together, so the same resampled
prompts are used for both models (a paired comparison).
"""

from collections import defaultdict

import numpy as np


def two_stage_bootstrap(groups, *value_arrays, n_boot=2000, seed=0):
    """Bootstrap means of one or more aligned 0/1 (or float) arrays.

    groups: probe id per row (use a single constant for clean prompts to get an
    ordinary prompt-level bootstrap). Returns an (n_boot, n_arrays) array.
    """
    rng = np.random.default_rng(seed)
    values = np.column_stack([np.asarray(v, dtype=float) for v in value_arrays])
    by_group = defaultdict(list)
    for i, g in enumerate(groups):
        by_group[g].append(i)
    members = [np.array(ix) for ix in by_group.values()]

    out = np.empty((n_boot, values.shape[1]))
    for b in range(n_boot):
        picked = rng.integers(len(members), size=len(members))
        rows = np.concatenate([rng.choice(members[g], size=len(members[g])) for g in picked])
        out[b] = values[rows].mean(axis=0)
    return out


def ci(samples, alpha=0.05):
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def summarize(groups, before, after=None, n_boot=2000, seed=0):
    """Point estimates + 95% CIs; with `after`, also the paired difference."""
    arrays = [before] if after is None else [before, after]
    boots = two_stage_bootstrap(groups, *arrays, n_boot=n_boot, seed=seed)
    res = {"before": float(np.mean(before)), "before_ci": ci(boots[:, 0])}
    if after is not None:
        res.update({
            "after": float(np.mean(after)),
            "after_ci": ci(boots[:, 1]),
            "delta": float(np.mean(after) - np.mean(before)),
            "delta_ci": ci(boots[:, 1] - boots[:, 0]),
        })
    return res


def rate_by_group(groups, values):
    """Mean of `values` per group, e.g. attack success rate per probe."""
    acc = defaultdict(list)
    for g, v in zip(groups, values):
        acc[g].append(float(v))
    return {g: {"rate": float(np.mean(v)), "n": len(v)} for g, v in sorted(acc.items())}
