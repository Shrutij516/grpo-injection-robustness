import numpy as np

from src.stats import rate_by_group, summarize, two_stage_bootstrap


def test_rate_by_group():
    r = rate_by_group(["cat", "cat", "dog", "dog"], [1, 0, 1, 1])
    assert r == {"cat": {"rate": 0.5, "n": 2}, "dog": {"rate": 1.0, "n": 2}}


def test_probe_clustering_widens_interval():
    # 10 probes x 20 prompts; each probe is either always obeyed or never.
    groups = np.repeat(np.arange(10), 20)
    values = np.repeat([1, 0] * 5, 20)
    clustered = two_stage_bootstrap(groups, values, seed=1)[:, 0]
    naive = two_stage_bootstrap(np.zeros(200), values, seed=1)[:, 0]
    assert clustered.std() > 2 * naive.std()


def test_paired_delta_and_ci_contains_estimate():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(15), 20)
    before = rng.random(300) < 0.6
    after = rng.random(300) < 0.2
    s = summarize(groups, before, after, n_boot=500)
    assert s["before_ci"][0] <= s["before"] <= s["before_ci"][1]
    assert s["delta_ci"][1] < 0  # a large drop is clearly significant


def test_deterministic_with_seed():
    g, v = [0, 0, 1, 1], [1, 0, 1, 1]
    assert (two_stage_bootstrap(g, v, seed=3) == two_stage_bootstrap(g, v, seed=3)).all()
