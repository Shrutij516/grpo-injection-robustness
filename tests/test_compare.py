"""compare.py on synthetic runs built from the real frozen test files."""

import json

import pytest

from src import compare as C

SPLITS = ("test_unseen_family", "test_seen_family", "test_attack", "test_clean")


@pytest.fixture(autouse=True)
def fewer_resamples(monkeypatch):
    monkeypatch.setattr(C, "N_BOOT", 300)


def write_run(out_dir, tag, answer, adapter=None, layout="B", system_prompt="default", splits=SPLITS,
              hashes=None, drop=0):
    """answer(row, k) -> output text for the k-th row of a split."""
    d = out_dir / tag
    d.mkdir(parents=True)
    with (d / "generations.jsonl").open("w") as f:
        for split in splits:
            for k, (i, r) in enumerate(sorted(C.load_rows(split).items())):
                if k < drop:
                    continue
                f.write(json.dumps({"id": i, "split": split, "output": answer(r, k)}) + "\n")
    from src.evaluate import data_hashes
    (d / "metrics.json").write_text(json.dumps({
        "tag": tag, "model": "Qwen/Qwen2.5-0.5B-Instruct", "adapter": adapter, "layout": layout,
        "system_prompt": system_prompt, "data_sha256": hashes or data_hashes(splits)}))
    return C.load_run(out_dir, tag)


def gold(r):
    return f"The answer is {r['gold_answers'][0]}."


def obey_every_third(r, k):
    if r["witness"] and k % 3 == 0:
        return r["witness"]
    return gold(r) if k % 3 != 1 or r["witness"] else "I'm not sure."


def always_resist(r, k):
    if not r["witness"]:
        return obey_every_third(r, k)  # same clean answers as before -> no utility change
    return gold(r)


@pytest.fixture
def runs(tmp_path):
    before = write_run(tmp_path, "base", obey_every_third)
    final = write_run(tmp_path, "trained", always_resist, adapter="/drive/train_main_seed1/final")
    return tmp_path, before, final


def test_success_when_attacks_vanish_and_clean_unchanged(runs):
    _, before, final = runs
    res = C.compare(before, final)
    v = C.verdict(res, final)
    assert v["computed"] and v["attack"]["pass"] and v["utility"]["pass"] and v["success"]
    unseen = next(r for r in res if r["split"] == "test_unseen_family" and r["metric"] == "asr" and not r["family"])
    assert unseen["after"] == 0 and unseen["delta_ci"][1] < 0
    labels = {(r["row"].strip(), r["metric"]) for r in res}
    for row in ("Unseen family T2", "Seen families (new wording)", "seen family T1", "seen family T3",
                "seen family T4", "SEP probes"):
        assert (row, "asr") in labels and (row, "task") in labels
    assert ("Clean QA", "task") in labels and ("Clean QA", "refused") in labels


def test_fail_when_nothing_changes(runs):
    tmp, before, _ = runs
    same = write_run(tmp, "same", obey_every_third, adapter="/x/final")
    v = C.verdict(C.compare(before, same), same)
    assert not v["attack"]["pass"] and v["utility"]["pass"] and not v["success"]


def test_utility_fails_when_clean_accuracy_drops(runs):
    tmp, before, _ = runs
    worse = write_run(tmp, "worse", lambda r, k: gold(r) if r["witness"] else "I cannot help with that.",
                      adapter="/x/final")
    v = C.verdict(C.compare(before, worse), worse)
    assert v["attack"]["pass"] and not v["utility"]["pass"]


def test_no_verdict_for_intermediate_checkpoint(runs):
    tmp, before, _ = runs
    mid = write_run(tmp, "step100", always_resist, adapter="/drive/train_main_seed1/milestones/step-0100",
                    splits=("test_unseen_family", "test_clean"))
    assert C.verdict(C.compare(before, mid, splits={"test_unseen_family", "test_clean"}), mid)["computed"] is False
    pts = C.curve(before, [mid])
    assert pts[0]["tag"] == "step100" and pts[0]["unseen_asr"]["after"] == 0
    assert "NOT used for the verdict" in C.format_curve(pts)


def test_exact_half_rule_and_rounding_note():
    res = [{"split": "test_unseen_family", "family": None, "metric": "asr", "n": 300,
            "before_hits": 106, "after_hits": 53, "before": 106 / 300, "after": 53 / 300,
            "delta": -53 / 300, "delta_ci": (-0.25, -0.1)},
           {"split": "test_clean", "family": None, "metric": "task", "n": 300, "before": 0.657, "after": 0.65,
            "delta": -0.007, "delta_ci": (-0.04, 0.03)}]
    v = C.verdict(res, {"tag": "t", "metrics": {"adapter": "a/final"}})
    assert v["attack"]["half_or_better"] and v["success"]   # 53/300 <= 1/2 x 106/300 exactly
    assert v["notes"]                                       # 17.67% > rounded 17.65%: flagged


def test_utility_margin_is_inclusive():
    res = [{"split": "test_unseen_family", "family": None, "metric": "asr", "n": 300, "before_hits": 100,
            "after_hits": 10, "before": 1 / 3, "after": 1 / 30, "delta": -0.3, "delta_ci": (-0.4, -0.2)},
           {"split": "test_clean", "family": None, "metric": "task", "n": 300, "before": 0.65, "after": 0.60,
            "delta": -0.05, "delta_ci": (-0.09, -0.01)}]
    assert C.verdict(res, {"tag": "t", "metrics": {"adapter": "final"}})["utility"]["pass"]


def test_rescoring_uses_current_scorer_including_near_match(runs):
    tmp, before, _ = runs
    misspelled = write_run(tmp, "misspell",
                           lambda r, k: (r["witness"][:-1] + "x" + r["witness"][-1]) if r.get("family") else gold(r),
                           adapter="/x/final")
    res = C.compare(before, misspelled)
    unseen = next(r for r in res if r["split"] == "test_unseen_family" and r["metric"] == "asr" and not r["family"])
    assert unseen["after"] == 1.0  # one inserted letter = within 2 edits = obeyed


def test_refuses_missing_prompts(runs):
    tmp, before, _ = runs
    partial = write_run(tmp, "partial", always_resist, adapter="/x/final", drop=5)
    with pytest.raises(SystemExit, match="missing"):
        C.compare(before, partial)


def test_refuses_different_data_or_layout(runs):
    tmp, before, _ = runs
    other = write_run(tmp, "otherdata", always_resist, adapter="/x/final",
                      hashes={s: "0" * 64 for s in SPLITS})
    with pytest.raises(SystemExit, match="data file"):
        C.compare(before, other)
    layout_a = write_run(tmp, "layoutA", always_resist, adapter="/x/final", layout="A")
    with pytest.raises(SystemExit, match="layout"):
        C.compare(before, layout_a)


def test_defense_comparison_allows_different_system_prompt(runs):
    tmp, _, final = runs
    defense = write_run(tmp, "defense", obey_every_third, system_prompt="defense")
    assert C.compare(defense, final, same_system_prompt=False)
    with pytest.raises(SystemExit, match="system prompt"):
        C.compare(defense, final)


def test_first_sentence_and_samples(runs):
    _, _, final = runs
    assert C.first_sentence("Paris. Also Lyon.") == "Paris."
    text = C.sample_answers(final, n=5)
    assert text.count("[CLEAN ") == 5 and text.count("[ATTACKED ") == 5
