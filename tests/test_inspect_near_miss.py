import json

import pytest

from src.inspect_near_miss import explain, from_samples, regenerate, show
from src.train_grpo import DEFAULT_CONFIG, load_config, near_miss, train

CASES = [("Thixvirum is the answer", "Thixvonri"), ("T h i x v o n r i", "Thixvonri"),
         ("Thixvonri", "Thixvonri"), ("It happened in the late 1960s.", "Klasego"),
         ("grab it", "Grabonko"), ("The answer is 1943.", "Thixvonri"), ("", "Thixvonri")]


@pytest.mark.parametrize("text,nonce", CASES)
def test_explain_fires_exactly_when_near_miss_does(text, nonce):
    assert bool(explain(text, nonce)) == near_miss(text, nonce)


def test_explain_names_the_rule():
    assert "same first 4 letters" in explain("Thixvirum", "Thixvonri")[0]
    assert "edits" in explain("It happened late.", "Klasego")[0]   # an ordinary word: a false alarm
    assert explain("T h i x v o n r i", "Thixvonri") == ["nonce spelled with spaces/punctuation"]


def test_from_samples_filters_steps_and_clean_rows(tmp_path):
    rows = [{"step": s, "witness": w, "gold": ["x"], "completion": c}
            for s, w, c in [(1, "Thixvonri", "Thixvirum"), (200, "Thixvonri", "Thixvirum"),
                            (225, None, "clean answer"), (300, "Klasego", "1943"), (325, "Klasego", "Klasegu")]]
    (tmp_path / "samples.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    recs = from_samples(tmp_path, 200, 300)
    assert [r["source"] for r in recs] == ["training step 200", "training step 300"]
    hits, text = show(recs, 15)
    assert len(hits) == 1 and "Thixvirum" in text


def test_regenerate_uses_training_prompts_and_saves(tmp_path):
    tiny = ["model=trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "precision=fp32",
            "grpo.num_generations=4", "grpo.per_device_train_batch_size=4",
            "grpo.gradient_accumulation_steps=2", "grpo.max_completion_length=8", "grpo.max_steps=2",
            "grpo.save_steps=2", "milestone_every=2", "sample_every=1"]
    run = tmp_path / "run"
    train(load_config(DEFAULT_CONFIG, tiny), run, use_cpu=True)
    recs = regenerate(run, ["final"], n_prompts=2, k=2)
    assert len(recs) == 4 and all(r["id"].startswith("train-") for r in recs)
    assert (run / "near_miss_regenerated.jsonl").exists()
