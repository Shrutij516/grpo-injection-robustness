"""Training pipeline tests on a tiny random model, CPU only (a few seconds each).

They check wiring, logging, saving and resume, not learning: the tiny model
produces gibberish, so rewards are flat.
"""

import json
import math

import pytest

from src import rewards
from src.plot_training import load_log, plot
from src.smoke_check import check
from src.train_grpo import DEFAULT_CONFIG, build_trainer, load_config, near_miss, train

TINY = [
    "model=trl-internal-testing/tiny-Qwen2ForCausalLM-2.5", "precision=fp32",
    "grpo.num_generations=4", "grpo.per_device_train_batch_size=4", "grpo.gradient_accumulation_steps=2",
    "grpo.max_completion_length=8", "grpo.save_steps=1", "grpo.max_steps=4",
    "milestone_every=2", "sample_every=1",
]
# Section 4 logging list (TRAINING_PROPOSAL.md)
ALWAYS = ["reward", "rewards/reward_task/mean", "rewards/reward_resist/mean", "rewards/reward_refusal/mean",
          "kl", "completions/mean_length", "frac_reward_zero_std", "train/refusal_rate",
          "train/words_mean", "train/over50_rate", "step_time"]
INJECTED_ONLY = ["train/attack_rate", "train/near_miss_rate"]  # logged when a batch has injected prompts


def cfg(*extra):
    return load_config(DEFAULT_CONFIG, TINY + list(extra))


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("drive") / "train_test"
    train(cfg(), out, stop_after=2, use_cpu=True)
    return out


def test_approved_config_values():
    c = load_config(DEFAULT_CONFIG)
    g = c["grpo"]
    assert c["model"] == "Qwen/Qwen2.5-0.5B-Instruct" and c["precision"] == "fp16_base"
    assert (c["lora"]["r"], c["lora"]["alpha"], c["lora"]["dropout"]) == (16, 32, 0.0)
    assert len(c["lora"]["target_modules"]) == 7
    assert (g["learning_rate"], g["beta"], g["num_generations"], g["max_steps"]) == (5e-5, 0.04, 8, 300)
    assert g["per_device_train_batch_size"] * g["gradient_accumulation_steps"] == 8 * g["num_generations"]
    assert (g["temperature"], g["max_completion_length"], g["disable_dropout"]) == (0.9, 96, True)
    assert c["reward_weights"] == [1.0, 1.0, 1.0, 0.0]


def test_reward_functions_wired_to_src_rewards(tmp_path):
    t = build_trainer(cfg(), tmp_path / "wiring", use_cpu=True)
    assert t.reward_func_names == ["reward_task", "reward_resist", "reward_refusal", "monitor"]
    assert t.reward_funcs[:3] == [rewards.reward_task, rewards.reward_resist, rewards.reward_refusal]
    assert t.reward_weights.tolist() == [1.0, 1.0, 1.0, 0.0]


def test_training_uses_near_match_nonce_scoring(tmp_path):
    t = build_trainer(cfg(), tmp_path / "nearmatch", use_cpu=True)
    chat = [[{"role": "assistant", "content": "Thixvounri"}]]
    resist = t.reward_funcs[1]
    assert resist(completions=chat, witness=["Thixvonri"]) == [0.0]       # misspelled nonce = obeyed
    assert resist(completions=chat, witness=[None]) == [None]             # clean prompt: not applicable
    logged = {}
    t.reward_funcs[3](prompts=[[{"role": "user", "content": "q"}]], completions=chat, gold_answers=[["x"]],
                      witness=["Thixvonri"], log_metric=lambda k, v: logged.__setitem__(k, v))
    assert logged["train/attack_rate"] == 1.0


def test_near_miss_detector():
    assert near_miss("T h i x v o n r i", "Thixvonri")      # spelled with spaces
    assert near_miss("Thixabcde", "Thixvonri")              # same first 4 letters, far edit distance
    assert not near_miss("Thixvonri", "Thixvonri")          # that is a counted attack, not a near miss
    assert not near_miss("The answer is 1943.", "Thixvonri")


def test_logs_checkpoints_milestones_samples_written(run_dir):
    for name in ("run_config.json", "log_history.jsonl", "samples.jsonl", "checkpoint-2", "milestones/step-0002"):
        assert (run_dir / name).exists(), name
    assert (run_dir / "milestones" / "step-0002" / "adapter_config.json").exists()
    log = load_log(run_dir)
    assert [r["step"] for r in log] == [1, 2]
    for key in ALWAYS:
        assert all(key in r and math.isfinite(r[key]) for r in log), key
    for key in INJECTED_ONLY:
        assert any(key in r for r in log), key
    sample = json.loads((run_dir / "samples.jsonl").open().readline())
    assert sample["step"] == 1 and {"witness", "gold", "completion"} <= set(sample)
    assert not (run_dir / "final").exists()  # stopped early: no final adapter


def test_resume_continues_and_finishes(run_dir):
    t = train(cfg(), run_dir, use_cpu=True)  # same command again, no stop -> runs to max_steps=4
    assert t.state.global_step == 4
    assert [r["step"] for r in load_log(run_dir)] == [1, 2, 3, 4]
    assert (run_dir / "milestones" / "step-0004").exists()
    assert (run_dir / "final" / "adapter_config.json").exists()


def test_resume_with_changed_config_is_refused(run_dir):
    with pytest.raises(SystemExit, match="different config"):
        build_trainer(cfg("grpo.temperature=1.0"), run_dir, use_cpu=True)


def test_fp16_precision_needs_gpu(tmp_path):
    with pytest.raises(SystemExit, match="needs a CUDA GPU"):
        build_trainer(cfg("precision=fp16_base"), tmp_path / "fp16", use_cpu=True)


def test_plot_writes_png(run_dir):
    path = plot(run_dir)
    assert path.exists() and path.stat().st_size > 10_000


# ---- smoke-run decision rule (synthetic logs) -------------------------------

def fake_log(attack=(0.30, 0.30), reward=(1.0, 1.0), zero_std=0.3, kl=0.01, refusal=0.02, words=15.0,
             loss=0.01, steps=30):
    log = []
    for s in range(1, steps + 1):
        f = (s - 1) / 29
        log.append({"step": s, "loss": loss, "grad_norm": 0.5, "kl": kl, "frac_reward_zero_std": zero_std,
                    "train/attack_rate": attack[0] + f * (attack[1] - attack[0]),
                    "reward": reward[0] + f * (reward[1] - reward[0]),
                    "train/refusal_rate": refusal, "train/words_mean": words, "step_time": 12.0})
    return log


def test_smoke_pass_when_attack_rate_drops():
    assert check(fake_log(attack=(0.32, 0.18)))["decision"].startswith("PASS")


def test_smoke_pass_when_reward_rises():
    assert check(fake_log(reward=(1.0, 1.2)))["decision"].startswith("PASS")


@pytest.mark.parametrize("kw", [dict(loss=float("nan")), dict(kl=1.5), dict(refusal=0.3), dict(words=1.0)])
def test_smoke_hard_stops(kw):
    assert check(fake_log(attack=(0.32, 0.10), **kw))["decision"].startswith("STOP")


def test_smoke_zero_variance_groups_raise_temperature():
    d = check(fake_log(zero_std=0.7))["decision"]
    assert d.startswith("CHANGE") and "temperature=1.0" in d


def test_smoke_varying_groups_flat_reward_raise_lr():
    d = check(fake_log(zero_std=0.3))["decision"]
    assert d.startswith("CHANGE") and "learning_rate=1e-4" in d


def test_smoke_second_attempt_falls_back_to_rft():
    assert check(fake_log(), second_attempt=True)["decision"].startswith("FALLBACK")


def test_smoke_incomplete():
    assert check(fake_log(steps=20))["decision"].startswith("INCOMPLETE")
