"""Pre-registered checks for the 30-step smoke run (TRAINING_PROPOSAL.md, Section 7).

Reads only training logs (never test data) and prints:
  1. hard stops: non-finite loss/grad norm, KL > 1.0, training refusal rate
     > 20%, mean answer < 2 words (rates use the last 10 steps);
  2. "reward moves": steps 21-30 vs 1-10, pass if the training attack rate
     drops >= 5 points OR the mean total reward rises >= 0.10;
  3. timing / memory and the projection for 300 steps;
  4. the decision, using the amended rule: one change only, then RFT.

  python -m src.smoke_check --run_dir /content/drive/MyDrive/grpo-inj/train_smoke_seed1
"""

import argparse
import json
import math
import statistics
from pathlib import Path

import yaml

from src.plot_training import load_log

ROOT = Path(__file__).resolve().parent.parent
EARLY, LATE = (1, 10), (21, 30)
ATTACK_DROP, REWARD_RISE = 0.05, 0.10
KL_MAX, REFUSAL_MAX, WORDS_MIN, ZERO_STD_MANY = 1.0, 0.20, 2.0, 0.5


def window_mean(log, key, lo, hi):
    vals = [r[key] for r in log if lo <= r["step"] <= hi and key in r and r[key] is not None]
    return sum(vals) / len(vals) if vals else float("nan")


def is_second_attempt(run_dir):
    """A smoke run that already uses the one allowed change (temperature or LR)."""
    default = yaml.safe_load((ROOT / "configs" / "grpo.yaml").read_text())["grpo"]
    cfg = json.loads((Path(run_dir) / "run_config.json").read_text())["config"]["grpo"]
    return (cfg["temperature"] != default["temperature"]
            or cfg["learning_rate"] != default["learning_rate"])


def check(log, second_attempt=False, max_steps=300):
    r = {"steps_logged": len(log), "last_step": log[-1]["step"] if log else 0}
    if r["last_step"] < LATE[1]:
        r["decision"] = f"INCOMPLETE: only {r['last_step']} steps logged; need {LATE[1]}"
        return r

    bad = [x["step"] for x in log for k in ("loss", "grad_norm")
           if k in x and (x[k] is None or not math.isfinite(x[k]))]
    kl_max = max((x["kl"] for x in log if "kl" in x), default=0.0)
    refusal = window_mean(log, "train/refusal_rate", *LATE)
    words = window_mean(log, "train/words_mean", *LATE)
    r["hard_stops"] = {
        "non-finite loss/grad_norm": (bool(bad), f"steps {bad[:5]}" if bad else "none"),
        f"KL > {KL_MAX}": (kl_max > KL_MAX, f"max KL {kl_max:.3f}"),
        f"refusal rate > {REFUSAL_MAX:.0%} (steps 21-30)": (refusal > REFUSAL_MAX, f"{refusal:.1%}"),
        f"mean answer < {WORDS_MIN:.0f} words (steps 21-30)": (words < WORDS_MIN, f"{words:.1f} words"),
    }

    a0, a1 = window_mean(log, "train/attack_rate", *EARLY), window_mean(log, "train/attack_rate", *LATE)
    w0, w1 = window_mean(log, "reward", *EARLY), window_mean(log, "reward", *LATE)
    r["moves"] = {
        "attack_rate": (a0, a1, a0 - a1 >= ATTACK_DROP),
        "reward": (w0, w1, w1 - w0 >= REWARD_RISE),
    }
    r["frac_reward_zero_std"] = window_mean(log, "frac_reward_zero_std", 1, LATE[1])

    times = [x.get("step_time") for x in log if x.get("step_time")]
    r["sec_per_step"] = statistics.median(times[2:] or times) if times else float("nan")
    r["projected_minutes"] = r["sec_per_step"] * max_steps / 60
    r["peak_mem_gb"] = max((x.get("gpu_peak_mem_gb", 0.0) for x in log), default=0.0)

    moved = r["moves"]["attack_rate"][2] or r["moves"]["reward"][2]
    if any(flag for flag, _ in r["hard_stops"].values()):
        r["decision"] = "STOP: a hard-stop check failed. Fix it before any further training."
    elif moved:
        r["decision"] = "PASS: reward moves. Proceed to the main 300-step run with this config."
    elif second_attempt:
        r["decision"] = ("FALLBACK: the one allowed change was already used and the reward still does not move. "
                         "Switch to rejection-sampling fine-tuning (RFT).")
    elif r["frac_reward_zero_std"] >= ZERO_STD_MANY:
        r["decision"] = ("CHANGE (one allowed): many groups have zero reward variance "
                         f"({r['frac_reward_zero_std']:.0%}). Rerun the smoke with --set grpo.temperature=1.0")
    else:
        r["decision"] = ("CHANGE (one allowed): groups vary "
                         f"({r['frac_reward_zero_std']:.0%} zero-variance) but reward barely moves. "
                         "Rerun the smoke with --set grpo.learning_rate=1e-4")
    return r


def report(r):
    lines = [f"steps logged: {r['steps_logged']} (last step {r['last_step']})"]
    if "hard_stops" in r:
        lines.append("\nHard-stop checks:")
        for name, (failed, detail) in r["hard_stops"].items():
            lines.append(f"  [{'FAIL' if failed else 'ok  '}] {name:44s} {detail}")
        lines.append("\nReward moves? (steps 1-10 vs 21-30)")
        a0, a1, ok_a = r["moves"]["attack_rate"]
        w0, w1, ok_w = r["moves"]["reward"]
        lines.append(f"  attack rate on training prompts: {a0:6.1%} -> {a1:6.1%}  "
                     f"(drop {a0 - a1:+.1%}; need >= {ATTACK_DROP:.0%})  {'PASS' if ok_a else 'no'}")
        lines.append(f"  mean total reward:               {w0:6.3f} -> {w1:6.3f}  "
                     f"(rise {w1 - w0:+.3f}; need >= {REWARD_RISE:.2f})  {'PASS' if ok_w else 'no'}")
        lines.append(f"  groups with zero reward variance (steps 1-30): {r['frac_reward_zero_std']:.0%}")
        lines.append(f"\nTiming: {r['sec_per_step']:.1f} s/step (median) -> ~{r['projected_minutes']:.0f} min "
                     f"for 300 steps; peak GPU memory {r['peak_mem_gb']:.1f} GB")
    lines.append(f"\nDECISION: {r['decision']}")
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()
    log = load_log(args.run_dir)
    print(report(check(log, is_second_attempt(args.run_dir))))
