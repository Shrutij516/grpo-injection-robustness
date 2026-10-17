"""Training curves from log_history.jsonl -> reward_curve.png (for the README).

Four small panels, each with its own single y-axis (no dual axes):
reward (total + components), attack rate on training prompts (+ near-miss
rate), KL to the reference model, and mean answer length in words.

  python -m src.plot_training --run_dir /content/drive/MyDrive/grpo-inj/train_seed1
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Reference categorical palette, fixed order (validated: light surface, 4 slots).
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]


def load_log(run_dir):
    """Training-step records only, de-duplicated by step (a resume can re-log
    steps after the last checkpoint; the later entry wins)."""
    by_step = {}
    for line in (Path(run_dir) / "log_history.jsonl").open():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # half-written last line after a disconnect
        if "reward" in rec:
            by_step[rec["step"]] = rec
    return [by_step[s] for s in sorted(by_step)]


def series(log, key):
    pts = [(r["step"], r[key]) for r in log if key in r and r[key] is not None and math.isfinite(r[key])]
    return [p[0] for p in pts], [p[1] for p in pts]


def smooth(ys, k=5):
    out = []
    for i in range(len(ys)):
        window = ys[max(0, i - k + 1): i + 1]
        out.append(sum(window) / len(window))
    return out


def _panel(ax, log, lines, title, ylabel):
    ax.set_facecolor(SURFACE)
    for (key, label), color in zip(lines, SERIES):
        xs, ys = series(log, key)
        if not xs:
            continue
        ax.plot(xs, ys, color=color, linewidth=0.8, alpha=0.3)
        sm = smooth(ys)
        ax.plot(xs, sm, color=color, linewidth=2, label=label)
        if len(lines) > 1:  # direct label at the line end (identity not by color alone)
            ax.annotate(label, (xs[-1], sm[-1]), xytext=(4, 0), textcoords="offset points",
                        fontsize=8, color=INK_2, va="center")
    ax.set_title(title, fontsize=10, color=INK, loc="left", pad=22 if len(lines) > 1 else 6)
    ax.set_xlabel("step", fontsize=8, color=INK_2)
    ax.set_ylabel(ylabel, fontsize=8, color=INK_2)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.tick_params(colors=INK_2, labelsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if len(lines) > 1:
        # One row between the title and the plot, so it never covers a line.
        ax.legend(fontsize=7, frameon=False, labelcolor=INK_2, loc="lower left",
                  bbox_to_anchor=(0, 1.0), ncol=len(lines), borderaxespad=0.2, handlelength=1.5)


def plot(run_dir, out_path=None, title=None):
    log = load_log(run_dir)
    if not log:
        raise SystemExit(f"[plot] no training steps logged in {run_dir}")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), facecolor=SURFACE)
    _panel(axes[0, 0], log, [("reward", "total"), ("rewards/reward_task/mean", "task"),
                             ("rewards/reward_resist/mean", "resist (injected only)"),
                             ("rewards/reward_refusal/mean", "refusal penalty")],
           "Reward (thin: per step, thick: 5-step mean)", "mean reward")
    _panel(axes[0, 1], log, [("train/attack_rate", "attack rate"), ("train/near_miss_rate", "near-miss rate")],
           "Attack rate on training prompts", "share of injected prompts")
    _panel(axes[1, 0], log, [("kl", "KL")], "KL to the reference (base) model", "KL")
    _panel(axes[1, 1], log, [("train/words_mean", "words")], "Mean answer length", "words")
    fig.suptitle(title or f"GRPO training: {Path(run_dir).name}", fontsize=12, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 0.97, 0.96))
    out_path = Path(out_path or Path(run_dir) / "reward_curve.png")
    fig.savefig(out_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    print(f"[plot] wrote {plot(args.run_dir, args.out)}")
