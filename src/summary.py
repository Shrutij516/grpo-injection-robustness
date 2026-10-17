"""Print the standard results table for one evaluation run.

  python -m src.summary --tag baseline_0p5b_layoutB --out_dir /content/drive/MyDrive/grpo-inj
"""

import argparse
import json
from pathlib import Path

ATTACK_ROWS = [
    ("Unseen family T2 (headline)", "test_unseen_family"),
    ("Seen families T1/T3/T4, new wording", "test_seen_family"),
    ("SEP probes (2nd out-of-distribution)", "test_attack"),
]


def fmt(a):
    return f"{a['rate']:6.1%}  [{a['ci95'][0]:5.1%}, {a['ci95'][1]:5.1%}]  n={a['n']}"


def table(m):
    lines = [f"{m['tag']}: {m['model']}, layout {m['layout']}, system prompt '{m['system_prompt']}'"
             + (f", adapter {m['adapter']}" if m.get("adapter") else ", untrained"),
             f"{'row':44s} {'ASR (95% CI)':32s} task acc under attack"]
    for label, split in ATTACK_ROWS:
        if split in m:
            v = m[split]
            lines.append(f"{label:44s} {fmt(v['attack_success_rate']):32s} {fmt(v['task_accuracy_under_attack'])}")
            if split == "test_seen_family":
                for fam, f in v["by_family"].items():
                    lines.append(f"{'   seen family ' + fam:44s} {fmt(f['attack_success_rate']):32s} "
                                 f"{fmt(f['task_accuracy_under_attack'])}")
    if "test_clean" in m:
        c = m["test_clean"]
        lines += ["",
                  f"{'Clean QA accuracy':44s} {fmt(c['accuracy'])}",
                  f"{'Clean QA refusal rate':44s} {fmt(c['refusal_rate'])}",
                  f"{'Clean QA answers > 50 words':44s} {fmt(c['too_long_rate'])}   mean words {c['mean_words']:.1f}"]
    return "\n".join(lines)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()
    print(table(json.loads((Path(args.out_dir) / args.tag / "metrics.json").read_text())))
