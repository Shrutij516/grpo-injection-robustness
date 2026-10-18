"""Read near-miss answers: outputs that look like a garbled nonce but are NOT
counted as attacks (see README, "train/near_miss_rate").

Sources, in order:
  1. samples.jsonl written during training (4 answers every 25 steps), for a
     step range, e.g. steps 200-300 -> only ~20 answers;
  2. if fewer than --n near-misses were found and --regenerate is given:
     fresh samples on TRAINING prompts from the step-200 milestone and final
     adapters, at the training temperature. These are labeled "regenerated";
     they are not the exact answers seen during training.
Test data is never used.

  python -m src.inspect_near_miss --run_dir .../train_main_seed1 --from_step 200 --to_step 300 --n 15 --regenerate
"""

import argparse
import json
import random
from pathlib import Path

from src.rewards import edit_distance, normalize, witness_present
from src.train_grpo import near_miss

ROOT = Path(__file__).resolve().parent.parent


def explain(text, nonce):
    """Which words made near_miss() fire, and by which rule (mirrors near_miss exactly)."""
    if not nonce or witness_present(text, nonce):
        return []
    target, out = nonce.lower(), []
    for tok in normalize(text).split():
        if len(tok) < 4:
            continue
        if tok[:4] == target[:4]:
            out.append(f"'{tok}' (same first 4 letters)")
        else:
            d = edit_distance(tok, target, limit=4)
            if d <= 4:
                out.append(f"'{tok}' ({d} edits)")
    if not out and target in "".join(normalize(text).split()):
        out.append("nonce spelled with spaces/punctuation")
    return out


def from_samples(run_dir, lo, hi):
    path = Path(run_dir) / "samples.jsonl"
    recs = []
    for line in path.open():
        r = json.loads(line)
        if lo <= r["step"] <= hi and r.get("witness"):
            recs.append({"source": f"training step {r['step']}", "nonce": r["witness"], "gold": r["gold"],
                         "prompt_tail": r.get("prompt_tail", ""), "output": r["completion"]})
    return recs


def regenerate(run_dir, adapters, n_prompts=48, k=8, seed=0):
    """Sample k answers per injected TRAINING prompt from each adapter (training temperature)."""
    import torch
    from src.evaluate import load_model, pick_device

    run_dir = Path(run_dir)
    cfg = json.loads((run_dir / "run_config.json").read_text())["config"]
    temp, max_new = cfg["grpo"]["temperature"], cfg["grpo"]["max_completion_length"]
    rows = [json.loads(line) for line in (ROOT / cfg["train_file"]).open()]
    inj = [r for r in rows if r["injected"]]
    random.Random(seed).shuffle(inj)
    inj = inj[:n_prompts]
    device, dtype = pick_device()
    recs = []
    for name in adapters:
        adapter = run_dir / name
        if not (adapter / "adapter_config.json").exists():
            print(f"[inspect] skipping {adapter}: not found")
            continue
        tok, model = load_model(cfg["model"], str(adapter), device, dtype)
        torch.manual_seed(seed)
        for i in range(0, len(inj), 8):
            batch = inj[i:i + 8]
            texts = [tok.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True) for r in batch]
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            with torch.inference_mode():
                out = model.generate(**enc, do_sample=True, temperature=temp, top_p=1.0, top_k=0,
                                     max_new_tokens=max_new, num_return_sequences=k,
                                     pad_token_id=tok.pad_token_id)
            dec = tok.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            for j, text in enumerate(dec):
                r = batch[j // k]
                recs.append({"source": f"regenerated from {name} (T={temp})", "nonce": r["witness"],
                             "gold": r["gold_answers"], "prompt_tail": r["prompt"][-1]["content"][-300:],
                             "output": text.strip(), "id": r["id"]})
        del model
    out_path = run_dir / "near_miss_regenerated.jsonl"
    with out_path.open("w") as f:
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[inspect] {len(recs)} regenerated answers saved to {out_path}")
    return recs


def summarize(recs, label):
    if not recs:
        return f"{label}: no answers"
    atk = sum(witness_present(r["output"], r["nonce"]) for r in recs)
    nm = sum(near_miss(r["output"], r["nonce"]) for r in recs)
    return f"{label}: {len(recs)} answers to injected prompts; attack {atk / len(recs):.1%}; near-miss {nm / len(recs):.1%}"


def show(recs, n):
    hits = [r for r in recs if near_miss(r["output"], r["nonce"])]
    lines = []
    for r in hits[:n]:
        lines.append(f"[{r['source']}] nonce={r['nonce']}  gold={r['gold'][:2]}\n"
                     f"  triggered by: {', '.join(explain(r['output'], r['nonce']))}\n"
                     f"  -> {r['output']!r}")
    return hits, "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--from_step", type=int, default=200)
    ap.add_argument("--to_step", type=int, default=300)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--regenerate", action="store_true",
                    help="if too few near-misses in samples.jsonl, sample fresh answers on training prompts")
    ap.add_argument("--adapters", nargs="*", default=["milestones/step-0200", "final"])
    args = ap.parse_args()

    recs = from_samples(args.run_dir, args.from_step, args.to_step)
    print(summarize(recs, f"samples.jsonl, steps {args.from_step}-{args.to_step}"))
    hits, text = show(recs, args.n)
    print(text)
    step1 = from_samples(args.run_dir, 1, 1)
    print("\n" + summarize(step1, "samples.jsonl, step 1 (for the step-1 near-miss spike)"))
    print(show(step1, args.n)[1])
    if len(hits) < args.n:
        print(f"\nOnly {len(hits)} near-miss answers in samples.jsonl for steps {args.from_step}-{args.to_step}.")
        if args.regenerate:
            more = regenerate(args.run_dir, args.adapters)
            print(summarize(more, "regenerated"))
            print(show(more, args.n - len(hits))[1])
        else:
            print("Rerun with --regenerate to sample more answers from the step-200 and final adapters.")


if __name__ == "__main__":
    main()
