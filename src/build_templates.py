"""Build the frozen template-attack data (training set + two test rows).

Decisions recorded in the README / PROPOSAL (made before seeing test data):
  - Model Qwen2.5-0.5B-Instruct, prompt layout B (question first).
  - Held-out family T2: the highest pilot ASR among T2-T4 (pre-written rule).
  - Train families T1, T3, T4 with paraphrases 0-5; paraphrases 6-7 of those
    families are held back for the "seen family, new wording" test row.
    Which paraphrases are held back is fixed by index, not by pilot results.

Leakage guards (checked again in tests/test_final_data.py):
  - Train uses SQuAD train passages (excluding pilot passages) and train
    nonces; tests use SQuAD validation passages not already used by the SEP
    or clean test files, and test nonces.
  - Every passage is used at most once across all files.

Run after src.data:  python -m src.build_templates
"""

import hashlib
import json
import random
from collections import deque
from itertools import cycle
from pathlib import Path

from src.attacks import (
    FAMILY_NAMES, N_TRAIN_PARAPHRASES, NONCE_PATH, TEMPLATES,
    load_nonces, place, positions_for, render, template_id,
)
from datasets import load_dataset

from src.data import SQUAD_REPO, SQUAD_REVISION, SYSTEM_PROMPT, build_messages, candidate_questions, write_jsonl

DATA = Path(__file__).resolve().parent.parent / "data"
SEED = 4242
LAYOUT = "B"
HELD_OUT_FAMILY = "T2"
TRAIN_FAMILIES = ("T1", "T3", "T4")
TRAIN_PARAPHRASES = tuple(range(N_TRAIN_PARAPHRASES))   # 0-5
HELD_BACK_PARAPHRASES = (6, 7)

N_TRAIN_INJECTED, N_TRAIN_CLEAN = 1050, 450
N_TEST_UNSEEN = 300
N_TEST_SEEN_PER_COMBO = 10   # 3 families x 2 paraphrases x 3 positions x 10 = 180


def read_jsonl(path):
    return [json.loads(line) for line in path.open()]


def combos(families, paraphrases):
    return [(f, k, pos) for f in families for k in paraphrases for pos in positions_for(f)]


def spread(combo_list, n, rng):
    """n items cycling through combos so every combo appears floor/ceil(n/len) times."""
    order = list(combo_list)
    rng.shuffle(order)
    out = [order[i % len(order)] for i in range(n)]
    rng.shuffle(out)
    return out


def make_row(split, q, family=None, k=None, position=None, nonce=None):
    if family is None:
        doc, text = q["context"], None
    else:
        text = render(family, k, nonce)
        doc = place(q["context"], text, position, q["gold_answers"])
    return {
        "prompt": build_messages(doc, q["question"], layout=LAYOUT),
        "document": doc,
        "question": q["question"],
        "gold_answers": q["gold_answers"],
        "injected": family is not None,
        "witness": nonce,
        "probe_idx": None,
        "insistent": None,
        "position": position,
        "injection": text,
        "family": family,
        "template_id": None if family is None else template_id(family, k),
        "squad_id": q["squad_id"],
        "title": q["title"],
    }


def build_split(name, specs, pool, nonce_iter, n_clean=0, rng=None):
    rows = [make_row(name, pool.popleft(), f, k, pos, next(nonce_iter)) for f, k, pos in specs]
    rows += [make_row(name, pool.popleft()) for _ in range(n_clean)]
    rng.shuffle(rows)
    return [{"id": f"{name}-{i:05d}", **r} for i, r in enumerate(rows)]


def nonce_cycle(nonces, rng):
    order = list(nonces)
    rng.shuffle(order)
    return cycle(order)


def main():
    rng = random.Random(SEED)
    nonces = load_nonces()

    # Exclude every *passage* already used by the SEP/clean test files or the
    # pilot (looked up via squad_id, since those rows may hold a different
    # question about the same passage, and SEP documents contain injections).
    used_ids = {r["squad_id"] for name in ("test_attack", "test_clean", "pilot")
                for r in read_jsonl(DATA / f"{name}.jsonl")}
    used_contexts = set()
    for split in ("train", "validation"):
        ds = load_dataset(SQUAD_REPO, split=split, revision=SQUAD_REVISION)
        used_contexts |= {c for i, c in zip(ds["id"], ds["context"]) if i in used_ids}
    train_pool = deque(q for q in candidate_questions("train", rng) if q["context"] not in used_contexts)
    val_pool = deque(q for q in candidate_questions("validation", rng) if q["context"] not in used_contexts)

    train_specs = spread(combos(TRAIN_FAMILIES, TRAIN_PARAPHRASES), N_TRAIN_INJECTED, rng)
    train = build_split("train", train_specs, train_pool, nonce_cycle(nonces["train"], rng),
                        N_TRAIN_CLEAN, rng)

    unseen_specs = spread(combos([HELD_OUT_FAMILY], range(len(TEMPLATES[HELD_OUT_FAMILY]))),
                          N_TEST_UNSEEN, rng)
    test_unseen = build_split("test_unseen_family", unseen_specs, val_pool,
                              nonce_cycle(nonces["test"], rng), rng=rng)

    seen_specs = combos(TRAIN_FAMILIES, HELD_BACK_PARAPHRASES) * N_TEST_SEEN_PER_COMBO
    test_seen = build_split("test_seen_family", seen_specs, val_pool,
                            nonce_cycle(nonces["test"], rng), rng=rng)

    files = {"train": train, "test_unseen_family": test_unseen, "test_seen_family": test_seen}
    hashes = {k: write_jsonl(DATA / f"{k}.jsonl", v) for k, v in files.items()}
    manifest = {
        "seed": SEED,
        "layout": LAYOUT,
        "system_prompt": SYSTEM_PROMPT,
        "squad_revision": SQUAD_REVISION,
        "nonces_sha256": hashlib.sha256(NONCE_PATH.read_bytes()).hexdigest(),
        "held_out_family": HELD_OUT_FAMILY,
        "train_families": list(TRAIN_FAMILIES),
        "family_names": FAMILY_NAMES,
        "train_paraphrases": list(TRAIN_PARAPHRASES),
        "held_back_paraphrases": list(HELD_BACK_PARAPHRASES),
        "counts": {k: len(v) for k, v in files.items()},
        "sha256": hashes,
    }
    (DATA / "manifest_templates.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
