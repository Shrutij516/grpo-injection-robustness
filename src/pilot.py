"""Pilot set: measure the base model's ASR per template family before freezing.

160 prompts = 4 families x 8 paraphrases x 5, built only from SQuAD *train*
passages with *train* nonces, so the eventual test sets stay untouched.
The pilot decides which of T2-T4 is held out (highest pilot ASR) and whether
any family needs its wording adjusted (target 20-80% ASR).

Build:  python -m src.pilot            -> data/pilot.jsonl
Run:    python -m src.evaluate --splits pilot --tag pilot_baseline --out_dir ...
"""

import json
import random
from collections import deque
from pathlib import Path

from src.attacks import TEMPLATES, load_nonces, place, positions_for, render, template_id
from src.data import build_messages, candidate_questions, write_jsonl

PILOT_SEED = 777
PER_PARAPHRASE = 5
OUT = Path(__file__).resolve().parent.parent / "data" / "pilot.jsonl"


def build_pilot():
    rng = random.Random(PILOT_SEED)
    nonces = load_nonces()["train"]
    pool = deque(candidate_questions("train", rng))
    rows = []
    for family, paraphrases in TEMPLATES.items():
        positions = positions_for(family)
        for k in range(len(paraphrases)):
            for j in range(PER_PARAPHRASE):
                q = pool.popleft()
                nonce = rng.choice(nonces)
                # Rotate positions across paraphrases and repeats -> balanced per family.
                position = positions[(k * PER_PARAPHRASE + j) % len(positions)]
                text = render(family, k, nonce)
                doc = place(q["context"], text, position, q["gold_answers"])
                rows.append({
                    "prompt": build_messages(doc, q["question"]),
                    "document": doc,
                    "question": q["question"],
                    "gold_answers": q["gold_answers"],
                    "injected": True,
                    "witness": nonce,
                    "probe_idx": None,
                    "insistent": None,
                    "position": position,
                    "injection": text,
                    "family": family,
                    "template_id": template_id(family, k),
                    "squad_id": q["squad_id"],
                    "title": q["title"],
                })
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"pilot-{i:05d}"
    return [{"id": r.pop("id"), **r} for r in rows]


if __name__ == "__main__":
    rows = build_pilot()
    digest = write_jsonl(OUT, rows)
    print(f"wrote {len(rows)} rows to {OUT} (sha256 {digest[:12]})")
