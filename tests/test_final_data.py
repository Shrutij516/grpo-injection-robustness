"""Leakage and integrity checks for the frozen template-attack files."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.attacks import load_nonces, render
from src.data import build_messages
from src.rewards import contains_phrase

DATA = Path(__file__).resolve().parent.parent / "data"
MANIFEST = json.loads((DATA / "manifest_templates.json").read_text())
NONCES = load_nonces()


def load(name):
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").open()]


TRAIN = load("train")
UNSEEN = load("test_unseen_family")
SEEN = load("test_seen_family")
SEP = load("test_attack")
CLEAN = load("test_clean")
PILOT = load("pilot")
TESTS = {"test_unseen_family": UNSEEN, "test_seen_family": SEEN, "test_attack": SEP, "test_clean": CLEAN}
ALL_TEST = UNSEEN + SEEN + SEP + CLEAN
TRAIN_INJ = [r for r in TRAIN if r["injected"]]


@pytest.mark.parametrize("name", ["train", "test_unseen_family", "test_seen_family"])
def test_files_match_manifest_hash(name):
    digest = hashlib.sha256((DATA / f"{name}.jsonl").read_bytes()).hexdigest()
    assert digest == MANIFEST["sha256"][name]


def test_nonce_file_matches_manifest():
    digest = hashlib.sha256((DATA / "raw" / "nonces.json").read_bytes()).hexdigest()
    assert digest == MANIFEST["nonces_sha256"]


def test_counts_and_family_assignment():
    assert len(TRAIN) == 1500 and len(TRAIN_INJ) == 1050
    assert len(UNSEEN) == 300 and len(SEEN) == 180
    assert {r["family"] for r in TRAIN_INJ} == {"T1", "T3", "T4"}
    assert {r["family"] for r in UNSEEN} == {"T2"}
    assert {r["family"] for r in SEEN} == {"T1", "T3", "T4"}
    assert not any(r.get("family") == "T2" for r in TRAIN)


def test_paraphrase_split():
    train_k = {int(r["template_id"].split("-")[1]) for r in TRAIN_INJ}
    seen_k = {int(r["template_id"].split("-")[1]) for r in SEEN}
    assert train_k == {0, 1, 2, 3, 4, 5} and seen_k == {6, 7}
    assert len({r["template_id"] for r in UNSEEN}) == 8


# ---- leakage: nothing from training appears in any test file ---------------

def test_no_train_nonce_in_any_test_file():
    train_nonces = set(NONCES["train"])
    for name, rows in TESTS.items():
        for r in rows:
            text = r["document"] + " " + r["question"]
            assert not any(contains_phrase(text, n, allow_plural=True) for n in train_nonces), (name, r["id"])
    assert {r["witness"] for r in UNSEEN + SEEN} <= set(NONCES["test"])
    assert {r["witness"] for r in TRAIN_INJ} <= train_nonces


def test_no_train_paraphrase_in_any_test_file():
    train_ids = {r["template_id"] for r in TRAIN_INJ}
    train_texts = {render(r["family"], int(r["template_id"].split("-")[1]), "{NONCE}")
                   for r in TRAIN_INJ}
    for name, rows in TESTS.items():
        for r in rows:
            assert r.get("template_id") not in train_ids, (name, r["id"])
            if r.get("template_id"):
                tmpl = r["injection"].replace(r["witness"], "{NONCE}")
                assert tmpl not in train_texts, (name, r["id"])


def test_no_train_passage_or_passage_probe_pair_in_any_test_file():
    train_passages = {r["squad_id"] for r in TRAIN}
    train_titles = {r["title"] for r in TRAIN}
    train_pairs = {(r["squad_id"], r["template_id"], r["witness"]) for r in TRAIN_INJ}
    for name, rows in TESTS.items():
        for r in rows:
            assert r["squad_id"] not in train_passages, (name, r["id"])
            assert r["title"] not in train_titles, (name, r["id"])
            assert (r["squad_id"], r.get("template_id"), r["witness"]) not in train_pairs


def test_each_passage_used_once_across_all_files():
    ids = [r["squad_id"] for r in TRAIN + ALL_TEST + PILOT]
    assert len(ids) == len(set(ids))
    clean_docs = [" ".join(r["document"].split()) for r in TRAIN + CLEAN if not r["injected"]]
    assert len(clean_docs) == len(set(clean_docs))


# ---- balance and integrity ---------------------------------------------------

def test_train_positions_and_paraphrases_balanced():
    c = Counter((r["template_id"], r["position"]) for r in TRAIN_INJ)
    assert len(c) == 54 and max(c.values()) - min(c.values()) <= 1
    n = Counter(r["witness"] for r in TRAIN_INJ)
    assert len(n) == 60 and max(n.values()) - min(n.values()) <= 1


def test_test_rows_balanced():
    assert all(r["position"] == "end" for r in UNSEEN)
    u = Counter(r["template_id"] for r in UNSEEN)
    assert max(u.values()) - min(u.values()) <= 1
    s = Counter((r["template_id"], r["position"]) for r in SEEN)
    assert len(s) == 18 and set(s.values()) == {10}


@pytest.mark.parametrize("rows", [TRAIN, UNSEEN, SEEN, CLEAN], ids=["train", "unseen", "seen", "clean"])
def test_stored_prompts_are_layout_b(rows):
    for r in rows:
        assert r["prompt"] == build_messages(r["document"], r["question"], layout="B")


def test_nonce_only_from_injection_and_gold_findable():
    for r in TRAIN + UNSEEN + SEEN:
        assert any(contains_phrase(r["document"], g) for g in r["gold_answers"]), r["id"]
        if r["injected"]:
            assert r["injection"] in r["document"]
            rest = [r["document"].replace(r["injection"], ""), r["question"], *r["gold_answers"]]
            assert not any(contains_phrase(t, r["witness"], allow_plural=True) for t in rest), r["id"]
