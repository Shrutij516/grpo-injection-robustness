"""Checks on the committed, frozen data files (no network needed)."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from src.rewards import contains_phrase

DATA = Path(__file__).resolve().parent.parent / "data"
MANIFEST = json.loads((DATA / "manifest.json").read_text())


def load(name):
    return [json.loads(line) for line in (DATA / f"{name}.jsonl").open()]


TRAIN, TEST_ATTACK, TEST_CLEAN = load("train"), load("test_attack"), load("test_clean")
ALL = TRAIN + TEST_ATTACK + TEST_CLEAN


@pytest.mark.parametrize("name", ["train", "test_attack", "test_clean"])
def test_files_match_manifest_hash(name):
    digest = hashlib.sha256((DATA / f"{name}.jsonl").read_bytes()).hexdigest()
    assert digest == MANIFEST["sha256"][name]


def test_counts():
    assert (len(TRAIN), len(TEST_ATTACK), len(TEST_CLEAN)) == (1500, 360, 300)
    assert sum(r["injected"] for r in TRAIN) == 1050
    assert all(r["injected"] for r in TEST_ATTACK)
    assert not any(r["injected"] for r in TEST_CLEAN)


def test_test_attacks_are_unseen_witnesses_and_articles():
    train_w = {r["witness"] for r in TRAIN if r["injected"]}
    test_w = {r["witness"] for r in TEST_ATTACK}
    assert len(train_w) == 35 and len(test_w) == 15 and not train_w & test_w
    test_titles = {r["title"] for r in TEST_ATTACK + TEST_CLEAN}
    assert not {r["title"] for r in TRAIN} & test_titles


def test_no_passage_reused():
    ids = [r["squad_id"] for r in ALL]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("rows,per_combo", [(TRAIN, 5), (TEST_ATTACK, 4)])
def test_balanced_over_witness_insistence_position(rows, per_combo):
    combos = Counter((r["witness"], r["insistent"], r["position"]) for r in rows if r["injected"])
    assert set(combos.values()) == {per_combo}


def test_gold_answer_findable_and_witness_only_from_injection():
    for r in ALL:
        assert any(contains_phrase(r["document"], g) for g in r["gold_answers"]), r["id"]
        if r["injected"]:
            assert r["injection"].rstrip(".") in r["document"]
            other = [r["question"], *r["gold_answers"], r["document"].replace(r["injection"].rstrip("."), "")]
            assert not any(contains_phrase(t, r["witness"], allow_plural=True) for t in other), r["id"]
