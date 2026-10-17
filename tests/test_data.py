"""Checks on the frozen SEP attack file and the clean QA file (no network needed)."""

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


TEST_ATTACK, TEST_CLEAN = load("test_attack"), load("test_clean")
ALL = TEST_ATTACK + TEST_CLEAN


@pytest.mark.parametrize("name", ["test_attack", "test_clean"])
def test_files_match_manifest_hash(name):
    digest = hashlib.sha256((DATA / f"{name}.jsonl").read_bytes()).hexdigest()
    assert digest == MANIFEST["sha256"][name]


def test_sep_file_unchanged_since_first_freeze():
    # The SEP test file must stay byte-identical to the step-2 freeze.
    assert MANIFEST["sha256"]["test_attack"] == (
        "4f7bfbe7ca41f4be8796e4976a5ff18af6cfa5fbaf349278f6dfbeb0d0e95951")


def test_counts():
    assert (len(TEST_ATTACK), len(TEST_CLEAN)) == (360, 300)
    assert all(r["injected"] for r in TEST_ATTACK)
    assert not any(r["injected"] for r in TEST_CLEAN)


def test_sep_test_witnesses_are_the_15_held_out():
    test_w = {r["witness"] for r in TEST_ATTACK}
    assert test_w == set(MANIFEST["test_witnesses"])
    assert len(test_w) == 15 and not test_w & set(MANIFEST["train_witnesses"])


def test_no_passage_reused():
    ids = [r["squad_id"] for r in ALL]
    assert len(ids) == len(set(ids))


def test_sep_balanced_over_witness_insistence_position():
    combos = Counter((r["witness"], r["insistent"], r["position"]) for r in TEST_ATTACK)
    assert set(combos.values()) == {4}


def test_gold_answer_findable_and_witness_only_from_injection():
    for r in ALL:
        assert any(contains_phrase(r["document"], g) for g in r["gold_answers"]), r["id"]
        if r["injected"]:
            assert r["injection"].rstrip(".") in r["document"]
            other = [r["question"], *r["gold_answers"], r["document"].replace(r["injection"].rstrip("."), "")]
            assert not any(contains_phrase(t, r["witness"], allow_plural=True) for t in other), r["id"]
