"""Prompt layouts: A (document, then question) and B (question, document, "Answer:")."""

import json
from pathlib import Path

import pytest

from src.data import LAYOUTS, SYSTEM_PROMPT, build_messages
from src.evaluate import check_same_settings, run_header

DATA = Path(__file__).resolve().parent.parent / "data"
ROWS = [json.loads(line) for name in ("pilot", "test_attack", "test_clean")
        for line in (DATA / f"{name}.jsonl").open()]
INJECTED = [r for r in ROWS if r["injected"]]


def user_text(r, layout):
    return build_messages(r["document"], r["question"], SYSTEM_PROMPT, layout)[1]["content"]


def test_layout_a_matches_frozen_prompts():
    # Default layout must reproduce exactly what the frozen files contain.
    for r in ROWS:
        assert build_messages(r["document"], r["question"]) == r["prompt"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_question_appears_exactly_once(layout):
    for r in ROWS:
        assert user_text(r, layout).count(r["question"]) == 1, r["id"]


@pytest.mark.parametrize("layout", LAYOUTS)
def test_injection_stays_inside_document_block(layout):
    # The real block opens at the first "<<<" and closes at the last ">>>";
    # T2 templates contain fake delimiters, which must stay between those.
    for r in INJECTED:
        u = user_text(r, layout)
        open_at = u.index("Document:\n<<<\n") + len("Document:\n<<<\n")
        close_at = u.rindex("\n>>>")
        start = u.index(r["injection"])
        assert open_at <= start and start + len(r["injection"]) <= close_at, r["id"]


def test_layout_order():
    r = next(r for r in INJECTED if r.get("family") == "T1")
    a, b = user_text(r, "A"), user_text(r, "B")
    assert a.index(r["injection"]) < a.index(r["question"])
    assert b.index(r["question"]) < b.index(r["injection"])
    assert b.rstrip().endswith("Answer:")


def test_unknown_layout_rejected():
    with pytest.raises(ValueError):
        build_messages("d", "q", layout="C")


class Args:
    tag, model, adapter, system_prompt, max_new_tokens, layout = "t", "m", None, "default", 96, "B"


def test_layout_saved_in_metrics_header():
    assert run_header(Args)["layout"] == "B"


def test_resume_with_different_layout_is_refused(tmp_path):
    info = tmp_path / "run_info.json"
    info.write_text(json.dumps({"args": {"model": "m", "adapter": None, "system_prompt": "default",
                                         "max_new_tokens": 96, "layout": "A"}}))
    with pytest.raises(SystemExit, match="layout"):
        check_same_settings(info, Args)


def test_runs_saved_before_layout_existed_count_as_layout_a(tmp_path):
    info = tmp_path / "run_info.json"
    info.write_text(json.dumps({"args": {"model": "m", "adapter": None, "system_prompt": "default",
                                         "max_new_tokens": 96}}))
    class OldA(Args):
        layout = "A"
    check_same_settings(info, OldA)  # no error: old runs were layout A
    with pytest.raises(SystemExit):
        check_same_settings(info, Args)  # layout B must use a new tag
