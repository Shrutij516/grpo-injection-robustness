from src.evaluate import compute_metrics
from src.rewards import score_completion
from src.summary import table


def test_table_renders_all_rows():
    gens = []
    for split, fam, tid, w, out in [("test_unseen_family", "T2", "T2-0", "Zaxnubre", "Zaxnubre"),
                                    ("test_seen_family", "T1", "T1-6", "Zaxnubre", "1943"),
                                    ("test_attack", None, None, "Paris", "1943"),
                                    ("test_clean", None, None, None, "1943")]:
        gens.append({"id": split, "split": split, "witness": w, "position": "end", "insistent": None,
                     "family": fam, "template_id": tid, **score_completion(out, ["1943"], w)})
    m = compute_metrics(gens)
    m.update(tag="t", model="Qwen/Qwen2.5-0.5B-Instruct", layout="B", system_prompt="defense", adapter=None)
    out = table(m)
    assert "system prompt 'defense'" in out and "untrained" in out
    assert "Unseen family T2" in out and "seen family T1" in out and "SEP probes" in out
    assert "Clean QA accuracy" in out
