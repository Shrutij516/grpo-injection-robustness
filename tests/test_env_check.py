import importlib.metadata as metadata

import pytest

from src import env_check


def fake_versions(table):
    def version(name):
        if name not in table:
            raise metadata.PackageNotFoundError(name)
        return table[name]
    return version


@pytest.mark.parametrize("v", ["0.10.0", "0.15.9"])
def test_old_torchao_fails_early_with_fix(monkeypatch, v):
    monkeypatch.setattr(env_check.metadata, "version", fake_versions({"torchao": v}))
    with pytest.raises(SystemExit, match="pip uninstall -y torchao"):
        env_check.require_compatible_torchao()


@pytest.mark.parametrize("table", [{}, {"torchao": "0.16.0"}, {"torchao": "0.18.1"}])
def test_absent_or_new_torchao_is_fine(monkeypatch, table):
    monkeypatch.setattr(env_check.metadata, "version", fake_versions(table))
    env_check.require_compatible_torchao()


def test_version_report_lists_all_libraries(capsys):
    env_check.print_versions("[x]")
    out = capsys.readouterr().out
    for name in ("python", "torch", "transformers", "trl", "peft", "torchao"):
        assert f"{name}=" in out


def test_training_module_runs_the_check_before_peft():
    src = open(env_check.__file__.replace("env_check.py", "train_grpo.py")).read()
    assert src.index("require_compatible_torchao()") < src.index("from peft import")
