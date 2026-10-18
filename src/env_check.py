"""Environment checks that must run before peft is imported.

Colab ships torchao 0.10.0. The pinned peft (0.21.2) refuses torchao older
than 0.16.0 and raises deep inside the first LoRA call, so we fail early with
a clear fix instead. Uses only importlib.metadata: importing torchao or peft
here would trigger the very error we are guarding against.
"""

import importlib.metadata as metadata
import platform

from packaging.version import Version

TORCHAO_MIN = Version("0.16.0")
LIBRARIES = ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "torchao")


def installed_version(name):
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def library_versions():
    return {"python": platform.python_version(), **{n: installed_version(n) for n in LIBRARIES}}


def require_compatible_torchao():
    v = installed_version("torchao")
    if v is not None and Version(v) < TORCHAO_MIN:
        raise SystemExit(
            f"[env] torchao {v} is installed, but peft needs torchao >= {TORCHAO_MIN} or no torchao at all. "
            "Fix: run `!pip uninstall -y torchao` (this project does not use it), then rerun.")


def print_versions(prefix="[env]"):
    print(prefix, "  ".join(f"{k}={v or 'not installed'}" for k, v in library_versions().items()), flush=True)
