#!/usr/bin/env bash
# Run from the repo root on Colab after mounting Drive and cloning:
#   !bash scripts/colab_setup.sh
# Installs pinned deps on top of Colab's preinstalled CUDA torch, checks the
# frozen data and reward logic with the unit tests, and prints the environment
# so every run log records it.
set -euo pipefail

pip install -q -r requirements.txt

python -m pytest -q tests/

python - <<'PY'
import torch, transformers, trl, peft, accelerate, datasets, platform
print("python", platform.python_version())
for m in (torch, transformers, trl, peft, accelerate, datasets):
    print(f"{m.__name__:13s} {m.__version__}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
else:
    print("WARNING: no GPU. Runtime > Change runtime type > T4 GPU")
PY
