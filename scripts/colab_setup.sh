#!/usr/bin/env bash
# Run from the repo root on Colab after mounting Drive and cloning.
# Installs pinned deps on top of Colab's preinstalled CUDA torch, then
# prints versions + GPU so every run log records its environment.
set -euo pipefail

pip install -q -r requirements.txt

python - <<'PY'
import torch, transformers, trl, peft, accelerate, datasets, platform
print("python", platform.python_version())
for m in (torch, transformers, trl, peft, accelerate, datasets):
    print(f"{m.__name__:13s} {m.__version__}")
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
PY
