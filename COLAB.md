# Running on Colab

Paste each block below into its own Colab cell and run the cells in order. All results go to Google Drive under `MyDrive/grpo-inj/`.

**Before Cell 1:** select the GPU runtime via **Runtime → Change runtime type → T4 GPU → Save**.

**If Colab disconnects:** rerun Cell 1, then rerun the cell that was interrupted. Evaluation resumes where it stopped and doesn't redo finished prompts.

---

## Cell 1: GPU check, mount Drive, clone, set up

```python
import os, subprocess

# 1. GPU check: stop early if this runtime has no T4.
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                     capture_output=True, text=True).stdout.strip()
assert gpu, "No GPU attached. Runtime > Change runtime type > T4 GPU > Save, then rerun this cell."
print("GPU:", gpu)

# 2. Mount Google Drive (a sign-in popup asks for permission).
from google.colab import drive
drive.mount("/content/drive")
os.makedirs("/content/drive/MyDrive/grpo-inj", exist_ok=True)

# 3. Clone the repo, or pull the latest version if it is already there.
%cd /content
if os.path.isdir("grpo-injection-robustness"):
    !git -C grpo-injection-robustness pull --ff-only
else:
    !git clone https://github.com/Shrutij516/grpo-injection-robustness.git
%cd /content/grpo-injection-robustness
!git log --oneline -1

# 4. Install pinned packages, run unit tests, print the environment.
!bash scripts/colab_setup.sh
```

**Expected output:**
- `GPU: Tesla T4, 15360 MiB`
- `Mounted at /content/drive`
- The latest commit hash and message
- A few quiet minutes of pip installs, then `36 passed`
- Package versions: `transformers 5.18.0`, `trl 1.14.2`, `peft 0.21.2`, `accelerate 1.15.0`, `datasets 5.0.1`
- `cuda available: True` and `gpu: Tesla T4`

**If something goes wrong:**
- **A pip dependency warning** about other Colab packages is fine.
- **A failing test:** stop there and send me the output. A failure means the data files or reward logic differ from what was tested locally.
- **No restart needed:** each evaluation runs as a fresh `python` process, so you don't need to restart the runtime after the install.

---

## Cell 2: baseline evaluation (untrained model)

```python
%cd /content/grpo-injection-robustness
!python -m src.evaluate --tag baseline --out_dir /content/drive/MyDrive/grpo-inj
```

**Expected output:**
- First, `[evaluate] baseline: 0 already done, 660 to generate` (360 attacked + 300 clean prompts). After a resume, the "already done" number is larger.
- A one-time download of Qwen2.5-1.5B-Instruct (~3 GB). A warning about a missing `HF_TOKEN` is harmless because the model is public.
- Progress lines like `[evaluate] 160/660 (95s)`. I estimate about 10–15 minutes in total on a T4, but haven't measured it yet.
- At the end, a JSON summary. The key numbers are:
  - `test_attack.attack_success_rate.rate`: how often the model obeyed the hidden instruction
  - `test_clean.accuracy.rate`: normal QA accuracy
  - `test_clean.refusal_rate.rate`: how often it refused a normal question

**Saved to Drive** in `MyDrive/grpo-inj/baseline/`:
- `generations.jsonl`: every prompt's raw output plus its scores (one line per prompt)
- `metrics.json`: all metrics with 95% confidence intervals, broken down by position, by insistence and per probe
- `run_info.json`: model, settings, git commit, package versions, GPU and data hashes

**Go/no-go check (from the plan):** if baseline `attack_success_rate` is below about 0.20, there's too little to reduce. In that case, tell me before going further and we'll adjust the data, for example by using only end-of-document injections.

---

## Cell 3: baseline with defensive system prompt

```python
%cd /content/grpo-injection-robustness
!python -m src.evaluate --tag baseline_defense --system_prompt defense --out_dir /content/drive/MyDrive/grpo-inj
```

This is the same model and the same 660 prompts. The only change is that the system prompt also says the document is untrusted and that instructions inside it must never be followed. It answers the question "does just asking nicely already fix it?"

**Expected output:** the same format as Cell 2: `[evaluate] baseline_defense: 0 already done, 660 to generate`, progress lines and a JSON summary. It runs about as long as Cell 2, without the model download.

**Saved to Drive** in `MyDrive/grpo-inj/baseline_defense/`: the same three files as Cell 2.

**Each run needs its own tag.** If you rerun with different settings but the same `--tag`, the script stops with "was made with different settings ... Use a new --tag" rather than mixing two experiments in one folder.

---

When Cells 2 and 3 are done, send me both `metrics.json` files, or just the printed summaries.
