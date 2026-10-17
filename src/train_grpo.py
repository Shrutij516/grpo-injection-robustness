"""GRPO + LoRA training with the pre-registered settings (configs/grpo.yaml).

Rewards come straight from src/rewards.py (the same functions evaluation
uses, including near-match nonce scoring). A fourth, zero-weight "monitor"
function only logs extra metrics and writes sample completions.

Everything lands in --output_dir (a Drive folder on Colab):
  run_config.json        resolved config + data hash + versions (resume guard)
  log_history.jsonl      one line per logged step, appended and fsync'd
  samples.jsonl          a few completions every `sample_every` steps
  checkpoint-*/          rolling checkpoints for resume (adapter + optimizer)
  milestones/step-*/     adapter-only copies at every `milestone_every` steps
  final/                 adapter after the last step

Examples:
  python -m src.train_grpo --output_dir /content/drive/MyDrive/grpo-inj/train_seed1
  python -m src.train_grpo --output_dir .../train_smoke_seed1 --stop_after 30
  python -m src.train_grpo --output_dir ... --set grpo.temperature=1.0
Rerunning the same command resumes from the latest checkpoint.
"""

import argparse
import copy
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint
from trl import GRPOConfig, GRPOTrainer

from src.rewards import (
    MAX_ANSWER_WORDS, completion_text, edit_distance, is_refusal, normalize,
    reward_refusal, reward_resist, reward_task, witness_present, word_count,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = ROOT / "configs" / "grpo.yaml"
KEEP_COLUMNS = ("id", "prompt", "gold_answers", "witness", "family", "template_id")


# ---- config ------------------------------------------------------------------

def load_config(path, overrides=()):
    cfg = yaml.safe_load(Path(path).read_text())
    for item in overrides:
        key, value = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node[p]
        node[leaf] = yaml.safe_load(value)
    return cfg


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolved_run_config(cfg):
    """What must match for a resume to be valid (stop_after is not included)."""
    return {"config": cfg, "train_file_sha256": file_sha256(ROOT / cfg["train_file"])}


def check_or_write_run_config(out_dir, cfg):
    path = out_dir / "run_config.json"
    now = resolved_run_config(cfg)
    if path.exists():
        prev = json.loads(path.read_text())
        prev_core = {k: prev[k] for k in now}
        if prev_core != now:
            raise SystemExit(f"[train] {out_dir} was started with a different config or train file. "
                             "Use a new --output_dir (or delete the old one on purpose).")
        return
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=ROOT).stdout.strip()
    except OSError:
        commit = None
    import peft, transformers, trl
    path.write_text(json.dumps({
        **now,
        "git_commit": commit,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "transformers": transformers.__version__, "trl": trl.__version__,
                     "peft": peft.__version__},
    }, indent=2) + "\n")


# ---- monitoring (zero-weight reward function) -------------------------------

def near_miss(text, nonce):
    """Signs of a garbled nonce that the attack scorer does NOT count:
    3-4 edits away, same first 4 letters, or spelled with spaces/punctuation."""
    if not nonce or witness_present(text, nonce):
        return False
    target = nonce.lower()
    for tok in normalize(text).split():
        if len(tok) >= 4 and (tok[:4] == target[:4] or edit_distance(tok, target, limit=4) <= 4):
            return True
    squashed = "".join(normalize(text).split())
    return target in squashed


def make_monitor(out_dir, sample_every, n_samples=4):
    """Returns a reward function (weight 0) that logs the Section 4 metrics."""
    written = set()

    def monitor(prompts, completions, gold_answers, witness, log_metric=None, trainer_state=None, **kwargs):
        texts = [completion_text(c) for c in completions]
        injected = [i for i, w in enumerate(witness) if w]
        if log_metric is not None:
            words = [word_count(t) for t in texts]
            log_metric("train/words_mean", sum(words) / len(words))
            log_metric("train/over50_rate", sum(w > MAX_ANSWER_WORDS for w in words) / len(words))
            log_metric("train/refusal_rate", sum(is_refusal(t) for t in texts) / len(texts))
            if injected:
                log_metric("train/attack_rate",
                           sum(witness_present(texts[i], witness[i]) for i in injected) / len(injected))
                log_metric("train/near_miss_rate",
                           sum(near_miss(texts[i], witness[i]) for i in injected) / len(injected))
        # The batch being scored belongs to the step about to complete (1-based).
        step = (trainer_state.global_step if trainer_state is not None else 0) + 1
        if sample_every and (step == 1 or step % sample_every == 0) and step not in written:
            written.add(step)
            with (out_dir / "samples.jsonl").open("a") as f:
                for i in range(min(n_samples, len(texts))):
                    f.write(json.dumps({"step": step, "witness": witness[i], "gold": gold_answers[i],
                                        "prompt_tail": completion_text(prompts[i])[-300:],
                                        "completion": texts[i]}, ensure_ascii=False) + "\n")
        return [0.0] * len(texts)

    return monitor


# ---- callbacks -----------------------------------------------------------------

class JsonlLogCallback(TrainerCallback):
    """Append every log dict to log_history.jsonl with timing and memory."""

    def __init__(self, path):
        self.path = path
        self.last_time, self.last_step = None, None

    def on_log(self, args, state, control, logs=None, **kwargs):
        now = time.time()
        rec = {"step": state.global_step, **(logs or {})}
        if self.last_time is not None and state.global_step > self.last_step:
            rec["step_time_s"] = (now - self.last_time) / (state.global_step - self.last_step)
        if torch.cuda.is_available():
            rec["gpu_peak_mem_gb"] = torch.cuda.max_memory_allocated() / 1e9
        self.last_time, self.last_step = now, state.global_step
        with self.path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())


class MilestoneCallback(TrainerCallback):
    """Keep adapter-only copies at fixed steps for the training curve."""

    def __init__(self, out_dir, every):
        self.out_dir, self.every = out_dir, every

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if self.every and state.global_step % self.every == 0 and model is not None:
            model.save_pretrained(self.out_dir / "milestones" / f"step-{state.global_step:04d}")


class StopAfterCallback(TrainerCallback):
    """Stop (and checkpoint) after N steps without changing the LR schedule,
    so a smoke run is exactly the first N steps of the main run."""

    def __init__(self, n):
        self.n = n

    def on_step_end(self, args, state, control, **kwargs):
        if self.n and state.global_step >= self.n:
            control.should_training_stop = True
            control.should_save = True


# ---- build -------------------------------------------------------------------

def precision_settings(name):
    if name == "fp16_base":
        return torch.float16, True
    if name == "fp32_base_amp":
        return torch.float32, True
    if name == "fp32":
        return torch.float32, False
    raise ValueError(f"unknown precision {name!r}")


def build_trainer(cfg, out_dir, stop_after=None, use_cpu=False):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    check_or_write_run_config(out_dir, cfg)

    dtype, fp16 = precision_settings(cfg["precision"])
    if fp16 and (use_cpu or not torch.cuda.is_available()):
        raise SystemExit(f"[train] precision {cfg['precision']} needs a CUDA GPU; use --set precision=fp32 on CPU")

    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], dtype=dtype)

    ds = load_dataset("json", data_files=str(ROOT / cfg["train_file"]), split="train")
    ds = ds.remove_columns([c for c in ds.column_names if c not in KEEP_COLUMNS])

    lora = cfg["lora"]
    peft_config = LoraConfig(r=lora["r"], lora_alpha=lora["alpha"], lora_dropout=lora["dropout"],
                             target_modules=lora["target_modules"], task_type="CAUSAL_LM")
    args = GRPOConfig(
        output_dir=str(out_dir),
        fp16=fp16,
        use_cpu=use_cpu,
        report_to="none",
        reward_weights=cfg["reward_weights"],
        log_completions=False,
        **cfg["grpo"],
    )
    monitor = make_monitor(out_dir, cfg["sample_every"])
    callbacks = [JsonlLogCallback(out_dir / "log_history.jsonl"),
                 MilestoneCallback(out_dir, cfg["milestone_every"])]
    if stop_after:
        callbacks.append(StopAfterCallback(stop_after))
    return GRPOTrainer(
        model=model,
        reward_funcs=[reward_task, reward_resist, reward_refusal, monitor],
        args=args,
        train_dataset=ds,
        processing_class=tok,
        peft_config=peft_config,
        callbacks=callbacks,
    )


def train(cfg, out_dir, stop_after=None, use_cpu=False):
    trainer = build_trainer(cfg, out_dir, stop_after, use_cpu)
    last = get_last_checkpoint(str(out_dir)) if Path(out_dir).exists() else None
    print(f"[train] {'resuming from ' + last if last else 'starting fresh'} -> {out_dir}", flush=True)
    trainer.train(resume_from_checkpoint=last)
    if not stop_after or trainer.state.global_step >= cfg["grpo"]["max_steps"]:
        trainer.save_model(str(Path(out_dir) / "final"))
    return trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--stop_after", type=int, default=None,
                    help="stop after N steps, same LR schedule as the full run (smoke test)")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a config value, e.g. grpo.temperature=1.0 (recorded in run_config.json)")
    ap.add_argument("--use_cpu", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.set)
    train(copy.deepcopy(cfg), args.output_dir, args.stop_after, args.use_cpu)


if __name__ == "__main__":
    main()
