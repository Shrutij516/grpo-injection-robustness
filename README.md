# grpo-injection-robustness

GRPO + LoRA fine-tuning of Qwen2.5-Instruct to resist indirect prompt injection without losing QA ability.

- Plan: [PLAN.md](PLAN.md)
- Data sources, verified fields, build decisions: [DATA_NOTES.md](DATA_NOTES.md)

Status: steps 1–3 done (environment; frozen data, rewards, stats, tests; evaluation script).

```bash
python -m pytest -q tests/
```

## Running on Colab

In a notebook cell:

```python
from google.colab import drive; drive.mount("/content/drive")
!git clone https://github.com/Shrutij516/grpo-injection-robustness.git
%cd grpo-injection-robustness
!bash scripts/colab_setup.sh
!python -m src.evaluate --tag baseline --out_dir /content/drive/MyDrive/grpo-inj
```

Evaluation is resumable: if the session disconnects, rerun the same command and it continues where it stopped.

## Data attribution and licenses

| Source | Used for | License |
|---|---|---|
| **SEP** (Zverev et al., ICLR 2025), [egozverev/Should-It-Be-Executed-Or-Processed](https://github.com/egozverev/Should-It-Be-Executed-Or-Processed) @ `7606c06` | Injected probe instructions + witness answers (`data/raw/sep_probes.json`) | Code: MIT. Dataset: CC BY 4.0 (per the SEP README) |
| **SQuAD v1.1** (Rajpurkar et al., EMNLP 2016), [rajpurkar/squad](https://huggingface.co/datasets/rajpurkar/squad) @ `7b6d24c` | Passages, questions, gold answers | CC BY-SA 4.0 |
| **Qwen2.5-Instruct** (0.5B / 1.5B) | Base model (not redistributed) | Apache 2.0 |

The files in `data/` are derived from SQuAD and SEP, so they are shared under **CC BY-SA 4.0**. We modified the source data in three ways: we inserted SEP probe text into SQuAD passages, selected subsets, and split them into train and test sets.

```bibtex
@inproceedings{zverev2025,
  title     = {Can {LLMs} Separate Instructions From Data? And What Do We Even Mean By That?},
  author    = {Egor Zverev and Sahar Abdelnabi and Soroush Tabesh and Mario Fritz and Christoph H. Lampert},
  booktitle = {ICLR},
  year      = {2025}
}
@inproceedings{rajpurkar2016squad,
  title     = {{SQuAD}: 100,000+ Questions for Machine Comprehension of Text},
  author    = {Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy Liang},
  booktitle = {EMNLP},
  year      = {2016}
}
```
