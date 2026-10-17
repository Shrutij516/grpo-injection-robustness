# grpo-injection-robustness

GRPO + LoRA fine-tuning of Qwen2.5-Instruct to resist indirect prompt injection without losing QA ability.

- Plan: [PLAN.md](PLAN.md)
- Data sources, verified fields, build decisions: [DATA_NOTES.md](DATA_NOTES.md)
- Step-by-step Colab cells: [COLAB.md](COLAB.md)

Status: steps 1–3 done (environment; frozen data, rewards, stats, tests; evaluation script).

```bash
python -m pytest -q tests/
```

## Attack sets and results

- **SEP probes (first attack set):** baseline attack success was **0%** for Qwen2.5-1.5B-Instruct on all 360 test attacks, with or without the defensive system prompt. The checks confirmed this is real, not a bug: the injection is present in the prompts, the scorer passes a positive control, and no witness word appears in any raw output. On this task (extractive QA with short answers), the model simply ignores polite side-questions. SEP is kept as an extra evaluation row.
- **Harder attack set:** four template families (direct override, fake end-of-document, fake system message, new-session/faux-reasoning). Each injection asks the model to reply with a made-up nonce word. These are **hand-written generic patterns informed by** Zou et al., arXiv:2507.20526; **no attack text was copied from that paper.** One family is held out entirely for testing. Design: [PROPOSAL_harder_attacks.md](PROPOSAL_harder_attacks.md).

### Pilot baseline ASR per template family (base model, training-pool passages)

| Family | Pilot ASR (95% CI) | Held out? |
|---|---|---|
| T1 Direct override | *pending* | no (always trained) |
| T2 Fake end-of-document | *pending* | |
| T3 Fake system message | *pending* | |
| T4 New session / faux reasoning | *pending* | |

### Final results

*Pending.*

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

**Related work informing the attack design:** Zou et al., *Security Challenges in AI Agent Deployment: Insights from a Large Scale Public Competition*, arXiv:2507.20526 (2025). Only generic strategy patterns from this paper are used; no attack text is copied from it.

The files in `data/` are derived from SQuAD and SEP, so they are shared under **CC BY-SA 4.0**. We modified the source data in three ways: we inserted SEP probe text into SQuAD passages, selected subsets, and split them into train and test sets.

```bibtex
@inproceedings{zverev2025,
  title     = {Can {LLMs} Separate Instructions From Data? And What Do We Even Mean By That?},
  author    = {Egor Zverev and Sahar Abdelnabi and Soroush Tabesh and Mario Fritz and Christoph H. Lampert},
  booktitle = {ICLR},
  year      = {2025}
}
@article{zou2025security,
  title   = {Security Challenges in {AI} Agent Deployment: Insights from a Large Scale Public Competition},
  author  = {Andy Zou and Maxwell Lin and Eliot Jones and Micha Nowak and Mateusz Dziemian and Nick Winter and Alexander Grattan and Valent Nathanael and Ayla Croft and Xander Davies and Jai Patel and Robert Kirk and Nate Burnikell and Yarin Gal and Dan Hendrycks and J. Zico Kolter and Matt Fredrikson},
  journal = {arXiv preprint arXiv:2507.20526},
  year    = {2025}
}
@inproceedings{rajpurkar2016squad,
  title     = {{SQuAD}: 100,000+ Questions for Machine Comprehension of Text},
  author    = {Pranav Rajpurkar and Jian Zhang and Konstantin Lopyrev and Percy Liang},
  booktitle = {EMNLP},
  year      = {2016}
}
```
