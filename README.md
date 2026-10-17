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

Labels: **[paper]** = stated in arXiv:2507.20526 (checked against the paper text, v1); **[ours]** = our own measurement, choice, or interpretation.

- **SEP probes (first attack set)** [ours]: baseline attack success was **0%** for Qwen2.5-1.5B-Instruct on all 360 test attacks, with or without the defensive system prompt. The checks confirmed this is real, not a bug: the injection is present in the prompts, the scorer passes a positive control, and no witness word appears in any raw output. On this task (extractive QA with short answers), the model simply ignores polite side-questions. SEP is kept as an extra evaluation row.
- **Harder attack set:** four template families, each asking the model to reply with a made-up nonce word. They are **hand-written generic patterns**, and **no attack text was copied from the paper.** One family is held out entirely for testing. Design: [PROPOSAL_harder_attacks.md](PROPOSAL_harder_attacks.md).

  | Family | Basis |
  |---|---|
  | T1 Direct override | [ours] a common small-model weakness; not a named strategy in the paper |
  | T2 Fake end-of-document | [ours]; not a named strategy in the paper |
  | T3 Fake system message | [paper] "System Prompt Overrides" (Section 3.4.2). Plain-text markers instead of real chat-template tokens is [ours]. |
  | T4 New session / faux reasoning | [paper] "New Session / Session Data Update" and "Faux Reasoning" (Section 3.4.2) |

- **Prompt layout B** (question first, then the document, then "Answer:") [ours]: being piloted because both base models almost never obeyed the templates in layout A. A possible reason is that the real question always came last.

### What we take from arXiv:2507.20526, and what it does not show

**What the paper says:**
- [paper] Indirect injections succeeded more often than direct ones: 27.1% vs 5.7% average ASR (Table 1, "All" row; 12.7% overall).
- [paper] System-prompt overrides were among the most prevalent strategies, with successful breaks across all models and behaviors. Faux reasoning and new-session attacks are the other named common strategies (Section 3.4.2).
- [paper] There is limited correlation between agent robustness and model size, capability, or inference-time compute (abstract; Section 3.3).

**Caveats on those numbers:**
- **The 27.1% vs 5.7% is not a controlled comparison.** These are averages over competition sessions, where human red-teamers iterated on multi-turn attacks against tool-using frontier agents. The behavior mix also differs between the two columns: "Conflicting Objectives" was direct-only.
- **The capability evidence is a proxy, not a sweep.** Capability is measured with GPQA, plus a few model pairs (e.g. GPT-4o vs GPT-4.5). The paper also reports "a non-negligible negative correlation" overall. Size enters only through such examples, not a controlled size sweep.

**What the paper does not show:**
- Anything about small open models like Qwen2.5 0.5B/1.5B. The tested models were frontier models; the smallest named is Llama 3.3 70B.
- Anything about plain document QA without tools. Its setting was agents with tools and policies.
- That our specific templates work on our models.
- Whether a fine-tuning defense such as GRPO reduces ASR. It only mentions existing defenses, such as instruction-hierarchy fine-tuning and filtering, which "adaptive attackers continue to outpace".
- Any ASR comparable to ours. Its ASR is per user session and judged by programs or LLMs; ours is per prompt with a nonce string match.

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
