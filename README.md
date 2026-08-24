# ReAct Paper Reproduction

> Language: English | [Tieng Viet](README.vi.md)

An inspectable implementation of the seven prompting methods evaluated in
"ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023):
Standard, CoT, CoT-SC, Act, ReAct, CoT-SC -> ReAct, and ReAct -> CoT-SC.
HotpotQA and FEVER share one CLI, model provider, batch runner, Wikipedia
environment, trajectory schema, logging path, and artifact convention.

The original paper used PaLM-540B. This project uses
`Qwen/Qwen2.5-7B-Instruct` by default and reproduces the method and experimental
protocol, not the paper's absolute model scores.

## Project Status

| Sprint | Scope | Status |
|---:|---|---|
| 0 | Repository, configuration, logging, CLI | Complete |
| 1 | HotpotQA dataset, evaluation, artifacts | Complete |
| 2 | Hugging Face, Standard, CoT | Complete |
| 3 | Wikipedia environment, Act | Complete |
| 4 | ReAct and trajectories | Complete |
| 5 | HotpotQA seven-method comparison | Functionally complete; superseded by the executed 500-example benchmark |
| 6 | Paper-faithful FEVER reproduction | Complete |
| 7 | Research report and failure analysis | Partially started |
| 8 | UI | Not started |
| 9 | Optional extensions/fine-tuning | Not started |

HotpotQA has completed 500-example runs for all seven methods. Historical runs
were made across different code versions and batch policies, so a controlled
rerun remains future cleanup. Existing analysis covers termination behavior,
hybrid paths, CoT-SC confidence, and method overlap.

FEVER is implemented and tested; its 500-example results are **TBD** until the
full Colab benchmark is run. ALFWorld, WebShop, and UI are not implemented.

The maintained roadmap is available as
[source](output/react-reproduction-roadmap.md) and
[PDF](output/pdf/react-reproduction-roadmap.pdf).

## Supported Surface

Tasks:

- `hotpotqa`
- `fever`

Methods:

- `standard`
- `cot`
- `cot-sc`
- `act`
- `react`
- `cot-sc-react`
- `react-cot-sc`

All commands use `python main.py benchmark`; there is no task-specific
executable.

## FEVER Protocol

FEVER is evaluated as claim verification with exactly three canonical labels:
`SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO`. The target model receives only the
claim. Gold evidence, contexts, and supporting pages are never included in the
prompt.

The prompts use the three manually composed examples from Appendix C.2:

1. Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. (`SUPPORTS`)
2. Stranger Things is set in Bloomington, Indiana. (`REFUTES`)
3. Beautiful reached number two on the Billboard Hot 100 in 2003. (`NOT ENOUGH INFO`)

Standard, CoT, Act, and ReAct are controlled ablations over those same examples.
Act and ReAct retain the paper's `Search[entity]`, `Lookup[string]`, and
`Finish[label]` action space. Search returns the first five sentences for a
resolved page or up to five Wikipedia suggestions when no page resolves;
repeated Lookup advances to the next occurrence.

FEVER uses Accuracy as its primary metric. ReAct has a default five-step budget.
CoT-SC uses 21 stochastic generations at temperature 0.7 and deterministic
first-observed tie-breaking, an implementation detail because the paper does not
specify ties. CoT-SC -> ReAct falls back when the winner has fewer than `n/2`
votes (10 or fewer for 21); ReAct -> CoT-SC falls back only when ReAct fails to
return a valid label within its budget. No fine-tuning is performed in Sprint 6.

## Paper vs Reproduction

| Paper | Reproduction |
|---|---|
| PaLM-540B | Qwen2.5-7B-Instruct |
| FEVER claim-only | Same |
| Three FEVER exemplars | Same Appendix C.2 examples |
| Search/Lookup/Finish | Same conceptual action space |
| CoT-SC `n=21` | Same |
| CoT-SC temperature `0.7` | Same |
| FEVER ReAct max steps `5` | Same |
| Primary metric Accuracy | Same |
| Fine-tuning | Not part of Sprint 6 |
| Batching | Engineering optimization; per-example state remains isolated |

Paper Table 1 FEVER Accuracy values are references only, not project results:

| Method | Paper Accuracy |
|---|---:|
| Standard | 57.1 |
| CoT | 56.3 |
| CoT-SC | 60.4 |
| Act | 58.9 |
| ReAct | 60.9 |
| CoT-SC -> ReAct | 64.6 |
| ReAct -> CoT-SC | 62.0 |

## Quick Start

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main.py --help
python main.py doctor
```

Model and dataset downloads are kept under the ignored
`cache/huggingface/` directory. Set `HF_TOKEN` in the environment for higher
Hugging Face Hub rate limits; do not store it in the repository.

## FEVER Colab Commands

Install once:

```python
!git clone https://github.com/ChuTungDuongg/ReACT_ICLR_2023_Reproduction.git
%cd ReACT_ICLR_2023_Reproduction
!python -m pip install -q -r requirements.txt
!python main.py doctor
```

Standard:

```bash
python main.py benchmark \
  --task fever \
  --method standard \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 32
```

CoT:

```bash
python main.py benchmark \
  --task fever \
  --method cot \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 16
```

CoT-SC:

```bash
python main.py benchmark \
  --task fever \
  --method cot-sc \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 16 \
  --cot-sc-samples 21 \
  --cot-sc-temperature 0.7
```

Act:

```bash
python main.py benchmark \
  --task fever \
  --method act \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 16 \
  --max-agent-steps 5
```

ReAct:

```bash
python main.py benchmark \
  --task fever \
  --method react \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 8 \
  --max-agent-steps 5
```

CoT-SC -> ReAct:

```bash
python main.py benchmark \
  --task fever \
  --method cot-sc-react \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 8 \
  --max-agent-steps 5 \
  --cot-sc-samples 21 \
  --cot-sc-temperature 0.7
```

ReAct -> CoT-SC:

```bash
python main.py benchmark \
  --task fever \
  --method react-cot-sc \
  --model Qwen/Qwen2.5-7B-Instruct \
  --num-samples 500 \
  --seed 42 \
  --batch-size 8 \
  --max-agent-steps 5 \
  --cot-sc-samples 21 \
  --cot-sc-temperature 0.7
```

Batch size is an engineering setting, not a paper hyperparameter. Start with
3-5 examples for a smoke run. `--show-trajectories` prints full reasoning and
tool traces; without it, CoT-SC logs only the winner, count, and vote map while
full samples remain in `trajectories.jsonl`.

## Architecture

```text
main.py / CLI
  -> task dataset adapter (HotpotQA or FEVER)
  -> shared Standard / CoT / CoT-SC / Act / ReAct / hybrid agents
  -> shared Hugging Face provider and batch scheduling
  -> per-example WikipediaEnvironment for interactive methods
  -> task evaluator (HotpotQA metrics or FEVER Accuracy)
  -> shared output serializer and live/file logging
```

Batching does not merge agent state. Each Act/ReAct example owns its current
page, Lookup offsets, history, counters, and termination state. Hybrid fallback
is dispatched only for examples that satisfy the paper heuristic.

## Run Artifacts

```text
outputs/<task>/<method>/<run_id>/
  config.json
  metrics.json
  predictions.jsonl
  trajectories.jsonl
  run.log
```

The default FEVER source is the official ReAct repository's
`data/paper_dev.jsonl`, pinned to a Git commit. It matches the official wrapper
and avoids both obsolete Hugging Face dataset scripts and evidence-expanded
duplicate rows in `labelled_dev`. The pinned source contains 9,999 claims.

`config.json` records dataset/split, model and resolved revision when exposed,
method, seed, sample IDs, batch and generation settings, task step budget,
CoT-SC settings and threshold, prompt version, code version, and timestamp.
FEVER predictions include claim, canonical prediction, correctness,
invalid/unparsed state, steps, tool calls, termination, latency, and relevant
vote/execution-path metadata. FEVER metrics include Accuracy, invalid count,
class distributions, per-class accuracy, confusion matrix, operational averages,
runtime, and termination reasons. Evidence scoring is intentionally outside
Sprint 6.

## Verification

```bash
python -m pytest -q
python -m compileall -q main.py src tests
python main.py --help
python main.py doctor
```

The offline suite uses scripted providers and fake Wikipedia clients; it does
not download Qwen. It covers FEVER labels, deterministic claim-only loading,
the exact three prompt examples and ablations, all seven paths, `11/21` and
`10/21` fallback boundaries, batch isolation, serialization, Accuracy, the
paper-fidelity audit, and HotpotQA regression.

## Known Limitations

- FEVER 500-example project results are TBD; no score is fabricated here.
- Qwen2.5-7B and current Wikipedia differ from PaLM-540B and the historical
  Wikipedia snapshot, so exact paper scores are not expected.
- MediaWiki availability and evolving article content affect Act/ReAct runs.
- HotpotQA historical runs used different code versions/batch policies.
- HotpotQA retains its legacy repeated-Search loop guard for historical behavior;
  FEVER disables that guard and lets ReAct consume the paper's five-step budget.
- HotpotQA agents do not emit `(title, sentence_id)` evidence pairs, so its
  supporting-fact and joint metrics remain limited.
- Fine-tuning, ALFWorld, WebShop, and UI are outside Sprint 6.

## Citation

```bibtex
@inproceedings{yao2023react,
  title     = {ReAct: Synergizing Reasoning and Acting in Language Models},
  author    = {Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and
               Izhak Shafran and Karthik Narasimhan and Yuan Cao},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```
