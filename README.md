# 🧠 ReAct Paper Reproduction

<p align="center">
  <img src="https://img.shields.io/badge/Paper-ICLR%202023-6f42c1" alt="ICLR 2023 paper">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Status-Sprint%201%20Complete-238636" alt="Sprint 1 complete">
  <img src="https://img.shields.io/badge/Tests-9%20Passing-18A0AE" alt="9 tests passing">
  <img src="https://img.shields.io/badge/Workflow-CLI%20First-102A43" alt="CLI-first workflow">
</p>

<p align="center">
  <a href="output/pdf/react-reproduction-roadmap.pdf">📄 Roadmap</a> •
  <a href="#installation">⚙️ Installation</a> •
  <a href="#cli-usage">⌨️ CLI</a> •
  <a href="#testing">🧪 Testing</a>
</p>

A production-oriented research repository for reproducing and benchmarking the
prompting strategies studied in **“ReAct: Synergizing Reasoning and Acting in
Language Models” (ICLR 2023)**.

> This is a reproduction study of ReAct with modern instruction-tuned language
> models. It is not an exact reproduction using PaLM-540B from the original
> paper.
>
> Đây là reproduction study của ReAct sử dụng modern instruction-tuned LLM,
> không phải exact reproduction bằng PaLM-540B như paper gốc.

✅ Current status: **Sprint 1 — Dataset + Evaluation Foundation (complete)**.
HotpotQA loading, deterministic sampling, normalized Exact Match, prediction
schemas, run serialization, and a mock benchmark runner are implemented. Model
inference and prompting agents intentionally remain pending until Sprint 2.

## 📚 Project documentation

- [Project roadmap PDF](output/pdf/react-reproduction-roadmap.pdf): the complete
  sprint-by-sprint plan, requirements, architecture, acceptance criteria,
  current status, dependencies, testing strategy, and limitations.
- [Configuration guide](configs/README.md): configuration ownership, precedence,
  and repository rules.
- [Source guide](src/README.md): package map, implementation status, and source
  development rules.
- [Test guide](tests/README.md): current coverage and model-free testing rules.
- [Experiment output guide](outputs/README.md): ignored runtime artifact layout.
- [Project document guide](output/README.md): stable generated documents intended
  for reading and sharing.

`outputs/` contains ignored benchmark runs. `output/` contains stable,
human-facing project documents such as the roadmap PDF.

## 📄 Paper citation

```bibtex
@inproceedings{yao2023react,
  title     = {ReAct: Synergizing Reasoning and Acting in Language Models},
  author    = {Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and
               Izhak Shafran and Karthik Narasimhan and Yuan Cao},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

## 🎯 Motivation

ReAct interleaves language-model reasoning with actions in an external
environment. This project centers experiments—not a chatbot—and aims to provide
a readable, reproducible benchmark harness suitable for research and an AI
engineering portfolio.

## 🔬 Research questions

1. How do Standard, Chain-of-Thought (CoT), Act-only, and ReAct prompting compare
   on knowledge-intensive reasoning tasks?
2. When tools help, which failure modes remain: retrieval, reasoning, parsing,
   looping, or premature termination?
3. How do modern open instruction-tuned models differ from the PaLM-540B setup
   used in the original paper?

## 🏗️ Architecture

The experiment path is being implemented sprint by sprint:

```text
CLI/config
    -> dataset loader
    -> prompting method + model provider
    -> optional Wikipedia environment
    -> evaluator
    -> versioned run artifacts and logs
```

Sprint 1 implements the dataset, evaluation, and artifact layers. The core
agent loops will be implemented directly in Python without LangChain,
LangGraph, CrewAI, AutoGen, or a similar agent framework. Hugging Face
Transformers will be the first model backend.

## 🧩 Standard vs. CoT vs. Act vs. ReAct

| Method | Explicit reasoning | Wikipedia actions | Planned trajectory |
|---|---:|---:|---|
| Standard | No | No | Question → Answer |
| CoT | Yes | No | Question → Reasoning → Answer |
| Act-only | No | Yes | Action ↔ Observation → Finish |
| ReAct | Yes | Yes | Thought → Action → Observation → Finish |

All four methods are planned; none is implemented in Sprint 1. The current
`MockPredictor` exists only to test the model-independent runner and must not be
interpreted as an experimental method or result.

## 🗂️ Repository structure

```text
.
├── main.py
├── configs/
│   ├── README.md
│   └── default.yaml
├── output/
│   ├── README.md
│   └── pdf/
│       └── react-reproduction-roadmap.pdf
├── outputs/
│   ├── README.md
│   └── .gitkeep
├── src/
│   ├── README.md
│   └── react_reproduction/
│       ├── agents/
│       ├── datasets/
│       │   ├── base.py
│       │   └── hotpotqa.py
│       ├── evaluation/
│       │   ├── metrics.py
│       │   └── schemas.py
│       ├── experiments/
│       │   ├── artifacts.py
│       │   └── runner.py
│       ├── llm/
│       ├── prompts/
│       ├── tools/
│       ├── cli.py
│       ├── config.py
│       └── logging_utils.py
├── tests/
│   ├── README.md
│   ├── conftest.py
│   ├── test_experiment_serialization.py
│   ├── test_hotpotqa.py
│   ├── test_metrics.py
│   └── test_smoke.py
└── requirements.txt
```

Key Sprint 1 modules:

- `datasets/base.py` defines the dataset-neutral `BenchmarkExample` consumed by
  experiment runners.
- `datasets/hotpotqa.py` loads `hotpotqa/hotpot_qa`, validates records, and
  samples examples deterministically with a local random generator.
- `evaluation/metrics.py` implements the standard lowercase, punctuation,
  article, and whitespace normalization followed by Exact Match.
- `evaluation/schemas.py` defines serializable prediction and metric records.
- `experiments/artifacts.py` creates collision-safe run directories and writes
  UTF-8 JSON/JSONL artifacts.
- `experiments/runner.py` performs model-independent HotpotQA evaluation. Its
  `MockPredictor` enables offline tests without downloading an LLM.

`agents`, `llm`, `prompts`, and `tools` remain planned namespaces and are not
silently implemented ahead of their scheduled sprints.

<a id="installation"></a>

## ⚙️ Installation

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux or macOS, activate the environment with
`source .venv/bin/activate`.

## ☁️ Google Colab workflow

```python
!git clone <repo_url> react-paper-reproduction
%cd react-paper-reproduction
!pip install -r requirements.txt
!python main.py --help
```

Once the benchmark sprints are complete, runs will not require editing Python
source code:

```python
!python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model <huggingface_model> \
    --num-samples 100 \
    --show-trajectories
```

<a id="cli-usage"></a>

## ⌨️ CLI usage

Available in Sprint 1:

```bash
python main.py --help
python main.py doctor
python main.py doctor --config configs/default.yaml --log-level DEBUG
```

`doctor` loads the configuration and reports the configured HotpotQA source.
The public `benchmark` parser is present to make the intended interface
discoverable, but it deliberately exits with a clear message until Hugging Face
inference and Standard/CoT prompting are implemented in Sprint 2. Sprint 1
tests the runner through dependency injection instead of exposing fake CLI
benchmark results.

Planned interface:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 100 \
    --seed 42
```

## 🔥 HotpotQA benchmark

Sprint 1 uses the official Hugging Face dataset
[`hotpotqa/hotpot_qa`](https://huggingface.co/datasets/hotpotqa/hotpot_qa), with
the `distractor` subset and `validation` split by default. The settings live in
`configs/default.yaml` and are exposed through typed project configuration.

Implemented:

- positive subset sizes including 10, 50, 100, and 500 samples;
- deterministic sampling controlled by `seed`;
- validation of required `id`, `question`, and `answer` fields;
- preservation of question type, difficulty, and supporting-fact metadata;
- normalized answer matching and Exact Match;
- total/correct/incorrect counts, average steps/tool calls, runtime, and
  termination-reason counts;
- incremental prediction serialization and final metrics serialization.

The first real dataset load populates the Hugging Face cache and may download
and process the dataset files. Dataset tests use injected in-memory records, so
the normal test suite remains offline and fast.

Not implemented yet:

- Hugging Face model inference;
- Standard and CoT prompts;
- Wikipedia actions, Act-only, and ReAct;
- full trajectory serialization and comparison reports.

Status: **foundation complete; model benchmark results TBD**.

## 📦 Experiment artifacts

The Sprint 1 runner creates one isolated directory per run:

```text
outputs/
└── hotpotqa/
    └── standard/
        └── 2026-08-23_013000/
            ├── config.json
            ├── metrics.json
            └── predictions.jsonl
```

`config.json` records model, dataset, task, method, sample count, seed,
generation settings, maximum agent steps, and timestamp. `predictions.jsonl`
contains one flushed record per example. `metrics.json` is written when the run
finishes. `trajectories.jsonl` and `run.log` will be integrated when the
corresponding agent and live-run functionality is implemented.

<a id="testing"></a>

## 🧪 Reproducibility and testing

Sampling uses `random.Random(seed)` rather than global random state. Run paths
are collision-safe, JSON writes use UTF-8, and every prediction contains its
latency and termination reason.

```bash
pytest
```

The Sprint 1 suite has nine model-free tests covering CLI smoke checks,
normalization, Exact Match, seeded sampling, Hugging Face loader arguments,
sample-size validation, mock execution, and artifact serialization.

## 🛡️ FEVER benchmark

Planned for Sprint 6 with `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` labels.
The implementation will reuse the HotpotQA experiment infrastructure.

Status: **TBD**.

## 📊 Experimental results

No model benchmark has been run, and no result is fabricated. A three-example
dataset-load smoke check and mock runner tests validate infrastructure only;
they are not research results.

| Task | Method | Samples | Primary metric | Result |
|---|---|---:|---|---|
| HotpotQA | Standard / CoT / Act / ReAct | TBD | Exact Match | TBD |
| FEVER | Standard / CoT / Act / ReAct | TBD | Accuracy | TBD |

## 🔎 Failure analysis

Later runs will retain sufficient metadata to inspect hallucination,
retrieval failure, reasoning failure, bad tool selection, action loops,
premature finishes, parsing errors, and max-step termination. Automated and
human-readable analysis is planned for Sprint 7.

## 🔄 Differences from the original paper

- Modern open, instruction-tuned Hugging Face models will replace PaLM-540B.
- Publicly available datasets and Wikipedia access will be used.
- Hardware, model scale, prompts, and serving stack therefore differ from the
  original experimental setup.
- The implementation emphasizes inspectable trajectories and reproducible
  run artifacts.

## ⚠️ Limitations

- Sprint 1 contains no model provider or prompting-agent implementation.
- The mock predictor validates infrastructure only and has no benchmark value.
- Exact numerical replication may be impossible without the original model and
  serving environment.
- Wikipedia content and search behavior can change over time.
- Local and Colab hardware can materially affect latency and feasible models.

## 🚀 Future work

The ordered roadmap covers HotpotQA foundations, Hugging Face inference,
Wikipedia actions, ReAct, method comparison, FEVER, failure analysis, and an
optional interactive demo. ALFWorld, WebShop, fine-tuning, alternative model
providers, latency/cost analysis, and RAG comparisons remain optional later
extensions.

## 🔗 References

- Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models,”
  ICLR 2023.
- HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
- FEVER: a Large-scale Dataset for Fact Extraction and VERification.
