# 🧠 ReAct Paper Reproduction

<p align="center">
  <img src="https://img.shields.io/badge/Paper-ICLR%202023-6f42c1" alt="ICLR 2023 paper">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/Status-Sprint%204%20Complete-238636" alt="Sprint 4 complete">
  <img src="https://img.shields.io/badge/Tests-36%20Passing-18A0AE" alt="36 tests passing">
  <img src="https://img.shields.io/badge/Interface-main.py-102A43" alt="CLI first">
</p>

<p align="center">
  <a href="output/pdf/react-reproduction-roadmap.pdf">📄 Roadmap</a> •
  <a href="#google-colab">☁️ Colab</a> •
  <a href="#cli">⌨️ CLI</a> •
  <a href="#repository-map">🗂️ File map</a> •
  <a href="#testing">🧪 Tests</a>
</p>

A framework-free, inspectable implementation of the Standard, Chain-of-Thought,
Act-only, and ReAct methods for HotpotQA. It uses Hugging Face models and a
live Wikipedia environment through one entry point: `main.py`.

> Đây là reproduction study dùng modern instruction-tuned LLM, không phải exact
> reproduction bằng PaLM-540B như paper gốc.

## ✅ Current status

**Sprints 0, 1, 2, 3, and 4 are complete. Development stops at Sprint 4 in the
current milestone.** Sprint 5, FEVER, ALFWorld, WebShop, and an interactive app
are not implemented.

| Sprint | Scope | Status |
|---:|---|---|
| 0 | Repository, configuration, logging, CLI skeleton | ✅ Complete |
| 1 | HotpotQA loader, metrics, schemas, artifact runner | ✅ Complete |
| 2 | Hugging Face provider, Standard, CoT | ✅ Complete |
| 3 | Wikipedia Search/Lookup/Finish, Act-only, loop/max-step guards | ✅ Complete |
| 4 | ReAct loop, live trajectories, trajectory persistence | ✅ Complete |
| 5+ | Comparison, FEVER, analysis, optional extensions | ⏸️ Not started |

The full requirements and sprint-by-sprint acceptance criteria are in the
[roadmap PDF](output/pdf/react-reproduction-roadmap.pdf).

## 🚀 Quick start

```bash
git clone https://github.com/ChuTungDuongg/ReACT_ICLR_2023_Reproduction.git
cd ReACT_ICLR_2023_Reproduction
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell instead:
# .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python main.py doctor
```

No notebook is required or included. The model and dataset are downloaded to
the ignored project-local `cache/huggingface/` directory on first use.

<a id="google-colab"></a>

## ☁️ Google Colab

Run these commands in Colab cells. CUDA is selected automatically when the
installed PyTorch build and GPU support it; otherwise inference falls back to
CPU. `bitsandbytes` is not required.

```python
!git clone https://github.com/ChuTungDuongg/ReACT_ICLR_2023_Reproduction.git
%cd ReACT_ICLR_2023_Reproduction
!python -m pip install -q -r requirements.txt
!python main.py doctor
```

Then run the requested five-sample benchmark:

```python
!python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --show-trajectories
```

Optionally set `HF_TOKEN` in the Colab environment before the run to receive
higher Hugging Face Hub rate limits. The command streams progress and every
Thought/Action/Observation to stdout while also writing `run.log`.

<a id="cli"></a>

## ⌨️ CLI

```bash
python main.py --help
python main.py doctor

python main.py benchmark \
    --task hotpotqa \
    --method standard \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 5 \
    --seed 42

python main.py benchmark \
    --task hotpotqa \
    --method cot \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 5 \
    --seed 42

python main.py benchmark \
    --task hotpotqa \
    --method act \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --show-trajectories

python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 5 \
    --seed 42 \
    --show-trajectories
```

Useful overrides include `--device auto|cpu|cuda|mps`, `--max-agent-steps`,
`--max-new-tokens`, `--temperature`, `--top-p`, `--log-level`, and
`--trust-remote-code`. The default deterministic generation uses temperature
`0.0`, top-p `1.0`, and seed `42`.

## 🧩 How the four methods work

| Method | Explicit reasoning | Wikipedia | Execution path |
|---|---:|---:|---|
| Standard | No | No | Question → concise answer |
| CoT | Yes | No | Question → reasoning → answer |
| Act-only | No | Yes | Action ↔ Observation → Finish |
| ReAct | Yes | Yes | Thought → Action → Observation → Finish |

The agents are direct Python loops, not wrappers around LangChain, LangGraph,
CrewAI, or AutoGen. This keeps parsing, environment state, termination reasons,
and trajectories visible for research inspection.

## 🏗️ Runtime flow

```text
main.py
  → CLI + validated YAML configuration
  → deterministic HotpotQA sample
  → HuggingFaceProvider (CUDA/CPU/MPS auto-detection)
  → selected agent
      └─ Act/ReAct → WikipediaEnvironment → MediaWiki API
  → Exact Match evaluator
  → flushed JSONL + JSON + live/file logs
```

<a id="repository-map"></a>

## 🗂️ Repository map

```text
.
├── main.py                         # Single CLI entry point
├── requirements.txt                # Runtime and test dependencies
├── configs/
│   └── default.yaml                # Versioned experiment defaults
├── output/
│   └── pdf/                        # Stable human-facing roadmap PDF
├── outputs/                        # Ignored runtime benchmark artifacts
├── src/react_reproduction/
│   ├── agents/                     # Standard, CoT, Act-only, ReAct loops/parsers
│   ├── datasets/                   # Shared example + HotpotQA loader
│   ├── evaluation/                 # Exact Match + serializable schemas
│   ├── experiments/                # Run orchestration and artifact writers
│   ├── llm/                        # Provider contract + Hugging Face backend
│   ├── prompts/                    # HotpotQA prompt builders
│   ├── tools/                      # Wikipedia client and stateful environment
│   ├── cli.py                      # Argument parsing and dependency wiring
│   ├── config.py                   # Typed YAML/environment configuration
│   └── logging_utils.py            # UTF-8 live stdout + run.log
└── tests/                           # 36 deterministic tests
```

Each first-level directory has its own README:

- [Source guide](src/README.md)
- [Configuration guide](configs/README.md)
- [Test guide](tests/README.md)
- [Runtime output guide](outputs/README.md)
- [Project document guide](output/README.md)

## 📦 Run artifacts

Every benchmark gets an isolated UTC-stamped directory:

```text
outputs/hotpotqa/react/<UTC timestamp>/
├── config.json          # Complete reproducibility settings
├── metrics.json         # Aggregate metrics and termination counts
├── predictions.jsonl    # One record per example, flushed immediately
├── trajectories.jsonl   # Act/ReAct steps, one flushed record per example
└── run.log              # Same inspectable progress retained from stdout
```

`predictions.jsonl` is flushed after every example for every method.
`trajectories.jsonl` is created for interactive Act/ReAct runs and contains the
model output, Thought, canonical Action, and actual environment Observation for
each step.

## 📊 Verified development smoke result

This is a small pipeline check, **not a paper-quality benchmark result** and not
evidence about larger Qwen models. No result is fabricated.

| Date | Task | Method | Model | Samples | Max steps | Exact Match | Terminations |
|---|---|---|---|---:|---:|---:|---|
| 2026-08-23 | HotpotQA | ReAct | Qwen2.5-0.5B-Instruct | 3 | 3 | 0.0 (0/3) | loop 1, max-step 1, parse 1 |

The run used real Hugging Face inference and live MediaWiki responses on local
CPU. All three examples produced persisted trajectories; the small model did
not reach a correct `Finish` answer in this smoke run.

<a id="testing"></a>

## 🧪 Testing and reproducibility

```bash
python -m pytest -q
python -m compileall -q main.py src tests
python main.py --help
python main.py doctor
```

The 36-test suite is offline/model-free and covers configuration/CLI smoke,
seeded HotpotQA loading, normalization/Exact Match, Hugging Face agent wiring,
Standard/CoT parsing, Wikipedia Search/Lookup/Finish, ambiguity, repeated
Lookup, loop detection, max steps, Act-only/ReAct recovery, incremental
prediction persistence, and trajectory persistence.

## ⚙️ Dependencies

- `torch`, `transformers`, and `accelerate`: local Hugging Face inference and
  automatic device placement;
- `datasets`: HotpotQA access and caching;
- `requests` and `tenacity`: MediaWiki calls with timeout/retry handling;
- `PyYAML` and `python-dotenv`: configuration;
- `pytest`: repository verification.

Model loading is lazy. CUDA uses BF16 when supported, otherwise FP16; CPU uses
FP32. `bitsandbytes` is intentionally not a required dependency.

## ⚠️ Limitations

- This milestone stops at Sprint 4; no cross-method benchmark table or
  statistical study has been run.
- The original PaLM-540B model, prompts, serving stack, and historical
  Wikipedia state are unavailable, so exact numerical replication is not
  expected.
- Wikipedia search/content can change, and network availability affects
  Act/ReAct runs.
- A 3B model may exceed some local machines; Colab GPU memory and runtime tier
  determine feasible speed. Use a smaller model only for pipeline debugging.
- Generated text can violate the action grammar. The parser takes the first
  valid action, records parsing failures, and the environment guards loops and
  maximum steps.

## 📄 Citation

```bibtex
@inproceedings{yao2023react,
  title     = {ReAct: Synergizing Reasoning and Acting in Language Models},
  author    = {Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and
               Izhak Shafran and Karthik Narasimhan and Yuan Cao},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```
