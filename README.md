# ReAct Paper Reproduction

A production-oriented research repository for reproducing and benchmarking the
prompting strategies studied in **“ReAct: Synergizing Reasoning and Acting in
Language Models” (ICLR 2023)**.

> This is a reproduction study of ReAct with modern instruction-tuned language
> models. It is not an exact reproduction using PaLM-540B from the original
> paper.
>
> Đây là reproduction study của ReAct sử dụng modern instruction-tuned LLM,
> không phải exact reproduction bằng PaLM-540B như paper gốc.

Current status: **Sprint 0 — Repository Bootstrap**. Dataset loading, model
inference, agents, and benchmark execution are intentionally not implemented
yet.

## Paper citation

```bibtex
@inproceedings{yao2023react,
  title     = {ReAct: Synergizing Reasoning and Acting in Language Models},
  author    = {Shunyu Yao and Jeffrey Zhao and Dian Yu and Nan Du and
               Izhak Shafran and Karthik Narasimhan and Yuan Cao},
  booktitle = {International Conference on Learning Representations},
  year      = {2023}
}
```

## Motivation

ReAct interleaves language-model reasoning with actions in an external
environment. This project centers experiments—not a chatbot—and aims to provide
a readable, reproducible benchmark harness suitable for research and an AI
engineering portfolio.

## Research questions

1. How do Standard, Chain-of-Thought (CoT), Act-only, and ReAct prompting compare
   on knowledge-intensive reasoning tasks?
2. When tools help, which failure modes remain: retrieval, reasoning, parsing,
   looping, or premature termination?
3. How do modern open instruction-tuned models differ from the PaLM-540B setup
   used in the original paper?

## Architecture

The planned experiment path is:

```text
CLI/config
    -> dataset loader
    -> prompting method + model provider
    -> optional Wikipedia environment
    -> evaluator
    -> versioned run artifacts and logs
```

The core agent loops will be implemented directly in Python without LangChain,
LangGraph, CrewAI, AutoGen, or a similar agent framework. Hugging Face
Transformers will be the first model backend.

## Standard vs. CoT vs. Act vs. ReAct

| Method | Explicit reasoning | Wikipedia actions | Planned trajectory |
|---|---:|---:|---|
| Standard | No | No | Question → Answer |
| CoT | Yes | No | Question → Reasoning → Answer |
| Act-only | No | Yes | Action ↔ Observation → Finish |
| ReAct | Yes | Yes | Thought → Action → Observation → Finish |

All four methods are planned; none is implemented in Sprint 0.

## Repository structure

```text
.
├── main.py
├── configs/
│   └── default.yaml
├── src/react_reproduction/
│   ├── agents/
│   ├── datasets/
│   ├── evaluation/
│   ├── experiments/
│   ├── llm/
│   ├── prompts/
│   ├── tools/
│   ├── cli.py
│   ├── config.py
│   └── logging_utils.py
├── tests/
│   └── test_smoke.py
└── outputs/
    └── .gitkeep
```

The empty package namespaces mark planned component boundaries. Their
implementations will be added only in the corresponding sprint.

## Installation

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

## Google Colab workflow

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

## CLI usage

Available in Sprint 0:

```bash
python main.py --help
python main.py doctor
python main.py doctor --config configs/default.yaml --log-level DEBUG
```

`doctor` loads the configuration and checks the local bootstrap. The
`benchmark` parser is present to make the intended interface discoverable, but
it deliberately exits with a “not implemented” message until later sprints.

Planned interface:

```bash
python main.py benchmark \
    --task hotpotqa \
    --method react \
    --model Qwen/Qwen2.5-3B-Instruct \
    --num-samples 100 \
    --seed 42
```

## HotpotQA benchmark

Planned for Sprints 1–5. Evaluation will include normalized Exact Match,
predictions, trajectory metadata, tool calls, agent steps, latency, and
termination reason.

Status: **TBD**.

## FEVER benchmark

Planned for Sprint 6 with `SUPPORTS`, `REFUTES`, and `NOT ENOUGH INFO` labels.
The implementation will reuse the HotpotQA experiment infrastructure.

Status: **TBD**.

## Experimental results

No benchmark has been run, and no result is fabricated.

| Task | Method | Samples | Primary metric | Result |
|---|---|---:|---|---|
| HotpotQA | Standard / CoT / Act / ReAct | TBD | Exact Match | TBD |
| FEVER | Standard / CoT / Act / ReAct | TBD | Accuracy | TBD |

## Failure analysis

Later runs will retain sufficient metadata to inspect hallucination,
retrieval failure, reasoning failure, bad tool selection, action loops,
premature finishes, parsing errors, and max-step termination. Automated and
human-readable analysis is planned for Sprint 7.

## Differences from the original paper

- Modern open, instruction-tuned Hugging Face models will replace PaLM-540B.
- Publicly available datasets and Wikipedia access will be used.
- Hardware, model scale, prompts, and serving stack therefore differ from the
  original experimental setup.
- The implementation emphasizes inspectable trajectories and reproducible
  run artifacts.

## Limitations

- Sprint 0 contains no dataset, model, agent, or evaluation implementation.
- Exact numerical replication may be impossible without the original model and
  serving environment.
- Wikipedia content and search behavior can change over time.
- Local and Colab hardware can materially affect latency and feasible models.

## Future work

The ordered roadmap covers HotpotQA foundations, Hugging Face inference,
Wikipedia actions, ReAct, method comparison, FEVER, failure analysis, and an
optional interactive demo. ALFWorld, WebShop, fine-tuning, alternative model
providers, latency/cost analysis, and RAG comparisons remain optional later
extensions.

## References

- Yao et al., “ReAct: Synergizing Reasoning and Acting in Language Models,”
  ICLR 2023.
- HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering.
- FEVER: a Large-scale Dataset for Fact Extraction and VERification.
