# Source code

> 🌐 **Language:** English | [Tiếng Việt](README.vi.md)

This directory uses the standard Python `src` layout. `main.py` adds `src/` to
the import path, so a fresh clone works without installing the project as a
package.

## Package map

| Path | Responsibility |
|---|---|
| `react_reproduction/cli.py` | Parses `doctor`/`benchmark`, resolves overrides, and wires model, agent, environment, and runner. |
| `react_reproduction/config.py` | Loads and validates YAML defaults plus supported environment overrides. |
| `react_reproduction/logging_utils.py` | Streams UTF-8 logs to stdout and duplicates benchmark logs to `run.log`. |
| `react_reproduction/datasets/` | Defines `BenchmarkExample` and deterministic HotpotQA loading/sampling. |
| `react_reproduction/llm/` | Defines the provider interface and lazy Hugging Face implementation with automatic device/dtype selection. |
| `react_reproduction/prompts/` | Builds Standard, CoT, Act-only, and ReAct prompts from the six manually composed HotpotQA examples in paper Appendix C.1. |
| `react_reproduction/agents/` | Contains contracts, parsers, four base agents, CoT-SC voting, and both hybrids. |
| `react_reproduction/tools/` | Calls MediaWiki and owns Search/Lookup/Finish state, ambiguity, loop, and max-step behavior. |
| `react_reproduction/evaluation/` | Computes official answer/supporting-fact/joint metrics and defines JSON-ready records. |
| `react_reproduction/experiments/` | Runs examples and incrementally persists predictions, trajectories, metrics, config, and logs. |

## Data flow

```text
CLI → HotpotQA → Hugging Face model → agent
                              Act/ReAct ↕ Wikipedia
    → evaluator → outputs/<task>/<method>/<timestamp>/
```

The implementation is complete through Sprint 4 plus both paper hybrid
fallback policies. Qwen2.5-7B-Instruct is the CLI default model. Keep
model/network work lazy, keep the core loops framework-free, and preserve
deterministic seeded behavior.

`agents/cot_sc.py` implements normalized majority voting over sampled CoT
answers. `agents/hybrid.py` implements ReAct → CoT-SC and CoT-SC → ReAct with
the paper fallback rules. Hybrid trajectories carry explicit `react`/`cot_sc`
phase labels and prediction metadata records vote confidence and selected path.
The batch APIs keep sequential fallbacks for test providers, while the Hugging
Face provider performs real padded GPU generation. CoT-SC batches every
sampling round; ReAct keeps one reusable Wikipedia environment per batch slot.

For Act/ReAct, `Search` accepts a concise entity or page title while `Lookup`
reads matching sentences inside the open article. A second identical `Search`
is skipped with a recovery hint; a third is classified as `action_loop`.
Repeated `Lookup` remains valid because it advances through matches. ReAct uses
paper-style termination by default in standalone and hybrid modes: no forced
answer is added after a loop or at the step limit. The CLI can explicitly opt
all ReAct legs into best-effort finalization; Act-only retains that behavior by
default.
