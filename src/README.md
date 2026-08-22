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
| `react_reproduction/prompts/` | Builds HotpotQA prompts for Standard, CoT, Act-only, and ReAct. |
| `react_reproduction/agents/` | Contains result/trajectory contracts, output parsers, and all four agent implementations. |
| `react_reproduction/tools/` | Calls MediaWiki and owns Search/Lookup/Finish state, ambiguity, loop, and max-step behavior. |
| `react_reproduction/evaluation/` | Normalizes answers, computes Exact Match, and defines JSON-ready records. |
| `react_reproduction/experiments/` | Runs examples and incrementally persists predictions, trajectories, metrics, config, and logs. |

## Data flow

```text
CLI → HotpotQA → Hugging Face model → agent
                              Act/ReAct ↕ Wikipedia
    → evaluator → outputs/<task>/<method>/<timestamp>/
```

The implementation is complete through Sprint 4. Keep model/network work lazy,
keep the core loops framework-free, and preserve deterministic seeded behavior.
