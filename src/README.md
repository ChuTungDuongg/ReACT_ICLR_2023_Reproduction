# Source Code

This directory uses the standard Python `src` layout. The application package
is `react_reproduction`, and `main.py` adds this directory to the import path so
the project works immediately after a clone and dependency installation.

## Package map

| Package/module | Responsibility | Status |
|---|---|---|
| `config.py` | Typed YAML/environment configuration | Implemented |
| `logging_utils.py` | Terminal and optional file logging | Implemented |
| `cli.py` | Public command-line interface | Bootstrap implemented |
| `datasets/` | Shared examples and dataset loaders | HotpotQA implemented |
| `evaluation/` | Normalization, Exact Match, result schemas | Implemented |
| `experiments/` | Run orchestration and artifact persistence | Mock runner implemented |
| `llm/` | Model-provider abstraction | Planned for Sprint 2 |
| `agents/` | Standard, CoT, Act-only, and ReAct agents | Planned for Sprints 2-4 |
| `prompts/` | Task- and method-specific prompts | Planned for Sprint 2 onward |
| `tools/` | Wikipedia environment | Planned for Sprint 3 |

## Development rules

- Keep the core ReAct loop framework-free and inspectable.
- Reuse shared schemas and runners across tasks.
- Keep model downloads and network calls out of module import time.
- Preserve deterministic behavior when a seed is provided.
- Add implementation only in the sprint that owns it.
