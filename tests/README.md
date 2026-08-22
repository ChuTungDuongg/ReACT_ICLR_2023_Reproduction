# Tests

The 36-test suite is deterministic and does not download a model or call live
Wikipedia. Scripted LLMs, injected datasets, and a fake Wikipedia client cover
the same control flow used by real runs.

Run from the repository root:

```bash
python -m pytest -q
```

## Coverage by file

| File | What it verifies |
|---|---|
| `test_smoke.py` | CLI help, configuration, and `doctor`. |
| `test_hotpotqa.py` | Seeded sampling, metadata, loader arguments, validation. |
| `test_metrics.py` | HotpotQA normalization and Exact Match. |
| `test_parsing.py` | Standard/CoT answer and reasoning parsing. |
| `test_standard_cot_agents.py` | Closed-book agent prompts/results. |
| `test_action_parsing.py` | Search/Lookup/Finish and ReAct Thought/Action parsing, including format drift. |
| `test_wikipedia.py` | Search, current article, repeated Lookup, missing pages, loops, and max steps. |
| `test_act_agent.py` | Act-only loop and parse recovery. |
| `test_react_agent.py` | Thought/Action/Observation loop, history, recovery, and termination. |
| `test_experiment_serialization.py` | Config, metrics, flushed prediction records, and flushed trajectories. |

Temporary test outputs are created only under `outputs/`, their resolved parent
is checked, and the exact temporary directory is removed afterward.
