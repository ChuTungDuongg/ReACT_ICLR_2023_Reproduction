# Tests

> 🌐 **Language:** English | [Tiếng Việt](README.vi.md)

The 68-test suite is deterministic and does not download a model or call live
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
| `test_hotpotqa_prompts.py` | Six Appendix C.1 examples per method, question-only target input, and numbered interactive history. |
| `test_metrics.py` | Official answer, supporting-fact, and joint HotpotQA metrics. |
| `test_parsing.py` | Standard/CoT answer and reasoning parsing. |
| `test_standard_cot_agents.py` | Closed-book agent prompts/results. |
| `test_hybrid_agents.py` | CoT-SC normalized voting, paper threshold, and both fallback orders. |
| `test_action_parsing.py` | Search/Lookup/Finish and ReAct Thought/Action parsing, including format drift. |
| `test_wikipedia.py` | Search, current article, repeated Lookup, missing pages, loops, and max steps. |
| `test_act_agent.py` | Act-only loop and parse recovery. |
| `test_react_agent.py` | Thought/Action/Observation loop, history, recovery, and termination. |
| `test_experiment_serialization.py` | Config, full metrics, terminal logging, flushed predictions, and trajectories. |

Temporary test outputs are created only under `outputs/`, their resolved parent
is checked, and the exact temporary directory is removed afterward.
