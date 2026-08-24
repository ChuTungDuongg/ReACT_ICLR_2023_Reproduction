# Source code

> Language: English | [Tieng Viet](README.vi.md)

`react_reproduction` implements one shared HotpotQA/FEVER pipeline:

| Package | Responsibility |
|---|---|
| `datasets/` | Shared `BenchmarkExample`, deterministic HotpotQA and claim-only FEVER adapters |
| `prompts/` | Appendix C.1 HotpotQA prompts and versioned Appendix C.2 FEVER ablations |
| `agents/` | Shared Standard, CoT, CoT-SC, Act, ReAct, and hybrid control flow |
| `llm/` | Lazy Hugging Face loading and padded batch generation |
| `tools/` | Paper-compatible Wikipedia Search/Lookup/Finish state |
| `evaluation/` | HotpotQA metrics and FEVER Accuracy adapters/schemas |
| `experiments/` | Shared batching, logging, trajectory, and artifact persistence |
| `cli.py` | Task wiring and all seven public methods |

Agents receive prompt builders and answer normalizers from the selected task,
so FEVER does not duplicate agent loops. Every interactive batch slot owns an
independent Wikipedia environment. Hybrid agents batch only the examples that
actually require fallback.

FEVER defaults are fixed to Appendix C.2 prompt version
`fever-appendix-c2-v1`, ReAct max steps 5, CoT-SC 21 samples at temperature
0.7, Accuracy evaluation, and no early repeated-Search termination. HotpotQA
retains max steps 7, its legacy loop guard, and its existing
answer/supporting-fact/joint evaluator.
