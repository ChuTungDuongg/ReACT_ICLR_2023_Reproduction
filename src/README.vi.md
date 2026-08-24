# Source code

> Ngôn ngữ: [English](README.md) | Tiếng Việt

Package `react_reproduction` dùng một pipeline chung cho HotpotQA và FEVER:

| Package | Trách nhiệm |
|---|---|
| `datasets/` | `BenchmarkExample`, loader deterministic HotpotQA và FEVER claim-only |
| `prompts/` | Prompt Appendix C.1 và các ablation FEVER Appendix C.2 có version |
| `agents/` | Control flow dùng chung cho cả bảy methods |
| `llm/` | Hugging Face lazy loading và padded batching |
| `tools/` | Wikipedia Search/Lookup/Finish theo semantics paper |
| `evaluation/` | HotpotQA metrics và FEVER Accuracy |
| `experiments/` | Batching, logging, trajectory và artifacts dùng chung |
| `cli.py` | Task wiring và bảy method public |

Agent nhận prompt builder và answer normalizer từ task adapter; FEVER không
duplicate agent loop. Mỗi sample interactive có Wikipedia environment riêng và
hybrid chỉ batch những sample thật sự cần fallback.

FEVER mặc định dùng prompt `fever-appendix-c2-v1`, ReAct 5 bước, CoT-SC 21
sample ở temperature 0.7, Accuracy và không terminate sớm do Search lặp.
HotpotQA giữ budget 7 bước, legacy loop guard và evaluator
answer/supporting-fact/joint hiện có.
