# ReAct Paper Reproduction Roadmap

Version: Sprint 6 completion checkpoint

## Project Objective

Reproduce the prompting protocols from ReAct (ICLR 2023) with inspectable
Python control flow, reproducible artifacts, and modern Hugging Face inference.
Method fidelity takes priority over batching or model-specific optimization.

## Current Scope

| Area | Status | Notes |
|---|---|---|
| HotpotQA | Implemented and benchmarked | Seven methods, 500 examples each |
| FEVER | Implemented | Seven methods; full 500-example results TBD |
| ALFWorld | Not started | Outside Sprint 6 |
| WebShop | Not started | Outside Sprint 6 |
| UI | Not started | Outside Sprint 6 |

## Sprint Status

| Sprint | Scope | Status |
|---:|---|---|
| 0 | Repository, typed config, logging, CLI | COMPLETE |
| 1 | HotpotQA data, metrics, schemas, artifacts | COMPLETE |
| 2 | Hugging Face, Standard, CoT | COMPLETE |
| 3 | Wikipedia Search/Lookup/Finish, Act | COMPLETE |
| 4 | ReAct loop and trajectories | COMPLETE |
| 5 | HotpotQA seven-method comparison | FUNCTIONALLY COMPLETED / SUPERSEDED BY EXECUTED BENCHMARK |
| 6 | Paper-faithful FEVER reproduction | COMPLETE |
| 7 | Research report and failure analysis | NOT STARTED |
| 8 | UI | NOT STARTED |
| 9 | Optional fine-tuning/extensions | NOT STARTED |

Sprint 5 caveat: historical HotpotQA runs used different code versions and
batch policies. A controlled rerun is future cleanup. Existing termination,
hybrid-path, CoT-SC confidence, and overlap analyses are historical inputs;
Sprint 7 itself has not started.

## Shared Architecture

```text
Benchmark CLI
  -> HotpotQA or FEVER dataset adapter
  -> shared Standard / CoT / CoT-SC / Act / ReAct / hybrid agents
  -> shared Hugging Face provider and batch scheduler
  -> independent WikipediaEnvironment per interactive sample
  -> task evaluator
  -> shared JSON/JSONL/log artifacts
```

FEVER-specific code is limited to its dataset adapter, Appendix C.2 prompt
definitions, label parser/normalizer, Accuracy evaluator, and five-step default.
No FEVER-specific executable, LLM provider, agent loop, batch runner, Wikipedia
tool, or serializer exists.

## FEVER Paper-Faithful Setup

- Claim-only input; no FEVER gold evidence or supporting page enters prompts.
- Dataset is the pinned official ReAct `data/paper_dev.jsonl` file.
- Canonical outputs: SUPPORTS, REFUTES, NOT ENOUGH INFO.
- Exact three manually composed task examples from Appendix C.2.
- Standard, CoT, Act, and ReAct use controlled ablations of those examples.
- Wikipedia action space remains Search, Lookup, and Finish.
- Search returns the first five article sentences or top-five suggestions.
- Repeated Lookup advances through sentence occurrences on the current page.
- FEVER does not terminate early on repeated Search; the five-step budget rules.
- FEVER ReAct default budget is five steps; HotpotQA remains seven.
- CoT-SC samples 21 trajectories at temperature 0.7.
- CoT-SC -> ReAct fallback: winning count below n/2.
- ReAct -> CoT-SC fallback: ReAct fails to answer within the step budget.
- Primary FEVER metric is claim-label Accuracy.
- No SFT, LoRA, QLoRA, RL, RAG, embeddings, or alternate retriever.

## Appendix C.2 Exemplars

| Claim | Gold label |
|---|---|
| Nikolaj Coster-Waldau worked with the Fox Broadcasting Company. | SUPPORTS |
| Stranger Things is set in Bloomington, Indiana. | REFUTES |
| Beautiful reached number two on the Billboard Hot 100 in 2003. | NOT ENOUGH INFO |

Prompt version is frozen as `fever-appendix-c2-v1`. The deterministic tie-break
for equal CoT-SC vote counts is the first observed valid normalized label. This
is a reproduction implementation detail; the paper does not specify tie handling.
For historical compatibility, HotpotQA retains its earlier repeated-Search loop
guard; FEVER uses the paper-compatible mode described above.

<!-- PAGEBREAK -->

## Paper vs Reproduction

| Paper | Reproduction |
|---|---|
| PaLM-540B | Qwen/Qwen2.5-7B-Instruct |
| FEVER claim-only | Same |
| Three FEVER exemplars | Same |
| Search/Lookup/Finish | Same conceptual action space |
| CoT-SC n=21 | Same |
| CoT-SC temperature=0.7 | Same |
| FEVER ReAct max steps=5 | Same |
| Primary metric Accuracy | Same |
| Fine-tuning | Not part of Sprint 6 |
| Batching | Engineering optimization with isolated state |

## Paper Table 1 Reference

These are PaLM-540B paper results, not project results and not test thresholds.

| Method | FEVER Accuracy |
|---|---:|
| Standard | 57.1 |
| CoT | 56.3 |
| CoT-SC | 60.4 |
| Act | 58.9 |
| ReAct | 60.9 |
| CoT-SC -> ReAct | 64.6 |
| ReAct -> CoT-SC | 62.0 |

## FEVER Outputs and Metrics

Each run writes `config.json`, `metrics.json`, `predictions.jsonl`,
`trajectories.jsonl` for trajectory methods, and `run.log` under
`outputs/fever/<method>/<run_id>/`.

Config captures dataset/split, model revision when available, method, sample
IDs, seed, batch size, generation parameters, step budget, CoT-SC parameters and
threshold, prompt/code versions, and timestamp. Predictions retain canonical
labels, invalid parse state, correctness, steps, retrieval tool calls,
termination, latency, votes, and hybrid execution path. Metrics report Accuracy,
invalid count, class distributions, per-class accuracy, confusion matrix,
operational averages, runtime, and termination counts.

<!-- PAGEBREAK -->

## Verification Checkpoint

| Requirement | Status |
|---|---|
| Claim-only | PASS |
| No gold evidence in prompts | PASS |
| Three Appendix C.2 exemplars | PASS |
| Standard/CoT/Act/ReAct controlled ablations | PASS |
| Dense ReAct Thought/Action/Observation | PASS |
| Search/Lookup/Finish | PASS |
| CoT-SC n=21 and temperature=0.7 | PASS |
| FEVER ReAct max steps=5 | PASS |
| CoT-SC -> ReAct n/2 heuristic | PASS |
| ReAct -> CoT-SC failure fallback | PASS |
| Accuracy primary metric | PASS |
| Batch state isolation | PASS |
| HotpotQA regression | PASS |
| No fine-tuning | PASS |

## Next Work

Do not automatically start Sprint 7. The next user-driven work is to run the
seven 500-example FEVER benchmarks on Colab, then perform a controlled comparison
and complete the broader Sprint 7 research report. FEVER scores remain TBD until
those runs exist.
