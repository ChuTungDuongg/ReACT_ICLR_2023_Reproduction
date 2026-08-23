# Runtime benchmark outputs

> 🌐 **Language:** English | [Tiếng Việt](README.vi.md)

This directory contains generated runs and is ignored by Git except for this
README and `.gitkeep`.

```text
outputs/<task>/<method>/<UTC timestamp>/
├── config.json
├── metrics.json
├── predictions.jsonl
├── trajectories.jsonl   # Act-only, ReAct, and hybrid methods
└── run.log
```

- `config.json` records the complete run settings, including `batch_size`.
- `predictions.jsonl` is appended in dataset order after each completed batch.
- `trajectories.jsonl` records every model output, Thought, canonical Action,
  and real Observation; it is also flushed per example.
- `metrics.json` contains all 12 official HotpotQA answer/supporting-fact/joint
  metrics plus operational metrics and is written after successful aggregation.
- `run.log` retains the output streamed live to the terminal/Colab cell.

For `react-cot-sc` and `cot-sc-react`, each prediction also records CoT-SC vote
counts/confidence, the selected path, and whether fallback was used. Hybrid
trajectory steps include a `phase` field with `react` or `cot_sc`.
With hybrid batching, an interrupted run can lose only the currently executing
batch; all earlier batches remain on disk.

The same full final metric set is printed to stdout unless `--quiet` is used.
When an agent emits no `(title, sentence_id)` evidence pairs, SP/joint metrics
score the empty prediction and evidence coverage makes that limitation explicit.

Do not commit run data, models, caches, datasets, credentials, or manually
maintained documentation here. Stable documentation belongs in `output/`.
