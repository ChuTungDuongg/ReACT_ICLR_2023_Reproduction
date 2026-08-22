# Runtime benchmark outputs

> 🌐 **Language:** English | [Tiếng Việt](README.vi.md)

This directory contains generated runs and is ignored by Git except for this
README and `.gitkeep`.

```text
outputs/<task>/<method>/<UTC timestamp>/
├── config.json
├── metrics.json
├── predictions.jsonl
├── trajectories.jsonl   # Act-only and ReAct
└── run.log
```

- `config.json` records the complete run settings.
- `predictions.jsonl` is appended and flushed after every example.
- `trajectories.jsonl` records every model output, Thought, canonical Action,
  and real Observation; it is also flushed per example.
- `metrics.json` contains all 12 official HotpotQA answer/supporting-fact/joint
  metrics plus operational metrics and is written after successful aggregation.
- `run.log` retains the output streamed live to the terminal/Colab cell.

The same full final metric set is printed to stdout unless `--quiet` is used.
When an agent emits no `(title, sentence_id)` evidence pairs, SP/joint metrics
score the empty prediction and evidence coverage makes that limitation explicit.

Do not commit run data, models, caches, datasets, credentials, or manually
maintained documentation here. Stable documentation belongs in `output/`.
