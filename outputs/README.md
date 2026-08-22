# Runtime benchmark outputs

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
- `metrics.json` is written after successful aggregation.
- `run.log` retains the output streamed live to the terminal/Colab cell.

Do not commit run data, models, caches, datasets, credentials, or manually
maintained documentation here. Stable documentation belongs in `output/`.
