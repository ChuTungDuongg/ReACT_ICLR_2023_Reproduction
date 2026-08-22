# Experiment Outputs

This directory is reserved for generated benchmark runs. Runtime artifacts are
ignored by Git; only this README and `.gitkeep` are version-controlled.

The Sprint 1 runner currently produces:

```text
outputs/
└── <task>/
    └── <method>/
        └── <UTC timestamp>/
            ├── config.json
            ├── metrics.json
            └── predictions.jsonl
```

- `config.json` records the complete run configuration.
- `predictions.jsonl` contains one flushed record per example.
- `metrics.json` contains final aggregate metrics.

Later sprints will add `trajectories.jsonl`, `run.log`, and comparison reports.
Do not place source code, manually maintained documentation, model weights, or
credentials here.

Human-facing generated project documents belong in `output/`, not `outputs/`.
