# Runtime benchmark outputs

> Language: English | [Tieng Viet](README.vi.md)

Generated runs are ignored by Git except for this README and `.gitkeep`:

```text
outputs/<task>/<method>/<run_id>/
  config.json
  metrics.json
  predictions.jsonl
  trajectories.jsonl
  run.log
```

Both `hotpotqa` and `fever` use this layout for all seven methods. Config records
sample IDs, prompt/code versions, model revision when available, batching,
generation, task step budget, and CoT-SC settings. FEVER uses Accuracy and adds
invalid counts, label distributions, per-class accuracy, and a confusion matrix.
CoT-SC/hybrid predictions expose vote and execution-path metadata; complete
sample text remains in trajectories. Files are flushed incrementally for Colab
diagnostics.

Do not commit runs, model weights, datasets, caches, credentials, or temporary
files. Stable documents belong in `output/`.
