# Runtime benchmark outputs

> Language: English | [Tieng Viet](README.vi.md)

This directory contains raw machine-generated benchmark artifacts:

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

The completed HotpotQA and FEVER benchmark trees are explicitly allowed by
`.gitignore` so they can be versioned for reproducibility. Do not place model
weights, datasets, caches, credentials, temporary files, or human-readable
reports here. Stable documents and analyses belong in `reports/`.
