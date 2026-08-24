# Runtime benchmark outputs

> Ngôn ngữ: [English](README.md) | Tiếng Việt

Run artifacts bị Git ignore, trừ README và `.gitkeep`:

```text
outputs/<task>/<method>/<run_id>/
  config.json
  metrics.json
  predictions.jsonl
  trajectories.jsonl
  run.log
```

HotpotQA và FEVER dùng chung layout cho cả bảy method. Config lưu sample IDs,
prompt/code version, model revision nếu có, batch/generation settings, step
budget và CoT-SC settings. FEVER dùng Accuracy cùng invalid counts, label
distribution, per-class accuracy và confusion matrix. CoT-SC/hybrid lưu vote và
execution path; full samples nằm trong trajectories. Không commit benchmark
runs, weights, dataset, cache, credentials hoặc file tạm.
