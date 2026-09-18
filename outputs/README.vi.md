# Runtime benchmark outputs

> Ngôn ngữ: [English](README.md) | Tiếng Việt

Thư mục này chứa các benchmark artifact thô do máy tạo:

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
execution path; full samples nằm trong trajectories. Hai cây benchmark hoàn tất
`hotpotqa` và `fever` được `.gitignore` cho phép đưa vào Git để tái lập kết quả.
Không đặt weights, dataset, cache, credentials, file tạm hoặc báo cáo đọc bởi
con người ở đây; các báo cáo thuộc `reports/`.
