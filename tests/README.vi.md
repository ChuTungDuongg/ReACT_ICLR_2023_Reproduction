# Tests

> Ngôn ngữ: [English](README.md) | Tiếng Việt

Chạy test suite offline từ root:

```bash
python -m pytest -q
```

Tests dùng scripted LLM, dataset inject và fake Wikipedia; không tải Qwen và
không gọi dịch vụ live. Phạm vi gồm HotpotQA regression, FEVER label parser,
claim-only loader, đúng ba exemplars Appendix C.2 và bốn ablation, action space,
CoT-SC 21 sample ở temperature 0.7, biên fallback 11/21 và 10/21, hai hybrid,
batch isolation, Accuracy, serialization và paper-fidelity audit.
