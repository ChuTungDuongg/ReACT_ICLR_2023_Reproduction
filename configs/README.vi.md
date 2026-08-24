# Cấu hình

> Ngôn ngữ: [English](README.md) | Tiếng Việt

`default.yaml` định nghĩa hai dataset và generation defaults:

- HotpotQA: `hotpotqa/hotpot_qa`, `distractor`, `validation`, ReAct 7 bước;
- FEVER: official `ysymyth/ReAct` `data/paper_dev.jsonl`, pin revision, ReAct 5 bước;
- sample count, seed, temperature/top-p/token budget;
- output directory và log level.

CLI flags override setting theo run. CoT-SC có paper defaults ở CLI: 21 sample,
temperature 0.7. Environment overrides được hỗ trợ là `REACT_OUTPUT_DIR` và
`REACT_LOG_LEVEL`. Credentials phải ở `.env` đã bị ignore.
