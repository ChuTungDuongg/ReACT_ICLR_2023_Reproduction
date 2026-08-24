# Configuration

> Language: English | [Tieng Viet](README.vi.md)

`default.yaml` defines both dataset sources and shared generation defaults:

- HotpotQA: `hotpotqa/hotpot_qa`, `distractor`, `validation`, 7 ReAct steps;
- FEVER: official `ysymyth/ReAct` `data/paper_dev.jsonl`, pinned revision, 5 ReAct steps;
- default sample count, seed, generation temperature/top-p/token budget;
- output directory and log level.

CLI flags override run settings. CoT-SC defaults are CLI-level paper constants:
21 samples and temperature 0.7. `REACT_OUTPUT_DIR` and `REACT_LOG_LEVEL` are the
supported environment overrides. Keep credentials in the ignored `.env` file.
