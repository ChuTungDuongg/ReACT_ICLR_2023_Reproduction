# Configuration

`default.yaml` contains version-controlled experiment defaults:

- HotpotQA dataset `hotpotqa/hotpot_qa`, `distractor`, `validation`;
- sample count, seed, and maximum agent steps;
- generation temperature, top-p, and maximum new tokens;
- output directory and live log level.

`src/react_reproduction/config.py` validates every value and resolves relative
paths from the repository root. CLI flags override run-specific values.
`REACT_OUTPUT_DIR` and `REACT_LOG_LEVEL` can override their YAML equivalents.

Do not store Hugging Face tokens, model weights, downloaded data, or generated
results here. Use `.env` for local secrets; it is ignored by Git.
