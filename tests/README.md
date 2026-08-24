# Tests

> Language: English | [Tieng Viet](README.vi.md)

Run the offline suite from the repository root:

```bash
python -m pytest -q
```

The suite uses scripted LLM providers, injected datasets, and fake Wikipedia
clients; it does not download Qwen or call live services. Coverage includes:

- HotpotQA regression for all existing agents, metrics, prompts, and artifacts;
- FEVER label normalization and invalid outputs;
- deterministic claim-only loading with no gold-evidence leakage;
- the exact three Appendix C.2 examples and controlled prompt ablations;
- Search/Lookup/Finish and five-sentence/suggestion semantics;
- all seven FEVER methods, 21-sample CoT-SC, temperature 0.7, and vote metadata;
- `11/21` no-fallback and `10/21` fallback boundaries;
- ReAct failure fallback, batch state isolation, Accuracy, and serialization;
- an explicit paper-fidelity audit.
