# Tests

The test suite validates research infrastructure without downloading a large
language model. Tests should be deterministic, fast, and runnable on Windows,
Linux, and Google Colab.

Run all tests from the repository root:

```bash
pytest
```

## Current coverage

- `test_smoke.py`: CLI help, project configuration, and `doctor` output.
- `test_metrics.py`: HotpotQA normalization and Exact Match.
- `test_hotpotqa.py`: deterministic sampling, loader arguments, metadata, and
  sample-size validation using in-memory data.
- `test_experiment_serialization.py`: end-to-end mock runner, run directory,
  config, predictions, and metrics serialization.
- `conftest.py`: makes the `src` package importable after a plain clone.

## Rules

- Do not require model downloads in unit tests.
- Inject dataset/model doubles instead of relying on network availability.
- Write temporary test artifacts only under the project `outputs/` directory
  and remove the exact verified test directory afterward.
- Add parser, loop, tool, and trajectory tests in their owning sprints.
