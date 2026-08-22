# Configuration

This directory contains version-controlled defaults for reproducible
experiments. Configuration must describe an experiment; it must not contain
credentials, downloaded data, model weights, or generated results.

## `default.yaml`

The default configuration currently defines:

- project name, output directory, and log level;
- the official HotpotQA dataset name, subset, and split;
- sample count, random seed, and future maximum agent steps;
- generation defaults such as temperature, top-p, and maximum new tokens.

The loader in `src/react_reproduction/config.py` validates these values and
resolves relative paths against the repository root. `REACT_OUTPUT_DIR` and
`REACT_LOG_LEVEL` can override their YAML equivalents. Model and run-specific
CLI overrides will be connected to benchmark execution in Sprint 2.

## Rules

- Keep defaults small enough for a smoke run.
- Keep dataset/model identifiers in configuration rather than duplicating them
  across modules.
- Never commit tokens or machine-specific absolute paths.
- Add new task sections only in the sprint that implements the task.
