# Reproduction reports

This directory contains human-readable analyses and presentation-ready research artifacts. Raw benchmark records remain under `outputs/`:

- `outputs/hotpotqa/` contains the selected HotpotQA runs.
- `outputs/fever/` contains the selected FEVER runs.

The selection utility validates every candidate against its config, metrics, prediction count, unique example IDs, and completion log. It prefers the largest valid run for each task and method, then the newest timestamp. This rule prevents a newer smoke test from superseding a full benchmark.

## Selected runs

| Task | Method | Selected run directory |
|---|---|---|
| HotpotQA | Standard | `outputs/hotpotqa/standard/2026-08-24_133826` |
| HotpotQA | CoT | `outputs/hotpotqa/cot/2026-08-24_143705` |
| HotpotQA | CoT-SC | `outputs/hotpotqa/cot-sc/2026-08-24_134140` |
| HotpotQA | Act | `outputs/hotpotqa/act/2026-08-23_163720` |
| HotpotQA | ReAct | `outputs/hotpotqa/react/2026-08-23_082017` |
| HotpotQA | ReAct → CoT-SC | `outputs/hotpotqa/react-cot-sc/2026-08-23_132325` |
| HotpotQA | CoT-SC → ReAct | `outputs/hotpotqa/cot-sc-react/2026-08-23_092357` |
| FEVER | Standard | `outputs/fever/standard/2026-09-17_185849` |
| FEVER | CoT | `outputs/fever/cot/2026-09-17_190026` |
| FEVER | CoT-SC | `outputs/fever/cot-sc/2026-09-17_190339` |
| FEVER | Act | `outputs/fever/act/2026-09-17_200652` |
| FEVER | ReAct | `outputs/fever/react/2026-09-17_173812` |
| FEVER | ReAct → CoT-SC | `outputs/fever/react-cot-sc/2026-09-17_223327` |
| FEVER | CoT-SC → ReAct | `outputs/fever/cot-sc-react/2026-09-17_203508` |

The arrow labels above are execution order, as confirmed by the implementation: `react-cot-sc` means ReAct → CoT-SC, and `cot-sc-react` means CoT-SC → ReAct.

## Generated artifacts

- `results_summary.json` is the complete machine-readable summary, including configuration, metrics, terminations, execution paths, and trajectory counts.
- `results_summary.csv` is a compact 14-row table for downstream analysis.
- `hotpotqa_report.md` and `fever_report.md` are dataset-specific academic reports.
- `react_reproduction_report.tex` is the combined LaTeX study, and
  `react_reproduction_report.pdf` is its compiled pdfLaTeX output.
- `figures/` contains vector PDF plots generated from the JSON summary.
- `legacy/` preserves superseded project-documentation material and the earlier HotpotQA PDF.

The historical Sprint 6 roadmap and its PDF are preserved in `legacy/` with
descriptive `react-reproduction-roadmap-sprint6` filenames.

## Regenerate summaries and figures

From the repository root, run:

```bash
python scripts/summarize_reproduction_results.py
```

To regenerate or validate only part of the output:

```bash
python scripts/summarize_reproduction_results.py --summaries-only
python scripts/summarize_reproduction_results.py --figures-only
python scripts/summarize_reproduction_results.py --check
```

The figure generator uses ReportLab, which is listed in `requirements.txt`. It writes deterministic vector PDFs to `reports/figures/`.

## Compile the LaTeX report

With a LaTeX distribution that provides `pdflatex`, run two passes from the
repository root:

```bash
cd reports
pdflatex -interaction=nonstopmode -halt-on-error react_reproduction_report.tex
pdflatex -interaction=nonstopmode -halt-on-error react_reproduction_report.tex
```

The expected output is `reports/react_reproduction_report.pdf`. The source was
verified with MiKTeX pdfTeX 1.40.28. It uses only local figures and contains its
bibliography directly, so no network access or BibTeX pass is required.
