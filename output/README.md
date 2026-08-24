# Project documents

> Language: English | [Tieng Viet](README.vi.md)

This directory contains stable, tracked documentation:

```text
output/
  react-reproduction-roadmap.md
  pdf/react-reproduction-roadmap.pdf
```

The Markdown file is the roadmap source of truth. Regenerate the PDF with:

```bash
python scripts/generate_roadmap_pdf.py
```

`reportlab` is included in the project requirements for this generator.

The roadmap reflects Sprint 6 completion and the paper-faithful FEVER settings.
Runtime predictions belong in the ignored `outputs/` directory.
