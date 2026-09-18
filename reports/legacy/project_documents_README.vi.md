# Tài liệu project

> Ngôn ngữ: [English](README.md) | Tiếng Việt

Thư mục này lưu tài liệu ổn định được track:

```text
reports/legacy/
  react-reproduction-roadmap-sprint6.md
  react-reproduction-roadmap-sprint6.pdf
```

Markdown là source of truth của roadmap. Tạo lại PDF bằng:

```bash
python scripts/generate_roadmap_pdf.py
```

Dependency `reportlab` của generator đã có trong project requirements.

Roadmap phản ánh Sprint 6 hoàn tất và các setting FEVER theo paper. Runtime
predictions phải nằm trong `outputs/` đã bị ignore.
