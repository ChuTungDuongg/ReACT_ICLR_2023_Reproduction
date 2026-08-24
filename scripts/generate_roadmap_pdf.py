"""Generate the tracked roadmap PDF from its Markdown source."""

from __future__ import annotations

from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "output" / "react-reproduction-roadmap.md"
TARGET = ROOT / "output" / "pdf" / "react-reproduction-roadmap.pdf"


def main() -> None:
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    lines = SOURCE.read_text("utf-8").splitlines()
    styles = _styles()
    story = _parse_markdown(lines, styles)
    document = SimpleDocTemplate(
        str(TARGET),
        pagesize=A4,
        leftMargin=17 * mm,
        rightMargin=17 * mm,
        topMargin=18 * mm,
        bottomMargin=17 * mm,
        title="ReAct Paper Reproduction Roadmap",
        author="ReAct Paper Reproduction",
        subject="Sprint 6 completion roadmap",
    )
    document.build(story, onFirstPage=_page, onLaterPages=_page)
    print(TARGET)


def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "RoadmapTitle",
            parent=base["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            textColor=colors.HexColor("#17324D"),
            alignment=TA_CENTER,
            spaceAfter=10,
        ),
        "h2": ParagraphStyle(
            "RoadmapH2",
            parent=base["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#8A3B12"),
            spaceBefore=11,
            spaceAfter=5,
            keepWithNext=True,
        ),
        "body": ParagraphStyle(
            "RoadmapBody",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9.2,
            leading=13,
            textColor=colors.HexColor("#20262E"),
            spaceAfter=5,
        ),
        "bullet": ParagraphStyle(
            "RoadmapBullet",
            parent=base["BodyText"],
            fontName="Helvetica",
            fontSize=9,
            leading=12.5,
            leftIndent=12,
            firstLineIndent=-7,
            bulletIndent=3,
            spaceAfter=2,
        ),
        "code": ParagraphStyle(
            "RoadmapCode",
            parent=base["Code"],
            fontName="Courier",
            fontSize=7.8,
            leading=10.5,
            leftIndent=7,
            rightIndent=7,
            borderColor=colors.HexColor("#CBD5E1"),
            borderWidth=0.6,
            borderPadding=7,
            backColor=colors.HexColor("#F5F7FA"),
            spaceAfter=7,
        ),
    }


def _parse_markdown(
    lines: list[str],
    styles: dict[str, ParagraphStyle],
) -> list[object]:
    story: list[object] = []
    paragraph: list[str] = []
    index = 0

    def flush_paragraph() -> None:
        if paragraph:
            story.append(Paragraph(_inline(" ".join(paragraph)), styles["body"]))
            paragraph.clear()

    while index < len(lines):
        line = lines[index]
        if line.startswith("```"):
            flush_paragraph()
            code_lines: list[str] = []
            index += 1
            while index < len(lines) and not lines[index].startswith("```"):
                code_lines.append(lines[index])
                index += 1
            story.append(
                Paragraph(
                    "<br/>".join(escape(item).replace(" ", "&nbsp;") for item in code_lines),
                    styles["code"],
                )
            )
        elif line.startswith("| "):
            flush_paragraph()
            table_lines = [line]
            index += 1
            while index < len(lines) and lines[index].startswith("|"):
                table_lines.append(lines[index])
                index += 1
            story.append(_table(table_lines, styles["body"]))
            index -= 1
        elif line == "<!-- PAGEBREAK -->":
            flush_paragraph()
            story.append(PageBreak())
        elif line.startswith("# "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[2:]), styles["title"]))
            story.append(Spacer(1, 2 * mm))
        elif line.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(_inline(line[3:]), styles["h2"]))
        elif line.startswith("- "):
            flush_paragraph()
            story.append(
                Paragraph(_inline(line[2:]), styles["bullet"], bulletText="-")
            )
        elif not line.strip():
            flush_paragraph()
        else:
            paragraph.append(line.strip())
        index += 1
    flush_paragraph()
    return story


def _table(lines: list[str], body_style: ParagraphStyle) -> Table:
    rows = [
        [cell.strip() for cell in line.strip().strip("|").split("|")]
        for line in lines
    ]
    if len(rows) > 1 and all(set(cell) <= {"-", ":"} for cell in rows[1]):
        rows.pop(1)
    header_style = ParagraphStyle(
        "RoadmapTableHeader",
        parent=body_style,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        fontSize=8.7,
        leading=11,
        spaceAfter=0,
    )
    cell_style = ParagraphStyle(
        "RoadmapTableCell",
        parent=body_style,
        fontSize=8.7,
        leading=11,
        spaceAfter=0,
    )
    rendered = []
    for row_index, row in enumerate(rows):
        style = header_style if row_index == 0 else cell_style
        rendered.append([Paragraph(_inline(cell), style) for cell in row])
    column_count = max(len(row) for row in rendered)
    page_width = A4[0] - 34 * mm
    widths = [page_width / column_count] * column_count
    table = Table(rendered, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#17324D")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F8FAFC")),
                ("GRID", (0, 0), (-1, -1), 0.45, colors.HexColor("#B8C2CC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return KeepTogether([table, Spacer(1, 2 * mm)])


def _inline(value: str) -> str:
    escaped = escape(value)
    pieces = escaped.split("`")
    return "".join(
        f'<font name="Courier">{piece}</font>' if index % 2 else piece
        for index, piece in enumerate(pieces)
    )


def _page(canvas: object, document: object) -> None:
    canvas.saveState()
    canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
    canvas.line(17 * mm, 13 * mm, A4[0] - 17 * mm, 13 * mm)
    canvas.setFont("Helvetica", 7.5)
    canvas.setFillColor(colors.HexColor("#52606D"))
    canvas.drawString(17 * mm, 8.5 * mm, "ReAct Paper Reproduction - Sprint 6")
    canvas.drawRightString(
        A4[0] - 17 * mm,
        8.5 * mm,
        f"Page {document.page}",
    )
    canvas.restoreState()


if __name__ == "__main__":
    main()
