# 설계 기준: docs/DESIGN.md B-3 export_report_pdf (markdown, path → 한글 폰트 PDF 경로).

import re
from pathlib import Path

import markdown
import pymupdf
from langchain_core.tools import tool

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
PAGE_RECT = pymupdf.paper_rect("a4")
MARGIN = 40

CSS = """
@font-face { font-family: nanum; src: url(NanumGothic-Regular.ttf); }
@font-face { font-family: nanum; font-weight: bold; src: url(NanumGothic-Bold.ttf); }
body { font-family: nanum; font-size: 8.5pt; line-height: 1.3; }
h1 { font-size: 12.5pt; margin: 7pt 0 2pt 0; }
h2 { font-size: 10.5pt; margin: 6pt 0 2pt 0; }
p { margin: 2pt 0; }
li { margin: 0; }
table { border-collapse: collapse; width: 100%; font-size: 7.5pt; }
th, td { border: 0.5pt solid #666666; padding: 1.5pt 3pt; }
th { background-color: #e8e8e8; }
"""


def render_markdown_pdf(markdown_text: str, path: str | Path) -> int:
    """Render markdown to a Korean-font PDF and return the page count (checked with PyMuPDF)."""
    html = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"])
    # 긴 URL이 페이지 밖으로 잘리지 않도록 줄바꿈 가능 지점을 넣는다
    html = re.sub(r"[^\s<>]{40,}", lambda m: re.sub(r"(.{24})", r"\1<wbr>", m.group(0)), html)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    story = pymupdf.Story(html=html, user_css=CSS, archive=pymupdf.Archive(str(FONT_DIR)))
    writer = pymupdf.DocumentWriter(str(output))
    content_rect = PAGE_RECT + (MARGIN, MARGIN, -MARGIN, -MARGIN)
    more = True
    while more:
        device = writer.begin_page(PAGE_RECT)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    with pymupdf.open(output) as document:
        return document.page_count


@tool("export_report_pdf")
def export_report_pdf(markdown_text: str, path: str) -> str:
    """Convert a markdown report to a PDF with a Korean font and return its path."""
    render_markdown_pdf(markdown_text, path)
    return str(path)
