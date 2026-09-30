# 설계 기준: docs/DESIGN.md B-3 export_report_pdf (markdown, path → 한글 폰트 PDF 경로).

import re
from pathlib import Path

import markdown
import pymupdf
from langchain_core.tools import tool

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
PAGE_RECT = pymupdf.paper_rect("a4")
MARGIN_X = 78
MARGIN_Y = 62

CSS = """
@font-face { font-family: nanum; src: url(NanumGothic-Regular.ttf); }
@font-face { font-family: nanum; font-weight: bold; src: url(NanumGothic-Bold.ttf); }
body { font-family: nanum; font-size: 10pt; line-height: 1.75; }
h1 { font-size: 15pt; margin: 14pt 0 6pt 0; }
h2 { font-size: 12.5pt; margin: 14pt 0 5pt 0; }
h3 { font-size: 11pt; margin: 10pt 0 4pt 0; }
p { margin: 5pt 0; }
li { margin: 2pt 0; }
img { width: 92%; }
table { border-collapse: collapse; width: 100%; font-size: 8.5pt; }
th, td { border: 0.5pt solid #666666; padding: 3pt 5pt; }
th { font-weight: bold; }
"""


def render_markdown_pdf(markdown_text: str, path: str | Path, asset_dir: str | Path | None = None) -> int:
    """Render markdown to a Korean-font PDF and return the page count (checked with PyMuPDF)."""
    html = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"])
    # 긴 URL이 페이지 밖으로 잘리지 않도록 줄바꿈 가능 지점을 넣는다
    html = re.sub(r"[^\s<>]{40,}", lambda m: re.sub(r"(.{24})", r"\1<wbr>", m.group(0)), html)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = pymupdf.Archive(str(FONT_DIR))
    if asset_dir:  # 차트 이미지 등 마크다운이 상대경로로 참조하는 파일
        archive.add(str(asset_dir))
    story = pymupdf.Story(html=html, user_css=CSS, archive=archive)
    writer = pymupdf.DocumentWriter(str(output))
    content_rect = PAGE_RECT + (MARGIN_X, MARGIN_Y, -MARGIN_X, -MARGIN_Y)
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
