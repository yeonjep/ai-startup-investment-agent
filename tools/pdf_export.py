# 설계 기준: docs/DESIGN.md B-3 export_report_pdf (markdown, path → 한글 폰트 PDF 경로).
# 본문은 이어서 배치하고, REFERENCE만 새 페이지에서 시작한다.

import re
from pathlib import Path

import markdown
import pymupdf
from langchain_core.tools import tool

FONT_DIR = Path(__file__).resolve().parents[1] / "assets" / "fonts"
PAGE_RECT = pymupdf.paper_rect("a4")
MARGIN_X = 42
MARGIN_TOP = 36
MARGIN_BOTTOM = 40

CSS = """
@font-face { font-family: nanum; src: url(NanumGothic-Regular.ttf); }
@font-face { font-family: nanum; font-weight: bold; src: url(NanumGothic-Bold.ttf); }
body { font-family: nanum; font-size: 10.5pt; line-height: 1.72; color: #1f2933; }
p { margin: 6pt 0; }
li { margin: 8pt 0; line-height: 1.7; }
ul, ol { margin: 8pt 0 10pt 16pt; }
img { width: 100%; margin: 6pt 0 4pt 0; }
a { color: #1d4e89; }
table { border-collapse: collapse; width: 100%; font-size: 8.4pt; line-height: 1.4; margin: 6pt 0 10pt 0; table-layout: fixed; }
th, td { border: 0.4pt solid #d5dde6; padding: 5pt 6pt; vertical-align: top; line-height: 1.45; }
td.hdr { color: #1b3a5f; font-weight: bold; border-bottom: 1pt solid #8aa0b8; }
.doc-title { font-size: 18pt; font-weight: bold; color: #1e4e8c; margin: 0 0 3pt 0; }
.doc-sub { font-size: 9.5pt; color: #5c6b7a; margin: 0 0 8pt 0; }
.toc { background-color: #f4f6f8; border: 0.6pt solid #e3e7ed; padding: 0 12pt; margin: 0 0 12pt 0; }
.toc-label { font-weight: bold; color: #1f2933; margin: 0 0 3pt 0; }
.toc-item { margin: 1pt 0; color: #1f2933; }
.section { background-color: #2f6fad; color: white; font-size: 12pt; font-weight: bold; line-height: 1.8; padding: 0 8pt; margin: 13pt 0 7pt 0; }
.subsection { color: #1e4e8c; font-size: 11pt; font-weight: bold; border-left: 3.5pt solid #2f6fad; padding: 0 0 0 7pt; margin: 10pt 0 4pt 0; }
.callout { background-color: #fff6e5; border-left: 3.5pt solid #e0a030; padding: 0 8pt; margin: 8pt 0; }
.page-break { break-before: page; page-break-before: always; margin-top: 0; }
"""

_COLUMN_WIDTHS = {
    ("항목", "지표", "원값", "점수", "판단 근거"): ["14%", "16%", "20%", "12%", "38%"],
    ("기업", "목표 시장 규모", "규모 범위·기준", "목표 시장 CAGR", "CAGR 범위·기간"): ["14%", "18%", "24%", "18%", "26%"],
    ("경쟁사", "국가", "비교"): ["18%", "12%", "70%"],
    ("후보", "국가", "선정 상태", "제외 기준", "사유"): ["16%", "10%", "12%", "14%", "48%"],
}


def _plain(html: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", html)).strip()


def _style_tables(html: str) -> str:
    def style(match: re.Match) -> str:
        table = match.group(0)
        headers = tuple(_plain(cell) for cell in re.findall(r"<th>(.*?)</th>", table, flags=re.S))
        widths = _COLUMN_WIDTHS.get(headers)
        if not widths:
            return table
        columns = "".join(f'<col style="width:{width}">' for width in widths)
        return table.replace("<table>", f"<table><colgroup>{columns}</colgroup>", 1)

    return re.sub(r"<table>.*?</table>", style, html, flags=re.S)


def prepare_report_html(html: str) -> str:
    """짧은 절의 강제 쪽나눔을 없애고, 제목·목차·표 스타일용 표시를 붙인다."""
    html = re.sub(r'<div class="page-break"></div>\s*', "", html)
    html = html.replace("<h1>REFERENCE</h1>", '<div class="page-break"></div><h1>REFERENCE</h1>', 1)

    headings = [_plain(text) for text in re.findall(r"<h[12]>(.*?)</h[12]>", html, flags=re.S)]
    toc = "".join(f'<div class="toc-item">{text}</div>' for text in headings)
    subtitle = "Semiconductor"
    meta = re.search(r"<p>(.*?)</p>", html, flags=re.S)
    if meta and any(token in meta.group(1) for token in ("총점", "평가 기준일", "전원 보류", "평가 대상")):
        subtitle = _plain(meta.group(1))
        html = html[: meta.start()] + html[meta.end() :]

    html = re.sub(
        r"<p>(<strong>(?:판단 사유|결측·선정 불확실성)</strong>.*?)</p>",
        r'<div class="callout">\1</div>',
        html,
        flags=re.S,
    )
    html = re.sub(r"<p><strong>(시장 지표|경쟁사 비교)</strong></p>", r'<div class="subsection">\1</div>', html)
    html = re.sub(r"<h3>(.*?)</h3>", r'<div class="subsection">\1</div>', html, flags=re.S)
    html = re.sub(r"<h[12]>(.*?)</h[12]>", r'<div class="section">\1</div>', html, flags=re.S)
    # th 배경은 다음 페이지 본문까지 반복되므로, 머리글은 굵은 글씨만 남긴다.
    html = re.sub(r"</?thead>", "", html)
    html = _style_tables(html)
    html = html.replace("<th>", '<td class="hdr">').replace("</th>", "</td>")
    header = (
        '<div class="doc-title">AI 스타트업 투자 평가 보고서</div>'
        f'<div class="doc-sub">{subtitle}</div>'
        f'<div class="toc"><div class="toc-label">목차</div>{toc}</div>'
    )
    return header + html


def _stamp_page_numbers(path: Path) -> int:
    document = pymupdf.open(path)
    try:
        for index, page in enumerate(document, start=1):
            box = pymupdf.Rect(0, page.rect.height - 28, page.rect.width, page.rect.height - 12)
            page.insert_textbox(
                box,
                str(index),
                fontsize=9,
                fontname="helv",
                align=pymupdf.TEXT_ALIGN_CENTER,
                color=(0.42, 0.46, 0.52),
            )
        temporary = path.with_suffix(".tmp.pdf")
        document.save(temporary)
    finally:
        document.close()
    temporary.replace(path)
    with pymupdf.open(path) as stamped:
        return stamped.page_count


def _soften_long_text(html: str) -> str:
    """태그 속성이 아니라 본문 글자만 나눠, 긴 URL이 페이지 밖으로 나가지 않게 한다."""

    def soften(text: str) -> str:
        return re.sub(r"[^\s<>]{40,}", lambda match: re.sub(r"(.{24})", r"\1<wbr>", match.group(0)), text)

    return "".join(part if part.startswith("<") else soften(part) for part in re.split(r"(<[^>]+>)", html))


def render_markdown_pdf(markdown_text: str, path: str | Path, asset_dir: str | Path | None = None) -> int:
    """Render markdown to a Korean-font PDF and return the page count (checked with PyMuPDF)."""
    html = markdown.markdown(markdown_text, extensions=["tables", "sane_lists"])
    html = prepare_report_html(html)
    html = _soften_long_text(html)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    archive = pymupdf.Archive(str(FONT_DIR))
    if asset_dir:  # 차트 이미지 등 마크다운이 상대경로로 참조하는 파일
        archive.add(str(asset_dir))
    story = pymupdf.Story(html=html, user_css=CSS, archive=archive)
    writer = pymupdf.DocumentWriter(str(output))
    content_rect = PAGE_RECT + (MARGIN_X, MARGIN_TOP, -MARGIN_X, -MARGIN_BOTTOM)
    more = True
    while more:
        device = writer.begin_page(PAGE_RECT)
        more, _ = story.place(content_rect)
        story.draw(device)
        writer.end_page()
    writer.close()
    return _stamp_page_numbers(output)


@tool("export_report_pdf")
def export_report_pdf(markdown_text: str, path: str) -> str:
    """Convert a markdown report to a PDF with a Korean font and return its path."""
    render_markdown_pdf(markdown_text, path)
    return str(path)
