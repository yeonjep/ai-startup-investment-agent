"""Markdown→한글 PDF, 가독성 최소 9pt / 최대 5페이지. 초과하면 명시적 오류."""
from pathlib import Path
import pymupdf
from markdown_it import MarkdownIt


def export_report_pdf(markdown: str, path: str | Path) -> str:
    path = Path(path)
    renderer = MarkdownIt('commonmark', {'html': False}).enable('table').disable('image')
    if '\n# REFERENCE' not in markdown:
        raise ValueError('REFERENCE section is required')
    body, references = markdown.rsplit('\n# REFERENCE', 1)
    html_sections = [renderer.render(body), renderer.render('# REFERENCE' + references)]
    media = pymupdf.paper_rect('a4')
    content = pymupdf.Rect(40, 40, media.width - 40, media.height - 40)
    for size in (10.5, 10, 9.5, 9):
        css = f'''body {{ font-family: sans-serif; font-size: {size}pt; line-height: 1.35; }}
        h1 {{font-size: 17pt; color: #123454; margin-top: 16pt;}}
        h2 {{font-size: 12pt; color: #123454; margin-top: 12pt;}}
        p {{margin: 5pt 0;}} table {{border-collapse: collapse; width: 100%; font-size: 9pt;}}
        th,td {{border-bottom: 0.5pt solid #cccccc; padding: 4pt;}}
        th {{font-weight: bold;}} li {{page-break-inside: avoid;}} a {{color: #244869;}}'''
        pdf = pymupdf.open()
        for html in html_sections:
            story = pymupdf.Story(html=html, user_css=css)
            section = story.write_with_links(lambda rect_num, filled: (media, content, None))
            pdf.insert_pdf(section)
            section.close()
        if len(pdf) <= 5:
            for n, page in enumerate(pdf):
                page.insert_text((media.width - 70, media.height - 20), f'{n+1} / {len(pdf)}', fontsize=8)
            path.parent.mkdir(parents=True, exist_ok=True)
            pdf.save(path, garbage=4, deflate=True)
            pdf.close()
            return str(path)
        pdf.close()
    raise ValueError('보고서가 5페이지를 초과합니다. Markdown 내용을 축약한 뒤 PDF를 다시 생성하세요.')
