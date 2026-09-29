from pathlib import Path
import shutil

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "data" / "docs"
MAX_TOTAL_PAGES = 200

DOCUMENTS = (
    ("semiconductor_outlook_2026", "2026 Semiconductor Industry Outlook _ Deloitte Insights.pdf", None),
    ("ai_chip_strategy", "251219+(별첨)+AI반도체+산업+도약+전략(안)+요약본.pdf", None),
    ("global_ai_chip_trends", "AI반도체 글로벌 첨단 기술·산업 동향 조사 및 대응방향 연구.pdf", (27, 81)),
    ("ai_chip_market_outlook", "[초점] 새로운 기회의 창으로 AI반도체 시장 현황과 전망.pdf", None),
)


def prepare_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    output_pages = []

    for doc_id, source_name, page_range in DOCUMENTS:
        source_path = RAW_DIR / source_name
        output_path = DOCS_DIR / f"{doc_id}.pdf"
        if not source_path.is_file():
            raise FileNotFoundError(f"Source PDF not found: {source_path}")

        if page_range is None:
            shutil.copy2(source_path, output_path)
        else:
            first_page, last_page = page_range
            with pymupdf.open(source_path) as source, pymupdf.open() as excerpt:
                if first_page < 1 or last_page > source.page_count or first_page > last_page:
                    raise ValueError(
                        f"Invalid page range {first_page}-{last_page} for {source_name} "
                        f"({source.page_count} pages)"
                    )
                excerpt.insert_pdf(
                    source,
                    from_page=first_page - 1,
                    to_page=last_page - 1,
                )
                excerpt.save(output_path)

        with pymupdf.open(output_path) as prepared:
            output_pages.append((output_path.name, prepared.page_count))

    total_pages = sum(page_count for _, page_count in output_pages)
    for filename, page_count in output_pages:
        print(f"{filename}: {page_count} pages")
    print(f"Total: {total_pages} pages")

    if total_pages > MAX_TOTAL_PAGES:
        raise SystemExit(
            f"Page limit exceeded: {total_pages} pages (maximum {MAX_TOTAL_PAGES})"
        )
    print(f"Page limit check: OK (maximum {MAX_TOTAL_PAGES})")


if __name__ == "__main__":
    prepare_docs()