from pathlib import Path
import shutil

import pymupdf


ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
DOCS_DIR = ROOT / "data" / "docs"
MAX_TOTAL_PAGES = 200

DOCUMENTS = (
    ("gov_2025_strategy", "251219+(별첨)+AI반도체+산업+도약+전략(안)+요약본.pdf", 10, None),
    ("deloitte_2026_outlook", "2026 Semiconductor Industry Outlook _ Deloitte Insights.pdf", 13, None),
    ("kisdi_2024_perspectives", "[초점] 새로운 기회의 창으로 AI반도체 시장 현황과 전망.pdf", 24, None),
    ("kisdi_2024_research", "AI반도체 글로벌 첨단 기술·산업 동향 조사 및 대응방향 연구.pdf", 55, (27, 81)),
)


def prepare_docs() -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    page_comparison = []

    for doc_id, source_name, designed_pages, page_range in DOCUMENTS:
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
            page_comparison.append((doc_id, designed_pages, prepared.page_count))

    designed_total = sum(designed for _, designed, _ in page_comparison)
    actual_total = sum(actual for _, _, actual in page_comparison)
    print(f"{'doc_id':<28} {'설계(쪽)':>8} {'실제(쪽)':>8} {'차이(쪽)':>8}")
    for doc_id, designed, actual in page_comparison:
        print(f"{doc_id:<28} {designed:>8} {actual:>8} {actual - designed:>+8}")
    print(f"{'합계':<28} {designed_total:>8} {actual_total:>8} {actual_total - designed_total:>+8}")

    if actual_total > MAX_TOTAL_PAGES:
        raise SystemExit(
            f"Page limit exceeded: {actual_total} pages (maximum {MAX_TOTAL_PAGES})"
        )
    print(f"Page limit check: OK (maximum {MAX_TOTAL_PAGES})")


if __name__ == "__main__":
    prepare_docs()