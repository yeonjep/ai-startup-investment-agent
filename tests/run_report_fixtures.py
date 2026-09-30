# Report Agent를 가짜 State 3종으로 실행해 PDF를 outputs/fixtures/에 만든다.
# 실행: uv run python -m tests.run_report_fixtures

from pathlib import Path

from agents.report import REPORT_BASENAME, generate_report
from tests.fixtures.report_fixtures import FIXTURES, load_fixture

OUT = Path(__file__).resolve().parents[1] / "outputs" / "fixtures"

if __name__ == "__main__":
    for name in FIXTURES:
        result = generate_report(load_fixture(name), OUT / name, REPORT_BASENAME)
        print(f"{name}: type={result['type']} pages={result['pages']} pdf={result['pdf_path']}")
