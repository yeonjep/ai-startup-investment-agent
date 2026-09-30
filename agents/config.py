# Runtime configuration aligned with docs/DESIGN.md B-5, C-3, D-1, and F.

import os
from datetime import date


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gpt-4.1-mini")
AS_OF_DATE = date.today().isoformat()
MAX_CANDIDATE_POOL = 10
MAX_CANDIDATES = 5
MAX_SELECTION_RETRIES = 1
MAX_RAG_RETRIES = 2
MAX_EVIDENCE_RETRIES = 1

SCORE_WEIGHTS = {
	"team": 25,
	"market": 20,
	"product_technology": 20,
	"competitive_advantage": 15,
	"traction": 10,
	"funding_risk": 10,
}
INVESTMENT_SCORE_THRESHOLD = 70
MAX_MISSING_EVIDENCE_FOR_INVESTMENT = 5

USD_KRW_FIXED_RATE = 1356.7
USD_KRW_RATE_SOURCE = (
	"서울외환시장 2026-09-29 주간거래 종가(15:30), 머니투데이·아시아경제 보도"
)
USD_KRW_RATE_BASE_DATE = "2026-09-29"


def get_as_of_date() -> str:
	value = os.getenv("AS_OF_DATE", AS_OF_DATE)
	try:
		return date.fromisoformat(value).isoformat()
	except ValueError as error:
		raise ValueError("AS_OF_DATE must use YYYY-MM-DD format") from error


def validate_max_candidates(value: object = None) -> int:
	raw_value = os.getenv("MAX_CANDIDATES", str(MAX_CANDIDATES)) if value is None else value
	try:
		value = int(raw_value)
	except ValueError as error:
		raise ValueError("MAX_CANDIDATES must be an integer from 1 to 10") from error
	if not 1 <= value <= MAX_CANDIDATE_POOL:
		raise ValueError("MAX_CANDIDATES must be between 1 and 10")
	return value