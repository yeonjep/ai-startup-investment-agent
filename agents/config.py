# Runtime configuration aligned with docs/DESIGN.md F.

import os


LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4.1-mini")
LLM_JUDGE_MODEL = os.getenv("LLM_JUDGE_MODEL", "gpt-4.1-mini")
MAX_CANDIDATE_POOL = 10
MAX_CANDIDATES = 5
MAX_SELECTION_RETRIES = 1
MAX_RAG_RETRIES = 2
MAX_EVIDENCE_RETRIES = 1