# LLM과 런타임 설정, 프롬프트 로더.

import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from agents.config import LLM_MODEL


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROMPTS_DIR = PROJECT_ROOT / "prompts"


def configure_runtime() -> None:
    load_dotenv(PROJECT_ROOT / ".env")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower()
    if tracing_enabled in {"true", "1", "yes"}:
        os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
        os.environ.setdefault("LANGCHAIN_PROJECT", "SKALA")


def create_llm(model_name: str | None = None, **kwargs: object) -> ChatOpenAI:
    configure_runtime()
    selected_model = model_name or os.getenv("LLM_MODEL") or LLM_MODEL
    if not selected_model:
        raise ValueError("Set LLM_MODEL in agents/config.py or .env after model selection.")
    return ChatOpenAI(model=selected_model, **kwargs)


def load_prompt(prompt_name: str) -> str:
    requested_path = Path(prompt_name)
    if requested_path.is_absolute():
        raise ValueError("Prompt name must be relative to the prompts/ directory.")

    candidates = [requested_path]
    if not requested_path.suffix:
        candidates.extend(
            requested_path.with_suffix(suffix) for suffix in (".md", ".txt", ".prompt")
        )

    for candidate in candidates:
        prompt_path = (PROMPTS_DIR / candidate).resolve()
        if not prompt_path.is_relative_to(PROMPTS_DIR):
            raise ValueError("Prompt path must stay inside the prompts/ directory.")
        if prompt_path.is_file():
            return prompt_path.read_text(encoding="utf-8")

    raise FileNotFoundError(f"Prompt not found in {PROMPTS_DIR}: {prompt_name}")