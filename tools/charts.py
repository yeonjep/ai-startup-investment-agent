# 보고서용 차트 (matplotlib). 점수·가중치는 State/config 값만 사용하고 새 값을 만들지 않는다.

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

from agents.config import INVESTMENT_SCORE_THRESHOLD, SCORE_WEIGHTS  # noqa: E402

FONT_PATH = Path(__file__).resolve().parents[1] / "assets" / "fonts" / "NanumGothic-Regular.ttf"
GROUPS = {
    "team": ("창업자/팀", ["team_experience", "technical_headcount"]),
    "market": ("시장성", ["market_tam", "market_cagr", "demand_clarity"]),
    "product_technology": ("제품/기술력", ["development_stage", "technical_originality"]),
    "competitive_advantage": ("경쟁 우위", ["competitive_difference", "entry_barrier"]),
    "traction": ("실적", ["customer_traction", "revenue_stage"]),
    "funding_risk": ("자금조달/리스크", ["funding_total", "risk_mitigation"]),
}
COLORS = {"투자": "#2f6f4f", "보류": "#9a9a9a", "base": "#dcdcdc", "gain": "#3b6ea5", "neutral": "#e0b040"}


def _setup_font() -> None:
    if FONT_PATH.is_file():
        font_manager.fontManager.addfont(str(FONT_PATH))
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=str(FONT_PATH)).get_name()
    plt.rcParams["axes.unicode_minus"] = False


def group_score_chart(record: dict[str, Any], path: Path) -> Path:
    """항목별 획득 점수 / 가중치(만점). 결측 대입 3점이 포함된 몫은 노란색으로 구분한다."""
    _setup_font()
    scores = record["scores"]
    fig, ax = plt.subplots(figsize=(6.4, 2.9), dpi=200)
    for row, (key, (label, metrics)) in enumerate(GROUPS.items()):
        weight = SCORE_WEIGHTS[key]
        values = [scores[m]["score"] for m in metrics]
        real = sum(s for m, s in zip(metrics, values) if scores[m]["status"] != "missing")
        neutral = sum(s for m, s in zip(metrics, values) if scores[m]["status"] == "missing")
        share = weight / 5 / len(metrics)
        ax.barh(row, weight, color=COLORS["base"], height=0.6)
        ax.barh(row, real * share, color=COLORS["gain"], height=0.6)
        ax.barh(row, neutral * share, left=real * share, color=COLORS["neutral"], height=0.6)
        ax.text(weight + 0.6, row, f"{(real + neutral) * share:.1f} / {weight}", va="center", fontsize=8)
    ax.set_yticks(range(len(GROUPS)), [label for label, _ in GROUPS.values()], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 32)
    ax.set_xlabel("획득 점수 (회색=만점, 파랑=근거 있는 점수, 노랑=중립 대입·미확인)", fontsize=8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title(f"항목별 점수 구성 (총점 {record['total_score']:.2f} / 100)", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def total_score_chart(records: list[dict[str, Any]], path: Path) -> Path:
    """후보별 총점과 투자 기준선(config 임계값)."""
    _setup_font()
    names = [r["company_profile"].get("name", "기업") for r in records]
    totals = [r.get("total_score") or 0 for r in records]
    fig, ax = plt.subplots(figsize=(6.4, 2.7), dpi=200)
    bars = ax.bar(range(len(records)), totals, color=[COLORS.get(r["decision"], "#9a9a9a") for r in records], width=0.5)
    ax.axhline(INVESTMENT_SCORE_THRESHOLD, color="#b03a2e", linestyle="--", linewidth=1)
    ax.text(len(records) - 0.5, INVESTMENT_SCORE_THRESHOLD + 1.5, f"투자 기준 {INVESTMENT_SCORE_THRESHOLD}점",
            ha="right", fontsize=8, color="#b03a2e")
    for bar, record, total in zip(bars, records, totals):
        ax.text(bar.get_x() + bar.get_width() / 2, total + 1.5,
                f"{total:.1f}\n결측 {len(record.get('missing_evidence', []))}", ha="center", fontsize=8)
    ax.set_xticks(range(len(records)), [n if len(n) <= 12 else n[:11] + "…" for n in names], fontsize=9)
    ax.set_ylim(0, 110)
    ax.set_ylabel("총점", fontsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_title("후보별 총점 비교", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path
