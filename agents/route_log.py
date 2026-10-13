# Routing decisions are appended to a JSONL file, not to graph state.

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def append_route_log(trace_id: str | None, node: str, decision: str, reason: str) -> None:
    """Write one routing decision. Skip when the run has no trace id."""
    if not trace_id:
        return
    path = PROJECT_ROOT / "outputs" / "logs" / f"{trace_id}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "trace_id": trace_id,
        "node": node,
        "decision": decision,
        "reason": reason,
        "ts": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
