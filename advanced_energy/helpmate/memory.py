"""Agent memory.

* Short-term: a per-ticket scratchpad the agent uses within a single run to
  accumulate retrieved context and notes (lives in the LangGraph state).
* Long-term: a small persisted store of resolved cases / decisions that lets
  the agent *recall* how similar problems were handled before, biasing tool
  choice on recurring clusters.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from helpmate import config

_LTM_PATH = config.DATA_DIR / "long_term_memory.json"


class ShortTermMemory:
    """Per-ticket scratchpad. Collects retrieved sources and free notes."""

    def __init__(self):
        self.notes: list[str] = []
        self.retrieved: list[dict] = []  # accumulated retrieval hits

    def note(self, text: str) -> None:
        self.notes.append(text)

    def add_retrieved(self, hits: list[dict]) -> None:
        seen = {(h.get("doc_id"), h.get("content")) for h in self.retrieved}
        for h in hits:
            if (h.get("doc_id"), h.get("content")) not in seen:
                self.retrieved.append(h)

    def available_ids(self) -> set[str]:
        """All source ids the agent has actually seen (for grounding checks)."""
        ids: set[str] = set()
        for h in self.retrieved:
            for key in ("doc_id", "ticket_id", "case_id"):
                v = h.get(key)
                if v:
                    ids.add(v)
        return ids

    def context_block(self) -> str:
        if not self.retrieved:
            return "(no sources retrieved yet)"
        blocks = []
        for h in self.retrieved:
            ident = h.get("doc_id") or h.get("ticket_id")
            extra = f" ticket={h.get('ticket_id')}" if h.get("ticket_id") else ""
            blocks.append(f"[{ident}{extra} score={h.get('score')}]\n{h.get('content')}")
        return "\n\n".join(blocks)


class LongTermMemory:
    """Persisted recall of prior cases/decisions across runs."""

    def __init__(self):
        self._records: list[dict] = []
        if _LTM_PATH.exists():
            try:
                self._records = json.loads(_LTM_PATH.read_text(encoding="utf-8"))
            except Exception:
                self._records = []

    def recall(self, category: str | None, query: str, k: int = 3) -> list[dict]:
        """Lightweight recall: prefer same-category prior decisions, then
        keyword overlap with the query."""
        q_tokens = set((query or "").lower().split())
        scored = []
        for r in self._records:
            score = 0.0
            if category and r.get("category") == category:
                score += 1.0
            overlap = q_tokens & set(r.get("subject", "").lower().split())
            score += 0.1 * len(overlap)
            if score > 0:
                scored.append((score, r))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [r for _, r in scored[:k]]

    def record_case(self, ticket_id: str, category: str, subject: str,
                    decision: str, sources: list[str]) -> None:
        self._records.append({
            "ticket_id": ticket_id,
            "category": category,
            "subject": subject,
            "decision": decision,
            "sources": sources,
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        _LTM_PATH.write_text(json.dumps(self._records, indent=2), encoding="utf-8")
