"""Agent trajectory logging — every thought, tool call, observation, decision.

The full trajectory is the auditable record of *why* the agent did what it did.
Each step is timestamped so trajectory efficiency can be evaluated.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone


@dataclass
class Step:
    kind: str            # thought | tool_call | observation | decision | critic
    content: str
    detail: dict = field(default_factory=dict)
    ts: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="milliseconds"))


class Trajectory:
    """Ordered, timestamped record of an agent run."""

    def __init__(self, ticket_id: str | None = None):
        self.ticket_id = ticket_id
        self.steps: list[Step] = []

    def add(self, kind: str, content: str, **detail) -> Step:
        step = Step(kind=kind, content=content, detail=detail)
        self.steps.append(step)
        return step

    # Convenience helpers ---------------------------------------------------
    def thought(self, content: str, **d):     return self.add("thought", content, **d)
    def tool_call(self, name: str, args: dict): return self.add("tool_call", name, args=args)
    def observation(self, content: str, **d): return self.add("observation", content, **d)
    def decision(self, content: str, **d):    return self.add("decision", content, **d)
    def critic(self, content: str, **d):      return self.add("critic", content, **d)

    @property
    def tool_calls(self) -> list[str]:
        return [s.content for s in self.steps if s.kind == "tool_call"]

    def to_list(self) -> list[dict]:
        return [asdict(s) for s in self.steps]

    def to_json(self) -> str:
        return json.dumps({"ticket_id": self.ticket_id, "steps": self.to_list()}, indent=2)

    def render(self) -> str:
        """Human-readable trace for the CLI / logs."""
        icons = {"thought": "🧠", "tool_call": "🔧", "observation": "👁", "decision": "✅", "critic": "🔍"}
        lines = []
        for s in self.steps:
            t = s.ts.split("T")[1] if "T" in s.ts else s.ts
            head = f"{icons.get(s.kind, '•')} [{t}] {s.kind.upper()}: {s.content}"
            lines.append(head)
            if s.detail.get("args"):
                lines.append(f"      args={json.dumps(s.detail['args'])}")
        return "\n".join(lines)
