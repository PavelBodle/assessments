"""The generated artifact — a structured Resolution / RCA note.

Defines the 7-section schema, renders it to markdown, and provides the
grounding check the critic uses: every section that draws on prior knowledge
must cite at least one retrieved source id (ticket_id / CASE / RES / KB).
"""
from __future__ import annotations

import re
from typing import ClassVar

from pydantic import BaseModel, Field

# Matches the ids the agent is allowed to cite as evidence.
CITATION_RE = re.compile(r"\b(?:TCK-\d{5}|HLD-[A-Z0-9]+|CASE-\d{5}|RES-\d{3}|KB-\d{3})\b")


class RCANote(BaseModel):
    """Structured resolution / root-cause note (matches the assignment skeleton)."""

    summary: str = Field(description="One-paragraph problem statement grounded in the ticket.")
    affected_scope: str = Field(description="Impacted user(s)/department, system/asset, related/duplicate tickets.")
    diagnosis_root_cause: str = Field(description="Most likely root cause with supporting evidence and reasoning.")
    resolution: str = Field(description="Concrete fix steps drawn from prior resolutions and KB articles.")
    verification: str = Field(description="How to confirm the issue is actually resolved.")
    preventive_action: str = Field(description="Follow-up to prevent recurrence; propose KB/change where warranted.")
    references: list[str] = Field(default_factory=list, description="Specific retrieved source ids relied upon.")

    # Sections that must be grounded in a retrieved source when prior knowledge
    # is used. Summary/Verification may be ticket-derived and are not required
    # to cite, but Diagnosis and Resolution must trace to evidence.
    GROUNDED_FIELDS: ClassVar[tuple[str, ...]] = ("diagnosis_root_cause", "resolution")

    def cited_ids(self) -> set[str]:
        ids = set()
        for f in self.GROUNDED_FIELDS:
            ids |= set(CITATION_RE.findall(getattr(self, f)))
        ids |= set(self.references)
        return ids

    def to_markdown(self) -> str:
        refs = "\n".join(f"- {r}" for r in self.references) or "- (none cited)"
        return f"""# Resolution / RCA Note

## Summary
{self.summary}

## Affected Scope & Environment
{self.affected_scope}

## Diagnosis & Root Cause
{self.diagnosis_root_cause}

## Resolution / Recommended Fix
{self.resolution}

## Verification
{self.verification}

## Preventive Action / KB Update
{self.preventive_action}

## References
{refs}
"""


def check_grounding(note: RCANote, available_ids: set[str]) -> dict:
    """Critic's core check.

    Returns a verdict describing whether each grounded section cites at least
    one *retrieved* source id (i.e. an id that actually appears in the context
    the agent was given — not invented).
    """
    problems: list[str] = []
    for f in RCANote.GROUNDED_FIELDS:
        text = getattr(note, f)
        cites = set(CITATION_RE.findall(text)) | set(note.references)
        grounded = cites & available_ids
        if not grounded:
            problems.append(
                f"Section '{f}' makes claims without citing any retrieved source."
            )

    # Hallucinated citation: an id cited that was never retrieved.
    invented = note.cited_ids() - available_ids
    if invented:
        problems.append(
            f"Cites source id(s) not present in retrieved context: {sorted(invented)}."
        )

    return {
        "grounded": not problems,
        "problems": problems,
        "cited_ids": sorted(note.cited_ids()),
        "available_ids": sorted(available_ids),
    }
