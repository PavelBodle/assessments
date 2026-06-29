"""Tools available to the agent.

Deliberately broader than any single ticket needs, so the coordinator must
*choose* (and avoid unnecessary calls). Read tools surface ground-truth data
and retrieval; action tools are mocked but logged to the audit trail.

Retrieval tools return JSON so the orchestrator can both show the model
readable evidence and parse the hits into short-term memory for grounding.
"""
from __future__ import annotations

import csv
import json
from functools import lru_cache

from langchain_core.tools import tool

from helpmate import config, indexing
from helpmate.guardrails import audit


# ── Data loaders (cached) ─────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _tickets() -> dict[str, dict]:
    rows = {}
    if config.TICKETS_CSV.exists():
        with open(config.TICKETS_CSV, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                rows[r["ticket_id"]] = r
    return rows


@lru_cache(maxsize=1)
def _cmdb() -> dict[str, dict]:
    if config.CMDB_JSON.exists():
        return json.loads(config.CMDB_JSON.read_text(encoding="utf-8"))
    return {}


# ── Read tools ────────────────────────────────────────────────────────────
@tool
def ticket_lookup(ticket_id: str) -> str:
    """Look up a ticket's structured record by its ticket_id (e.g. TCK-00042).
    Use to confirm metadata such as category, priority, status, requester."""
    t = _tickets().get(ticket_id.strip())
    if not t:
        return f"No ticket found with id {ticket_id}."
    return json.dumps(t)


@tool
def similar_ticket_search(query: str) -> str:
    """Semantic search over prior resolved tickets / resolution notes. Returns
    the most similar past cases with their source ids and resolution text. Use
    to find how comparable problems were diagnosed and fixed. If it returns an
    empty list, the search was COLD — reformulate or broaden before drafting."""
    hits = indexing.search_similar_tickets(query, k=3)
    return json.dumps({"query": query, "hits": hits})


@tool
def kb_search(query: str) -> str:
    """Semantic search over the knowledge base of step-by-step fix articles.
    Returns matching KB articles with their KB ids. Use for known-issue
    procedures. An empty list means no relevant article was found."""
    hits = indexing.search_kb(query, k=3)
    return json.dumps({"query": query, "hits": hits})


@tool
def cmdb_lookup(requester_id: str) -> str:
    """Look up the requester's asset/CMDB record (device, OS, VPN client
    version, etc.) by requester_id (e.g. USR-00123). Use when the diagnosis
    depends on the user's hardware or installed versions."""
    rec = _cmdb().get(requester_id.strip())
    if not rec:
        return f"No CMDB record for {requester_id}."
    return json.dumps(rec)


# ── Action tools (mocked + audited) ───────────────────────────────────────
@tool
def create_notification(recipient: str, subject: str, message: str) -> str:
    """Send a notification (mock). Use to inform a user/team of status, an
    outage, or a required action."""
    audit("create_notification", {"recipient": recipient, "subject": subject, "message": message})
    return f"NOTIFICATION SENT to {recipient}: {subject}"


@tool
def update_ticket(ticket_id: str, status: str, note: str) -> str:
    """Update a ticket's status and add a work note (mock). Valid statuses:
    Open, In Progress, Resolved, Closed, Escalated."""
    audit("update_ticket", {"ticket_id": ticket_id, "status": status, "note": note})
    return f"TICKET {ticket_id} updated -> status={status}"


@tool
def create_escalation(ticket_id: str, reason: str, suggested_problem_ticket: str) -> str:
    """Escalate a ticket and suggest a problem/RCA record (mock). Use for P1 /
    mass-impact incidents or when first-line cannot resolve."""
    audit("create_escalation", {
        "ticket_id": ticket_id, "reason": reason,
        "suggested_problem_ticket": suggested_problem_ticket,
    })
    return f"ESCALATION created for {ticket_id}; suggested problem record: {suggested_problem_ticket}"


@tool
def request_human_approval(action: str, reason: str) -> str:
    """Request a human-in-the-loop approval checkpoint before a consequential
    action (mock; returns PENDING). Use for high-impact actions such as P1
    escalations or anything affecting many users."""
    audit("request_human_approval", {"action": action, "reason": reason})
    return (
        f"HUMAN APPROVAL REQUESTED for: {action} (reason: {reason}). "
        "Status: PENDING — awaiting on-call approver."
    )


READ_TOOLS = [ticket_lookup, similar_ticket_search, kb_search, cmdb_lookup]
ACTION_TOOLS = [create_notification, update_ticket, create_escalation, request_human_approval]
ALL_TOOLS = READ_TOOLS + ACTION_TOOLS
TOOLS_BY_NAME = {t.name: t for t in ALL_TOOLS}
