"""HelpMate multi-agent system (LangGraph): coordinator + workers + critic.

Agency lives in the COORDINATOR: it runs a ReAct-style tool-calling loop and
chooses which tools to invoke and when to stop — tool order is never hard-coded.
A DRAFTING worker then composes the RCA note from gathered context, and a CRITIC
verifies grounding, sending the draft back for revision when a claim is uncited.
"""
from __future__ import annotations

import json
from typing import Annotated, Any, Optional, TypedDict

from langchain_core.messages import (
    AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage,
)
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

from helpmate import config, prompts
from helpmate.guardrails import audit, scan_for_injection, wrap_untrusted
from helpmate.llm import get_llm, with_retries
from helpmate.memory import LongTermMemory, ShortTermMemory
from helpmate.rca import RCANote, check_grounding
from helpmate.tools import ALL_TOOLS, TOOLS_BY_NAME
from helpmate.trajectory import Trajectory

MAX_REVISIONS = 2


# ── Structured outputs for the worker stages ──────────────────────────────
class Disposition(BaseModel):
    disposition: str = Field(description="resolution_note | escalate | route_out_of_scope | needs_clarification")
    rationale: str
    routing_target: Optional[str] = Field(default=None, description="Team to route to if out of scope")
    suggested_problem_ticket: Optional[str] = Field(default=None)
    caveat: Optional[str] = Field(default=None, description="Conflict/missing-data caveat if any")
    injection_noted: bool = Field(default=False)


# ── Graph state ───────────────────────────────────────────────────────────
class State(TypedDict):
    ticket: dict
    messages: Annotated[list[BaseMessage], add_messages]
    trajectory: Trajectory
    stm: ShortTermMemory
    injection: dict
    disposition: Optional[Disposition]
    note: Optional[RCANote]
    critic: Optional[dict]
    revisions: int
    tool_calls_made: int


def _msg_text(msg: BaseMessage) -> str:
    """Flatten an AIMessage's content to plain text.

    Gemini 2.5 returns content as a list of blocks (text + a thinking
    signature); we keep only the human-readable text.
    """
    c = msg.content
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        parts = []
        for b in c:
            if isinstance(b, dict) and b.get("text"):
                parts.append(b["text"])
            elif isinstance(b, str):
                parts.append(b)
        return " ".join(parts)
    return str(c)


def _ticket_brief(t: dict) -> str:
    fields = ["ticket_id", "category", "affected_system", "priority", "department",
              "requester_id", "status", "subject"]
    head = "\n".join(f"{k}: {t.get(k, '')}" for k in fields if t.get(k) is not None)
    desc = wrap_untrusted(t.get("description", ""))
    return f"{head}\ndescription:\n{desc}"


# ── Nodes ─────────────────────────────────────────────────────────────────
def intake(state: State) -> dict:
    t = state["ticket"]
    traj = Trajectory(ticket_id=t.get("ticket_id"))
    stm = ShortTermMemory()
    inj = scan_for_injection(f"{t.get('subject','')} {t.get('description','')}")

    traj.thought(f"Intake ticket {t.get('ticket_id')} — {t.get('priority')} "
                 f"{t.get('category')} / {t.get('affected_system')}.")
    if inj["detected"]:
        traj.decision("Prompt-injection markers detected in ticket text; treating "
                      "description strictly as untrusted data and ignoring embedded "
                      "instructions.", patterns=inj["patterns"])
        audit("injection_detected", {"ticket_id": t.get("ticket_id"), "patterns": inj["patterns"]})

    # Long-term recall to bias the coordinator on recurring problems.
    recalled = LongTermMemory().recall(t.get("category"), t.get("subject", ""))
    recall_hint = ""
    if recalled:
        traj.thought(f"Long-term memory recalls {len(recalled)} prior similar case(s).")
        recall_hint = "\n\nPrior cases recalled from memory: " + json.dumps(
            [{"ticket_id": r["ticket_id"], "decision": r["decision"]} for r in recalled])

    messages = [
        SystemMessage(content=prompts.COORDINATOR_SYSTEM),
        HumanMessage(content=f"Triage this incoming ticket:\n\n{_ticket_brief(t)}{recall_hint}"),
    ]
    return {"trajectory": traj, "stm": stm, "injection": inj, "messages": messages,
            "revisions": 0, "tool_calls_made": 0}


def coordinator(state: State) -> dict:
    """ReAct step: the LLM decides whether to call tools or stop."""
    llm = with_retries(get_llm().bind_tools(ALL_TOOLS))
    ai: AIMessage = llm.invoke(state["messages"])
    traj = state["trajectory"]
    if ai.tool_calls:
        for tc in ai.tool_calls:
            traj.tool_call(tc["name"], tc["args"])
    else:
        traj.decision(f"Coordinator finished gathering context: {_msg_text(ai)[:400]}")
    return {"messages": [ai]}


def run_tools(state: State) -> dict:
    """Execute the coordinator's tool calls; feed retrieval into memory."""
    traj, stm = state["trajectory"], state["stm"]
    last: AIMessage = state["messages"][-1]
    out_msgs: list[BaseMessage] = []
    n = state["tool_calls_made"]

    for tc in last.tool_calls:
        name, args, call_id = tc["name"], tc["args"], tc["id"]
        n += 1
        tool = TOOLS_BY_NAME.get(name)
        if tool is None:
            result = f"Unknown tool {name}."
        else:
            result = tool.invoke(args)

        # Parse retrieval results into short-term memory + recognise cold hits.
        if name in ("similar_ticket_search", "kb_search"):
            try:
                payload = json.loads(result)
                hits = payload.get("hits", [])
            except Exception:
                hits = []
            if hits:
                stm.add_retrieved(hits)
                ids = [h.get("doc_id") for h in hits]
                traj.observation(f"{name} returned {len(hits)} hit(s): {ids}")
            else:
                traj.observation(f"{name} returned NO relevant hits (cold retrieval) "
                                 f"for query={args.get('query')!r}.")
        else:
            traj.observation(f"{name} -> {result[:200]}")

        out_msgs.append(ToolMessage(content=str(result), tool_call_id=call_id, name=name))

    return {"messages": out_msgs, "tool_calls_made": n}


def route_after_coordinator(state: State) -> str:
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        if state["tool_calls_made"] >= config.MAX_TOOL_CALLS:
            state["trajectory"].decision("Tool-call budget reached; proceeding to compose.")
            return "compose"
        return "tools"
    return "compose"


def compose(state: State) -> dict:
    """Decide disposition, then (if warranted) draft a grounded RCA note."""
    traj, stm, t = state["trajectory"], state["stm"], state["ticket"]
    context = stm.context_block()

    # Stage A — disposition (the branch comes from the model's reasoning).
    disp_llm = with_retries(get_llm().with_structured_output(Disposition))
    disp: Disposition = disp_llm.invoke([
        SystemMessage(content=prompts.DISPOSITION_SYSTEM),
        HumanMessage(content=f"Ticket:\n{_ticket_brief(t)}\n\nGathered context:\n{context}\n\n"
                             f"Injection markers present: {state['injection']['detected']}."),
    ])
    traj.decision(f"Disposition = {disp.disposition}. {disp.rationale}",
                  routing_target=disp.routing_target,
                  suggested_problem_ticket=disp.suggested_problem_ticket)

    feedback = state.get("critic", {}) or {}
    if disp.disposition in ("route_out_of_scope", "needs_clarification"):
        # No RCA forced — produce a short routing/clarification note instead.
        return {"disposition": disp, "note": None}

    # Stage B — draft the grounded RCA note (diagnosis/drafting worker).
    revision_hint = ""
    if feedback.get("problems"):
        revision_hint = ("\n\nThe previous draft FAILED critic review. Fix these: "
                         + "; ".join(feedback["problems"])
                         + " Cite only ids that appear in the context.")
        traj.thought("Revising draft per critic feedback.")

    draft_llm = with_retries(get_llm().with_structured_output(RCANote))
    note: RCANote = draft_llm.invoke([
        SystemMessage(content=prompts.DRAFT_SYSTEM),
        HumanMessage(content=f"Ticket:\n{_ticket_brief(t)}\n\nRetrieved context (cite ONLY these ids):\n"
                             f"{context}\n\nDisposition: {disp.disposition}. "
                             f"Caveat to include if any: {disp.caveat}.{revision_hint}"),
    ])
    traj.thought("Draft RCA note composed.", references=note.references)
    return {"disposition": disp, "note": note}


def critic(state: State) -> dict:
    """Verify grounding; flag uncited claims for revision."""
    traj, stm = state["trajectory"], state["stm"]
    note = state["note"]
    if note is None:  # out-of-scope / clarification — nothing to ground.
        traj.critic("No RCA note to verify (non-incident disposition).")
        return {"critic": {"grounded": True, "problems": []}}

    verdict = check_grounding(note, stm.available_ids())
    if verdict["grounded"]:
        traj.critic(f"PASS — every grounded claim cites a retrieved source: {verdict['cited_ids']}")
    else:
        traj.critic(f"FAIL — {verdict['problems']}")
    return {"critic": verdict}


def route_after_critic(state: State) -> str:
    verdict = state["critic"] or {}
    if verdict.get("grounded"):
        return "finalize"
    if state["revisions"] >= MAX_REVISIONS:
        state["trajectory"].decision("Max revisions reached; escalating ungrounded "
                                     "draft to a human reviewer (HITL).")
        return "finalize"
    return "revise"


def bump_revision(state: State) -> dict:
    return {"revisions": state["revisions"] + 1}


def finalize(state: State) -> dict:
    traj, t, disp = state["trajectory"], state["ticket"], state["disposition"]
    note, verdict = state.get("note"), state.get("critic") or {}

    # Record the decision in long-term memory for future recall.
    sources = sorted(state["stm"].available_ids())
    decision = disp.disposition if disp else "unknown"
    if not verdict.get("grounded") and note is not None:
        decision += " (HITL: ungrounded — needs human review)"
    LongTermMemory().record_case(
        ticket_id=t.get("ticket_id", "?"), category=t.get("category", ""),
        subject=t.get("subject", ""), decision=decision, sources=sources,
    )
    traj.decision(f"Finalised. Disposition={decision}; grounded={verdict.get('grounded')}; "
                  f"sources={sources}")
    return {}


# ── Build graph ───────────────────────────────────────────────────────────
def build_graph():
    g = StateGraph(State)
    g.add_node("intake", intake)
    g.add_node("coordinator", coordinator)
    g.add_node("tools", run_tools)
    g.add_node("compose", compose)
    g.add_node("critic", critic)
    g.add_node("revise", bump_revision)
    g.add_node("finalize", finalize)

    g.set_entry_point("intake")
    g.add_edge("intake", "coordinator")
    g.add_conditional_edges("coordinator", route_after_coordinator,
                            {"tools": "tools", "compose": "compose"})
    g.add_edge("tools", "coordinator")
    g.add_edge("compose", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"revise": "revise", "finalize": "finalize"})
    g.add_edge("revise", "compose")
    g.add_edge("finalize", END)
    return g.compile()


_GRAPH = None


def get_graph():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    return _GRAPH


def run_ticket(ticket: dict) -> dict:
    """Run the full agent on one ticket; return a structured result."""
    final = get_graph().invoke(
        {"ticket": ticket}, config={"recursion_limit": 50}
    )
    note: Optional[RCANote] = final.get("note")
    disp: Optional[Disposition] = final.get("disposition")
    return {
        "ticket": ticket,
        "disposition": disp.model_dump() if disp else None,
        "note_markdown": note.to_markdown() if note else None,
        "note": note.model_dump() if note else None,
        "critic": final.get("critic"),
        "injection": final.get("injection"),
        "trajectory": final["trajectory"],
        "tool_calls": final["trajectory"].tool_calls,
    }
