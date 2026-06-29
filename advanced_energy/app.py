"""HelpMate — Streamlit demo.

Pick a held-out ticket (or edit your own), run the multi-agent system, and watch
the coordinator choose tools, the critic verify grounding, and the final cited
RCA note — with the full timestamped trajectory.

Run locally:  streamlit run app.py
"""
from __future__ import annotations

import streamlit as st

from helpmate import config
from helpmate.datasets import all_heldout_tickets

st.set_page_config(page_title="HelpMate — Agentic IT Support", page_icon="🛠️", layout="wide")

# Friendly metadata for each held-out scenario.
SCENARIO_META = {
    "normal":             ("✅", "Normal ticket", "A routine incident that should retrieve similar past cases."),
    "cold_retrieval":     ("❄️", "Cold / failed retrieval", "First search returns nothing — agent must reformulate or switch tools."),
    "contradictory_data": ("⚠️", "Missing / contradictory data", "Category conflicts with the description — agent must flag it, not fabricate."),
    "out_of_scope":       ("🚪", "Out-of-scope request", "Not an IT incident — agent should route it, not force an RCA."),
    "priority_branching": ("🚨", "P1 priority branching", "Mass outage — agent should escalate, notify and request human approval."),
    "prompt_injection":   ("🛡️", "Adversarial (prompt injection)", "Description hides an injected command — agent must resist it."),
}

ICONS = {"thought": "🧠", "tool_call": "🔧", "observation": "👁",
         "decision": "✅", "critic": "🔍"}


def scenario_label(t: dict) -> str:
    emoji, name, _ = SCENARIO_META.get(t.get("scenario", "normal"), ("✅", "Normal ticket", ""))
    return f"{emoji}  {t['ticket_id']} · {name} — {t['subject'][:55]}"


# ── Header ────────────────────────────────────────────────────────────────
st.title("🛠️ HelpMate — Agentic IT Support Ticket Resolution")
st.caption("Coordinator → Retrieval / Drafting workers → Critic · grounded RCA notes · full decision trajectory")

# ── Sidebar: status + quick steps ─────────────────────────────────────────
with st.sidebar:
    st.subheader("Status")
    if config.GEMINI_API_KEY:
        st.success("API key loaded ✓")
    else:
        st.error("GEMINI_API_KEY not set — add it to .env or Streamlit secrets.")
    st.markdown(f"**Model:** `{config.LLM_MODEL}`")
    st.markdown(f"**Embeddings:** `{config.EMBEDDING_MODEL}`")
    st.divider()
    st.markdown(
        "**Quick steps**\n"
        "1. Pick a ticket below\n"
        "2. (Optional) edit its fields\n"
        "3. Click **Run HelpMate**\n"
        "4. Read the tabs: trajectory → RCA → critic"
    )
    st.caption("See the **Guide** and **Configuration** tabs for details.")

tab_run, tab_config, tab_guide = st.tabs(["▶️  Run agent", "⚙️  Configuration", "📖  Guide"])

# ══════════════════════════════════════════════════════════════════════════
# RUN TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_run:
    tickets = all_heldout_tickets()
    labels = {scenario_label(t): t for t in tickets}

    choice = st.selectbox("**1 · Choose a held-out ticket**", list(labels.keys()),
                          help="These tickets are never indexed, so retrieval is tested honestly.")
    ticket = dict(labels[choice])

    meta = SCENARIO_META.get(ticket.get("scenario", "normal"))
    if meta and ticket.get("scenario"):
        st.info(f"{meta[0]} **{meta[1]}** — {meta[2]}")

    with st.expander("**2 · Review / edit the ticket**", expanded=True):
        c1, c2, c3 = st.columns(3)
        ticket["priority"] = c1.text_input("Priority", ticket.get("priority", "P3"))
        ticket["category"] = c2.text_input("Category", ticket.get("category", ""))
        ticket["affected_system"] = c3.text_input("Affected system", ticket.get("affected_system", ""))
        ticket["subject"] = st.text_input("Subject", ticket.get("subject", ""))
        ticket["description"] = st.text_area("Description", ticket.get("description", ""), height=110)

    run = st.button("▶  Run HelpMate", type="primary", use_container_width=True,
                    disabled=not config.GEMINI_API_KEY)

    if run:
        from helpmate.graph import run_ticket
        with st.spinner("Coordinator reasoning → retrieving → drafting → critic verifying…"):
            st.session_state["result"] = run_ticket(ticket)

    result = st.session_state.get("result")
    if not result:
        st.caption("Run a ticket to see the agent's trajectory, RCA note, and critic verdict.")
    else:
        disp = result.get("disposition") or {}
        critic = result.get("critic") or {}
        inj = result.get("injection") or {}

        st.divider()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Disposition", (disp.get("disposition") or "—").replace("_", " "))
        m2.metric("Tool calls", len(result["tool_calls"]))
        m3.metric("Grounded", "✅ yes" if critic.get("grounded") else "⚠️ no")
        m4.metric("Injection", "🛡️ detected" if inj.get("detected") else "none")

        rtabs = st.tabs(["🧭 Trajectory", "📄 RCA Note", "🔍 Critic", "📚 Retrieved context", "🧱 Raw"])

        with rtabs[0]:
            st.caption("Every thought, tool call, observation and decision — timestamped.")
            for s in result["trajectory"].to_list():
                ts = s["ts"].split("T")[1][:12] if "T" in s["ts"] else s["ts"]
                st.markdown(f"{ICONS.get(s['kind'], '•')} **{s['kind']}** · `{ts}` — {s['content']}")
                if s["detail"].get("args"):
                    st.code(s["detail"]["args"], language="json")

        with rtabs[1]:
            if result.get("note_markdown"):
                st.markdown(result["note_markdown"])
                st.download_button("⬇ Download RCA note (.md)", result["note_markdown"],
                                   file_name=f"{ticket['ticket_id']}_rca.md")
            else:
                st.warning(f"No RCA note generated — disposition: **{(disp.get('disposition') or '').replace('_',' ')}**.")
                if disp.get("routing_target"):
                    st.markdown(f"**Routed to:** {disp['routing_target']}")
                if disp.get("rationale"):
                    st.markdown(disp["rationale"])

        with rtabs[2]:
            if critic.get("grounded"):
                st.success("PASS — every grounded claim cites a retrieved source.")
            else:
                st.error("FAIL / HITL — ungrounded claims flagged:")
                for p in critic.get("problems", []):
                    st.markdown(f"- {p}")
            cc1, cc2 = st.columns(2)
            cc1.markdown("**Cited ids**")
            cc1.write(critic.get("cited_ids") or "—")
            cc2.markdown("**Available (retrieved) ids**")
            cc2.write(critic.get("available_ids") or "—")
            if disp.get("caveat"):
                st.info(f"Caveat noted: {disp['caveat']}")

        with rtabs[3]:
            st.markdown("**Tools the coordinator chose (in order):**")
            st.code(" → ".join(result["tool_calls"]) or "(none)", language="text")
            st.markdown("**Observations returned to the agent:**")
            for h in result["trajectory"].to_list():
                if h["kind"] == "observation":
                    st.text(h["content"])

        with rtabs[4]:
            st.json({"disposition": disp, "critic": critic, "injection": inj,
                     "tool_calls": result["tool_calls"]})

# ══════════════════════════════════════════════════════════════════════════
# CONFIGURATION TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_config:
    st.subheader("Active configuration")
    st.caption("Every model choice is read once from `.env` (or Streamlit secrets). "
               "Switching models is a one-line change — no code edits.")

    rows = {
        "LLM provider": config.LLM_PROVIDER,
        "LLM model": config.LLM_MODEL,
        "LLM temperature": config.LLM_TEMPERATURE,
        "Embedding model": config.EMBEDDING_MODEL,
        "Max tool calls / ticket": config.MAX_TOOL_CALLS,
        "Retrieval min score": config.RETRIEVAL_MIN_SCORE,
        "API key": "loaded ✓" if config.GEMINI_API_KEY else "missing ✗",
    }
    st.table({"Setting": list(rows.keys()), "Value": [str(v) for v in rows.values()]})

    st.subheader("How to change models")
    st.markdown(
        "Edit **`.env`** (local) or the **Secrets** dashboard (Streamlit Cloud), then rerun:"
    )
    st.code(
        "LLM_PROVIDER=gemini\n"
        "# Gemma has far higher free limits than Gemini Flash (5 RPM / 20 RPD)\n"
        "LLM_MODEL=gemma-4-26b-a4b-it\n"
        "# LLM_MODEL=gemini-2.5-flash\n"
        "LLM_TEMPERATURE=0.1\n"
        "EMBEDDING_MODEL=BAAI/bge-small-en-v1.5\n"
        "MAX_TOOL_CALLS=12\n"
        "GEMINI_API_KEY=your-key-here",
        language="bash",
    )
    st.markdown(
        "- **LLM** is instantiated only in `helpmate/llm.py` (single factory).\n"
        "- **Embeddings** run locally via HuggingFace — no extra key needed.\n"
        "- The agent retries with backoff on rate-limit (429) errors automatically."
    )

# ══════════════════════════════════════════════════════════════════════════
# GUIDE TAB
# ══════════════════════════════════════════════════════════════════════════
with tab_guide:
    st.subheader("How to use HelpMate")
    st.markdown(
        """
**What it does** — HelpMate triages an IT support ticket end to end: it retrieves
similar past tickets and KB articles, drafts a structured **Resolution / RCA note**,
and a **critic** checks that every claim cites a real retrieved source before handoff.

**Using the app**
1. Open the **Run agent** tab and pick a held-out ticket. The 5 battery scenarios
   (❄️ cold retrieval, ⚠️ contradictory data, 🚪 out-of-scope, 🚨 P1, 🛡️ injection)
   are designed to show the agent *adapting* — not following a fixed script.
2. Optionally edit any field, or paste your own ticket text into **Description**.
3. Click **Run HelpMate** and wait ~25–80s (it makes several reasoning calls).
4. Explore the result tabs:
   - **🧭 Trajectory** — the agent's live decision path: which tools it chose, in
     what order (never hard-coded), and why.
   - **📄 RCA Note** — the generated artifact, with inline citations + download.
   - **🔍 Critic** — the grounding verdict (which ids were cited vs. retrieved).
   - **📚 Retrieved context** — the evidence fed to the draft.

**What to look for**
- The **coordinator picks tools dynamically** and stops on its own.
- On a **cold retrieval** it reformulates instead of inventing an answer.
- For a **P1 outage** it escalates + notifies + requests human approval (HITL).
- For an **out-of-scope** request it routes instead of forcing an RCA.
- A **prompt injection** in the description is detected and ignored.

**Tip** — the **Configuration** tab shows the active model and how to swap it.
        """
    )
    st.info("Prefer the terminal? Run `python -m helpmate.cli HLD-P1` or "
            "`python -m eval.run_eval` for the full scored battery.")

# ── Footer (always visible) ───────────────────────────────────────────────
st.divider()
st.markdown(
    """
    <div style="text-align:center; opacity:0.85; font-size:0.9rem;">
      Made by <b>Pavel Bodle</b> &nbsp;|&nbsp;
      <a href="https://www.linkedin.com/in/pavelbodle/" target="_blank">LinkedIn</a> &nbsp;|&nbsp;
      <a href="https://github.com/PavelBodle" target="_blank">GitHub</a>
    </div>
    """,
    unsafe_allow_html=True,
)
