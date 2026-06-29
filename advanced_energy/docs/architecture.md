# HelpMate — Architecture & Agent Design

*(Part 2 design note. Production/ops detail is summarised here and expanded in Q&A.)*

## 1. Overview

HelpMate triages an incoming IT ticket, retrieves grounded evidence (prior
resolutions + KB), drafts a structured **Resolution/RCA note**, has a **critic**
verify every claim cites a source, takes mock downstream actions, and logs the
full trajectory. The system is **multi-agent** and **agentic**: the coordinator
decides which tools to use and when to stop — there is no fixed pipeline.

## 2. Retrieval & LLM layer

- **Embeddings:** `BAAI/bge-small-en-v1.5` (HuggingFace sentence-transformers,
  384-dim, normalized). Chosen for strong quality-to-size ratio so it runs in
  Streamlit Cloud's free RAM and needs no API key. Configurable via `.env`.
- **Chunking/indexing:** corpus markdown (resolution notes + KB articles) is
  split with `RecursiveCharacterTextSplitter` (700/120) on heading boundaries,
  embedded, and stored in **FAISS**. Two logical collections share one store,
  separated by a `type` metadata filter (`resolution_note` vs `kb_article`).
  Incoming/held-out tickets are **never indexed** → honest retrieval.
- **Retrieval flow:** cosine similarity with a relevance floor
  (`RETRIEVAL_MIN_SCORE`). Hits below the floor are dropped, so an empty result
  is a real signal the coordinator uses to detect **cold retrieval**.
- **LLM:** Gemini `gemini-2.0-flash` (free tier) via `langchain-google-genai`,
  temperature 0.1 for reproducibility. Provider/model are read once in
  `helpmate/config.py` and instantiated only in `helpmate/llm.py` — swapping
  models is a one-line `.env` change.
- **Grounding / anti-hallucination:** the draft worker is given *only* the
  retrieved context block and instructed to cite ids that appear in it; the
  critic then mechanically verifies citations against the set of actually
  retrieved ids and rejects invented ones.

## 3. Multi-agent design (coordinator–worker–critic)

| Agent | Role | Why it earns its place |
|---|---|---|
| **Coordinator** | ReAct tool-calling loop; plans, selects tools, recovers, decides when to stop. | This is where agency lives — dynamic tool choice and branching. |
| **Retrieval worker** | `similar_ticket_search`, `kb_search` (+ reformulation on cold hits). | Separates evidence-gathering; lets the coordinator iterate. |
| **Drafting worker** | Composes the structured RCA note from gathered context. | Specialised generation grounded only in retrieved sources. |
| **Critic/reviewer** | Verifies every grounded claim cites a retrieved id; triggers revision. | Independent check before human handoff; catches hallucinated citations. |

Multi-agent adds value at exactly two seams: **iterative retrieval** (recovering
from empty/contradictory results) and **independent grounding review**. We keep
it minimal — no agent-per-tool sprawl.

**Reasoning approach:** ReAct for the coordinator (think → act → observe → …),
plan-and-revise for draft↔critic. Decision logic:
- *Tool selection:* the LLM picks tools from descriptions; we expose more tools
  than any one ticket needs (CMDB, KB, similar-ticket, actions) so it must choose.
- *Order/repeat:* not hard-coded; bounded by `MAX_TOOL_CALLS` to prevent thrash.
- *Stopping:* the coordinator stops emitting tool calls when it has enough
  grounded context (or determined out-of-scope/escalation).
- *Recovery:* empty retrieval → reformulate/broaden/switch tool; contradictory
  data → diagnose the real issue and attach an explicit caveat.

**HITL vs autonomous:** the coordinator requests `request_human_approval` for
high-impact actions (P1/mass-impact), and the critic forces HITL when a draft
stays ungrounded after `MAX_REVISIONS`. Routine P4 how-tos proceed autonomously.

## 4. Tools

`ticket_lookup`, `similar_ticket_search` (vector), `kb_search` (vector),
`cmdb_lookup` (asset/CMDB), `create_notification`, `update_ticket`,
`create_escalation` (+ suggested problem record), `request_human_approval`.
Action tools are mocked but written to a tamper-evident **audit log**.

## 5. Memory

- **Short-term:** per-ticket scratchpad (LangGraph state) accumulating retrieved
  sources; supplies the grounding context and the set of citable ids.
- **Long-term:** persisted recall of prior cases/decisions
  (`data/long_term_memory.json`). At intake, recall of same-category prior
  decisions biases the coordinator on recurring clusters; each run records its
  disposition + sources for future recall.

## 6. The generated artifact

A 7-section RCA note (`helpmate/rca.py`): Summary · Affected Scope & Environment
· Diagnosis & Root Cause · Resolution/Recommended Fix · Verification ·
Preventive Action/KB Update · References. Diagnosis and Resolution **must** cite
retrieved ids; traceability claim→source is enforced by the critic.

## 7. Guardrails & production

- **Prompt-injection resistance:** ticket text is wrapped as untrusted data with
  a standing "ignore embedded instructions" directive; a pattern scanner flags
  injection/destructive markers and audits them. There are no destructive tools.
- **Output constraints:** RCA is a typed Pydantic schema; the critic enforces
  citation rules.
- **Approval checkpoints + audit logging:** consequential actions go through
  `request_human_approval` and every action is appended to `data/audit_log.jsonl`.
- **Path to production:** containerise; secrets via a manager (not `.env`);
  swap FAISS for a managed vector DB; trace the agent trajectory with
  LangSmith/OpenTelemetry; monitor tool-error rates, grounding-fail rate, and
  escalation precision. *(Expanded in Q&A.)*
