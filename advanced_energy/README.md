# HelpMate — Agentic IT Support Ticket Resolution Agent

An autonomous multi-agent assistant that triages an IT support ticket, retrieves
grounded evidence (prior resolutions + KB via RAG), drafts a structured
**Resolution / RCA note**, has a **critic** verify every claim cites a source,
takes mock downstream actions, and logs the full agent trajectory.

> Agency is the point: the **coordinator dynamically chooses tools and when to
> stop — no hard-coded pipeline.** It handles cold retrieval, contradictory data,
> out-of-scope requests, P1 priority branching, and prompt-injection.

## Stack

- **LLM:** Google Gemini API — `langchain-google-genai`. Default `gemma-4-26b-a4b-it`
  (Gemma's free tier is far more generous than Gemini Flash's 5 RPM / 20 RPD).
  Swap to `gemini-2.5-flash` etc. by editing `.env` only.
- **Embeddings:** HuggingFace `BAAI/bge-small-en-v1.5` (local, no key)
- **Vector store:** FAISS · **Orchestration:** LangGraph · **UI:** Streamlit
- **Modular models:** every model id lives in `.env` only (`helpmate/config.py`
  reads it once; `helpmate/llm.py` is the single factory). Swap models without
  touching code.

## Setup

```bash
cd advanced_energy
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
#   (or: python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt)

cp .env.example .env        # then add your free GEMINI_API_KEY
#   key: https://aistudio.google.com/apikey
```

## Run

```bash
.venv/bin/python data/generate_data.py        # 1) synthesize tickets + corpus (seeded)
.venv/bin/python -m helpmate.indexing         # 2) build the FAISS index
.venv/bin/python -m helpmate.cli HLD-P1       # 3) run one held-out ticket (terminal)
.venv/bin/python -m eval.run_eval             # 4) run the 5-scenario battery + scores
.venv/bin/streamlit run app.py                # 5) interactive demo
```

## Project layout

```
data/           generate_data.py · tickets.csv · corpus/ · heldout/ · index/
helpmate/       config · llm (factory) · indexing · tools · memory · rca ·
                guardrails · trajectory · prompts · graph (LangGraph) · cli
eval/           scenarios.py (behaviour scoring) · run_eval.py (battery)
docs/           architecture.md · data_model.md · assumptions.md
app.py          Streamlit demo
```

## Adaptive scenario battery (held-out)

`HLD-COLD` (cold retrieval) · `HLD-CONFLICT` (contradictory data) ·
`HLD-OOS` (out of scope) · `HLD-P1` (priority branching + HITL) ·
`HLD-INJECT` (prompt injection). Scored by `eval/run_eval.py` on tool-selection
correctness, trajectory efficiency, failure recovery, correct escalation, and
grounding/citation faithfulness.

## Deploy (free, shareable)

Streamlit Community Cloud: push to GitHub, point it at `app.py`, set the same
keys in the **Secrets** dashboard (see `.streamlit/secrets.toml.example`). The
prebuilt `data/index/` and generated data are committed so the app starts
without re-embedding. Fallbacks: Hugging Face Spaces (Streamlit SDK) or Render.

See [`docs/architecture.md`](docs/architecture.md) for the full design note.
