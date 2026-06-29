"""Central configuration — read from the environment exactly once.

Every model / path setting lives here so the rest of the codebase never
reads ``os.environ`` or hard-codes a model id. Switching the LLM or the
embedding model is a one-line change in ``.env`` (or Streamlit secrets).
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env if present (no-op when the vars are injected by Streamlit Cloud).
load_dotenv()


def _get(key: str, default: str | None = None) -> str | None:
    """Read a key from the OS env first, then Streamlit secrets if available."""
    val = os.environ.get(key)
    if val:
        return val
    try:  # Streamlit Cloud provides config through st.secrets, not os.environ.
        import streamlit as st  # noqa: WPS433 (optional dependency at runtime)

        if key in st.secrets:
            return str(st.secrets[key])
    except Exception:  # pragma: no cover - streamlit not installed / no secrets
        pass
    return default


# ── Paths ────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
HELDOUT_DIR = DATA_DIR / "heldout"
INDEX_DIR = DATA_DIR / "index"
TICKETS_CSV = DATA_DIR / "tickets.csv"
CMDB_JSON = DATA_DIR / "cmdb.json"
AUDIT_LOG = DATA_DIR / "audit_log.jsonl"

# ── Model configuration (single source of truth) ─────────────────────────
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini")
LLM_MODEL = _get("LLM_MODEL", "gemini-2.5-flash")
LLM_TEMPERATURE = float(_get("LLM_TEMPERATURE", "0.1"))
GEMINI_API_KEY = _get("GEMINI_API_KEY")
EMBEDDING_MODEL = _get("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

# ── Agent behaviour ──────────────────────────────────────────────────────
MAX_TOOL_CALLS = int(_get("MAX_TOOL_CALLS", "12"))

# Retrieval relevance gate — scores below this are treated as "no useful hit"
# so the coordinator can recognise a cold-retrieval situation.
RETRIEVAL_MIN_SCORE = float(_get("RETRIEVAL_MIN_SCORE", "0.30"))
