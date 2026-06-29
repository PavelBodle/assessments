"""Loaders for the held-out incoming tickets (never indexed)."""
from __future__ import annotations

import json

from helpmate import config

_HELDOUT = config.HELDOUT_DIR / "heldout_tickets.json"


def load_heldout() -> dict:
    if not _HELDOUT.exists():
        raise RuntimeError("Held-out set missing. Run `python data/generate_data.py`.")
    return json.loads(_HELDOUT.read_text(encoding="utf-8"))


def all_heldout_tickets() -> list[dict]:
    data = load_heldout()
    return data.get("normal", []) + data.get("battery", [])


def battery() -> list[dict]:
    return load_heldout().get("battery", [])
