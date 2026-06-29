"""Run the adaptive scenario battery and score agentic behaviour.

Usage:  python -m eval.run_eval
Requires GEMINI_API_KEY (the agent calls the LLM).
"""
from __future__ import annotations

import sys

from helpmate.datasets import battery
from helpmate.graph import run_ticket
from eval.scenarios import score_scenario, summarise


def main() -> int:
    results_by_scenario: dict[str, dict] = {}
    for ticket in battery():
        scenario = ticket["scenario"]
        print(f"\n{'='*70}\nSCENARIO: {scenario}  ({ticket['ticket_id']})")
        print(f"  subject: {ticket['subject']}")
        result = run_ticket(ticket)
        checks = score_scenario(scenario, result)
        results_by_scenario[scenario] = checks

        print(f"  disposition: {(result.get('disposition') or {}).get('disposition')}")
        print(f"  tool calls : {result['tool_calls']}")
        for dim, (ok, note) in checks.items():
            print(f"    [{'PASS' if ok else 'FAIL'}] {dim}: {note}")

    s = summarise(results_by_scenario)
    print(f"\n{'='*70}\nBEHAVIOUR SCORE: {s['passed']}/{s['total']} checks passed (rate={s['rate']})")
    return 0 if s["rate"] >= 0.8 else 1


if __name__ == "__main__":
    sys.exit(main())
