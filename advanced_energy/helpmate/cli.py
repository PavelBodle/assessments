"""Run HelpMate on one held-out ticket from the terminal.

Usage:
    python -m helpmate.cli                # lists held-out ticket ids
    python -m helpmate.cli HLD-P1         # run a specific held-out ticket
"""
from __future__ import annotations

import sys

from helpmate.datasets import all_heldout_tickets
from helpmate.graph import run_ticket


def main(argv: list[str]) -> int:
    tickets = {t["ticket_id"]: t for t in all_heldout_tickets()}
    if len(argv) < 2:
        print("Held-out tickets:")
        for tid, t in tickets.items():
            print(f"  {tid:14s} {t.get('scenario','normal'):20s} {t['subject']}")
        print("\nRun one with:  python -m helpmate.cli <TICKET_ID>")
        return 0

    tid = argv[1]
    if tid not in tickets:
        print(f"Unknown ticket {tid}. Run with no args to list options.")
        return 1

    result = run_ticket(tickets[tid])
    print("\n" + "=" * 72)
    print(result["trajectory"].render())
    print("=" * 72)
    disp = result.get("disposition") or {}
    print(f"\nDisposition: {disp.get('disposition')} — {disp.get('rationale')}")
    print(f"Tool calls : {result['tool_calls']}")
    print(f"Critic     : grounded={ (result.get('critic') or {}).get('grounded') }")
    if result.get("note_markdown"):
        print("\n" + result["note_markdown"])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
