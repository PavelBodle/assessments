# Data Model & Held-out Split (Part 1C)

## Datasets

| Artifact | Path | Description |
|---|---|---|
| Tickets | `data/tickets.csv` | 160 structured tickets (all assignment fields). |
| Resolution notes | `data/corpus/resolutions/RES-*.md` | 22 RCA-style notes, each tied to a `ticket_id`/`case_id`. |
| KB articles | `data/corpus/kb/KB-*.md` | 8 hand-authored step-by-step fix articles. |
| CMDB | `data/cmdb.json` | Per-`requester_id` asset record (device, OS, VPN client). |
| Held-out | `data/heldout/heldout_tickets.json` | Incoming tickets **excluded from the index**. |

## Keys & joins

```
tickets.ticket_id   (PK)
tickets.case_id     (FK) ──▶ escalated problem/RCA record (CASE-xxxxx); NULL/"" if first-line resolved
tickets.requester_id ───▶ cmdb.json[requester_id]            (asset lookup)
resolution_note.ticket_id ─▶ tickets.ticket_id              (a note documents a resolved ticket)
resolution_note.case_id   ─▶ shared CASE for a recurring problem cluster
RCA references            ─▶ {ticket_id | CASE | RES | KB} ids (citation traceability)
```

Only a realistic subset of tickets escalates: `case_id` is populated for P1s and
for tickets belonging to a recurring **problem cluster** (e.g. `CASE-00042` VPN,
`CASE-00051` Outlook, `CASE-00067` lockouts). Most P3/P4 tickets resolve at first
line with `case_id` empty.

## Realism & correlations (built into `data/generate_data.py`)

- **Priority skew:** ~74% P3/P4, ~20% P2, ~6% P1 (mostly routine, quick).
- **Elevated categories:** VPN and Email spike via recurring clusters.
- **Clusters / duplicates:** 4 recurring problems generate related/near-duplicate
  tickets so similar-ticket retrieval has dense neighbourhoods.
- **Sensible correlations:** P1 → Escalated + `case_id` set + higher SLA-breach
  probability + longer time-to-resolve; `Workaround`/`No Issue Found` → higher
  `reopened` rate; SLA-breach probability rises with priority.

## Held-out split (honest retrieval)

`heldout_tickets.json` holds incoming tickets that are **never embedded/indexed**,
so retrieval and draft generation are demoed against unseen inputs:

- **3 normal** tickets (`HLD-0001..3`) whose problems *should* retrieve cluster
  siblings (VPN drop, Outlook sync, account lockout).
- **5 adaptive battery** scenarios (`HLD-COLD`, `HLD-CONFLICT`, `HLD-OOS`,
  `HLD-P1`, `HLD-INJECT`) — each carries a `scenario` tag and an `expected`
  behaviour used by `eval/run_eval.py`.

Regenerate everything deterministically (seed 42) with:

```bash
python data/generate_data.py
```
