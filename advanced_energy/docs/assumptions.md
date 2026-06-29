# Assumptions & "With More Time"

## Assumptions

- **Synthetic everything.** Tickets, corpus, CMDB are generated (seed 42). Volumes
  are intentionally small per the brief ("do not over-build").
- **External actions are mocked.** Notifications, ticket updates, escalations and
  approvals only write to the audit log; no real systems are touched.
- **Single-tenant, English-only.** No multi-language or RBAC modelling.
- **Reproducibility:** LLM at temperature 0.1; embeddings normalized; all model
  ids pinned via `.env`. Generation is seeded; LLM output is still mildly
  non-deterministic across runs.
- **Grounding scope:** the critic requires citations on *Diagnosis* and
  *Resolution* (the prior-knowledge sections). Summary/Verification may be
  ticket-derived.
- **Cold-retrieval signal** = retrieval below the relevance floor returns an
  empty hit list; the coordinator treats that as "reformulate", not "proceed".

## With more time

- **Retrieval:** hybrid (BM25 + dense) with a reranker; per-section chunk metadata;
  evaluate recall@k on a labelled set rather than spot checks.
- **Agent:** add a planner that estimates blast radius from live signals; richer
  long-term memory (vector-recall of past trajectories, not just decisions);
  self-consistency on the diagnosis.
- **Critic:** NLI-based claim-level entailment against sources, not just id
  presence; numeric-claim checking.
- **Eval:** an LLM-judge for note quality alongside the behavioural checks; a
  larger held-out battery with adversarial variants; regression tracking.
- **Production:** managed vector DB, LangSmith/OTel tracing of every trajectory,
  secrets manager, autoscaling, per-tool circuit breakers and rate limits,
  human-approval UI wired to a real on-call queue.
- **Data:** more category variety and noisier free-text; deliberate label noise
  to stress the contradictory-data path.
