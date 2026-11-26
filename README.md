# VQCC Competition Plan

Executable competition plan for the VQCC competition.

The full plan lives in [PLAN.md](PLAN.md).

## Strategy

- **Path A (recommended):** Task III only for the $10k Single-Task Prize, with public Task I/II circuits as a valid-completion fallback. Target: `dfthcsos`.
- **Phase 0 (Days 1–3):** Resolve plan-changing unknowns (metric oracle, salt semantics, Pareto/ROI, oracle validation, throughput) before any optimization.
- **Phase 1 (Days 4–35):** Alternating JAX continuous optimization + deterministic QROM compilation + salt verification into a SQLite family bank, with explicit demotion triggers.
- **Phase 2 (Days 36–50):** Submission policy, empirical Single-Task threshold, $500 compute ceiling, and team/merge deadlines.

See `PLAN.md` for all decision gates, kill rules, and "what not to do" constraints.
