# Final Plan — VQCC Competition (Executable Version)

> **Execution status:** Phase 0 Gates 0a/0b resolved. See [phase0/GATE_LOG.md](phase0/GATE_LOG.md) for findings. The text below has been amended to match the actual verifier.

## Strategic Position

**Binary choice (resolve NOW):**
- **Path A (recommended):** Task III only ($10k Single-Task Prize) + Task I/II public fallback for overall GM. Drop Longest Leader entirely. Pre-authorize $500 cloud compute ceiling.
- **Path B:** Task I/II only (lower prize, higher probability). Drop Task III.
- **Path C:** Exit. Redirect 8 weeks elsewhere.

**Target (Path A):** $10k Single-Task Prize on Task III (`dfthcsos`).  
**Fallback:** Valid complete submission using public Task I/II circuits.  
**Do not target:** Overall top-6 (crowded, low ROI).  
**Do not pursue:** Longest Leader (conflicts with keeping Task III private until Jan 20).

---

## Phase 0 — Resolve Plan-Changing Unknowns (Days 1–3)

### Gate 0a (Day 1–2): Read the metric implementation
The competition provides a metric implementation link. **If readable:** use it directly, skip Gate 0d validation, and proceed to Phase 1. **If unreadable or takes >2 days:** proceed to Gate 0d.

**Deliverable:** A working oracle that reproduces public leaderboard chemistry scores within 2–3% (not 1%, to account for float32/float64 drift).

**Kill rule:** If Gate 0a takes >2 days or the implementation is partial, trigger Phase 2 compression (cut to 15 days).

**Status (Day 1):** implementation readable via public verifier mirror + companion paper; official cross-check pending. The implemented hard gates are `Λ > 49.0` and `E_gap > 3.914898 Ha` (the inequalities in the original draft were reversed).

### Gate 0b (Day 1–2): Resolve salt semantics
**Resolved:** the verifier seeds its sampler with the submission's SHA-256 digest **combined with a private, unpredictable salt** that the grader keeps secret until submissions are fixed (paper §II and `SCORING_README.md`). Repeating verification with the same submission and salt repeats the same samples and adds no independent evidence.

**Consequence:** proceed with robustness design. Treat every public score as one draw under an unknown salt. Do not assume the salt is content-derived only, and do not assume it is frozen across Jan 4 / Jan 20 until confirmed in writing.

**Deliverable:** Written confirmation of salt semantics from rules/forum/notebook discussions.

### Gate 0c (Day 2–3): Pareto-status check + ROI rule
Take best public factorization families + best public loaders + best continuous λ_eff-only refits. Compute joint cost vs. component costs.

**Pivot to loader-only** if expected improvement per compute-day is lower than the best alternative (loader-only, alternate family, or Task I/II fallback). **Do not use a fixed 3% gap.**

**Deliverable:** ROI calculation with explicit compute-day estimates.

### Gate 0d (Day 3): Oracle validation bundle (only if Gate 0a failed)
Validate the oracle on **20–30 stratified candidates per family shape**. Metrics:
- Global Spearman/Kendall by family shape
- **Cliff-adjacent rank correlation** (candidates within 5% of `20 mHa`, `Λ ≥ 49.0`, `E_gap ≥ 3.914898 Ha`)
- Top-k regret
- Calibration slope/offset
- Hard-gate precision/recall

**Kill if:** cliff-adjacent rank correlation < 0.8 **or** hard-gate precision < 0.95.  
**Accept if:** absolute offset is acceptable only if it preserves rankings and gate decisions.

### Gate 0e (Day 3): Throughput profile
Measure compile time, salt-verify time, memory peak, and cache hit rate on 10 candidates. Set candidates/day and salts/family from observed throughput.

**Throughput fallback:** If the full loop (including 50–100 perturbations) does not fit 4 cores / 16 GiB:
- Reduce candidate count
- Reduce salt count
- Use cloud compute (pre-authorized $500 ceiling)
- Accept slower iteration

**Deliverable:** Confirmed throughput budget for Phase 1.

**Phase 0 gate:** Do not start Phase 1 until Gates 0a–0e are resolved.

---

## Phase 1 — Co-Design Loop (Days 4–35)

### Architecture: Alternating loop, not a single differentiable pipeline

```
┌─────────────────────────────────────────────────────┐
│  JAX/PyTorch continuous step                         │
│  → U, W, H2_I, H1_I, Hsym_lower                     │
│  → angle/bit-width allocation (smooth)              │
│  → augmented Lagrangian on: error ≤ 20 mHa, Λ > 49  │
│  → objective: minimize λ_eff × expected loader cost │
└──────────────────────┬──────────────────────────────┘
                       │ quantized angles
                       ▼
┌─────────────────────────────────────────────────────┐
│  Deterministic QROM compiler (Rust/Python)           │
│  → angle_k_*.kmb, ≤128 qubits each                  │
│  → log every rejected compilation with penalty      │
└──────────────────────┬──────────────────────────────┘
                       │ candidate family
                       ▼
┌─────────────────────────────────────────────────────┐
│  Salt/verification check                             │
│  → run on submission-hash + private salt draws     │
│  → for final candidates: adaptive escalation        │
│    (50–100 perturbed submissions)                   │
└──────────────────────┬──────────────────────────────┘
                       │ accept/reject + feedback
                       ▼
              SQLite family bank
```

### Loop

1. **Continuous step:** JAX optimization of factorization + bit-width allocation. Hard barriers on `error_1`, `error_2`, `E_gap`, `Λ`. Use the differentiable surrogate **only** for smooth allocation, not for loader structure.

2. **Quantize + compile:** Extract angles, compile loaders with deterministic compiler. If any loader >128 qubits → reject and feed back a penalty. **Log every rejected compilation with its structural penalty vector.**

3. **Post-quantization stability check:** Recompute `error_1`, `error_2`, `E_gap` from the **quantized** angles (not continuous ones) before accepting.

4. **Salt check:** Run across sampled private-salt draws (per Gate 0b), since the real salt is unknown. For final candidates, escalate to **adaptive salt perturbation** (50–100 perturbed submissions) and adversarial margin testing.

5. **Accept/reject:** Accept only if cost beats bank best by more than the empirical 95th-percentile bound of salt-perturbation cost, **and** gate margins clear each cliff by a fixed safety buffer (`error ≤ 19.0 mHa`, `Λ ≥ 50.0`, `E_gap ≥ 3.95 Ha`). A literal 5% E_gap margin (≥4.11 Ha) would reject the baseline at 3.965 Ha, so use the absolute buffer above.

6. **Demotion trigger:** If cliff-adjacent surrogate correlation < 0.7 for 3 consecutive days, or hard-gate precision < 0.90 on any family, **demote JAX to bit-width/angle allocation only** and run discrete search (CP-SAT, evolutionary, LNS) for loader structure.

7. **Decoupling test:** After demotion, verify for 5 consecutive optimization steps that angle/bit-width allocations are independent of loader-structure feedback. If coupling re-emerges, demote further (JAX → offline warm-start only).

8. **Feedback:** Pass loader structure and salt failures back to the continuous step as updated surrogate penalties.

### Operationalized Dimensions

**Tree structure (discrete, not continuous):** `tree_edges` (the Givens tree) is a discrete structure. Treat tree optimization as a discrete sub-problem (CP-SAT, evolutionary, LNS), not a smooth JAX variable.

**Non-kickmix pricing (analytic):** The chemistry score includes Givens rotations, state prep, and QROM pricing that are **not** kickmix-verified. The oracle must replicate the analytic pricing, not just `kickmix` verification.

**Constraint independence (confirmed structurally):** `error_1` and `error_2` are two separate Frobenius residuals, each with its own 0.020 Ha gate, recomputed from the **quantized** words recovered from the circuits. No shared 20 mHa budget appears in the verifier; keep separate margins for each.

**E_gap physical meaning (confirmed):** the spec explicitly states the acceptance thresholds "do not certify a 20 mHa many-electron energy error." Passing `E_gap > 3.914898 Ha` is a numerical gate, not proof of valid chemistry. The score `sqrt(E_gap*(2Λ−E_gap)) * T_tot * Q_tot` can fall as E_gap rises toward 2Λ, so chemistry validity must be checked separately against reference data.

**Task I/II fallback verification:** Verify Task I/II circuits under the same salt/verification semantics as Task III before relying on them for the overall GM fallback.

**Family diversity:** Keep ≥5 distinct `(R, B, C, J)` shapes in the SQLite bank. Diversity is a hedge against salt-perturbation sensitivity, not just overfitting.

**Version pinning:** Pin JAX, Python, `kickmix`, and all dependency versions. Run a small regression suite (fixed public candidates, fixed salts, expected metric outputs) weekly to catch drift.

---

## Phase 2 — Competitive Positioning (Days 36–50)

### Submission Policy

| Date | Action |
|------|--------|
| Now–Dec | Submit valid complete entry: best public Task I/II circuits + your current best Task III family. Verify Task I/II circuits under the same salt/verification semantics as Task III. |
| Jan 4 | **Do not release best Task III.** Keep private for Single-Task Prize. |
| Jan 10–19 | Reserve 1–2 submission slots for final swaps. Account for **~45 min scoring latency** per submission. |
| Jan 20 | Final submission deadline. |

### Single-Task Threshold

Define empirically after Phase 0:

> **Threshold = best salt-surviving public chemistry cost − (2 × empirical 95th-percentile perturbation bound) − safety margin.**

If your best family does not beat this threshold by Day 45, **reassess**: either pivot to a different Task III family shape or accept the participation-valid fallback and redirect compute to protecting overall GM.

### Dollar Budget

**Pre-authorize $500 cloud compute ceiling** now (before Phase 1). Set the explicit dollar ceiling as a function of Phase 1 iteration count, which is itself a function of Gate 0e throughput.

**Kill rule:** If Phase 1 exceeds $500 by Day 20, fall back to public Task III + public Task I/II.

### Team/Merger

Entry and team-merger deadlines are Jan 13. If solo, no action. If recruiting or merging, resolve by Jan 10.

---

## Decision Gates

| Gate | When | Kill/Continue |
|------|------|---------------|
| G0a: Metric implementation readable | Day 2 | If no → proceed to Gate 0d, compress Phase 2 to 15 days |
| G0b: Salt semantics confirmed | Day 2 | **Done:** salt is private → robustness design (multi-salt perturbation testing) |
| G0c: Pareto-status ROI check | Day 3 | If loader-only has higher ROI → pivot to loader-only, skip co-design |
| G0d: Oracle validation bundle | Day 3 | If cliff-adjacent ρ < 0.8 or hard-gate precision < 0.95 → stop, cannot optimize reliably |
| G0e: Throughput profile | Day 3 | If full loop does not fit 4 cores / 16 GiB → reduce candidates/salts or use cloud |
| G1: First co-designed family beats public chemistry best under salt check | Day 35 | If no → reassess family shape or pivot to loader-only attack |
| G2: Family beats Single-Task threshold | Day 45 | If no → accept fallback; protect overall GM |
| G3: Final family survives 50–100 perturbed submissions with gate margins | Day 50 | If no → revert to best verified bank entry |

---

## What Not to Do

- **Do not anchor to 33.2.** The live top is ~29.9. Optimize against salt-surviving scores, not notebook titles.
- **Do not build a general agentic harness first.** The public toolkits (`vqcc-agent-toolkit`, `kickmix-tools`) already cover serialization and verification. Spend your time on co-design and salt robustness.
- **Do not run e-graphs + CP-SAT + stochastic search concurrently on 4 cores.** Stage-gate: one heavy solver at a time.
- **Do not leak best Task III work on Jan 4.** Keep private for Single-Task Prize.
- **Do not trust a single salt.** Every accepted family must pass multi-salt verification with gate margins.
- **Do not assume the private salt is frozen or content-derived only** across Jan 4 / Jan 20 unless confirmed in writing (Gate 0b: confirmed private).
- **Do not use a fixed 3% Pareto gap.** Use ROI rule (Gate 0c).
- **Do not set dollar budget before throughput profile.** Pre-authorize $500 ceiling, then set explicit ceiling after Gate 0e.
- **Do not continue auditing.** Execute Gate 0a now.

---

## One-Line Summary

**Read the metric implementation on Day 1 (Gate 0a), resolve salt semantics and Longest Leader conflict as binary choices now, co-design factorization and QROM loaders in a closed JAX + `kickmix` loop under hard gate margins with explicit demotion triggers, keep the best Task III family private until Jan 20, and treat every public score as a single draw from a private, unknown salt rather than a stable target.**
