# Phase 1 — co-design loop (started)

## Infrastructure

| File | Role |
| --- | --- |
| `bank.py` | SQLite family bank (`families`, `rejections`) with structural penalty logging |
| `seed_bank.py` | Seeds the bank from the Phase 0 salt audit |
| `compile_loader.py` | Deterministic ANGLE_k compiler (official builder), with `brot`/`J` re-quantization |
| `search_loader.py` | Loader-structure search: compile → multi-salt verify → accept/reject into bank |
| `refit_weights.py` | Continuous λ_eff-only refit of native `W` with JAX surrogate + official validation |

Python environment: `~/vqcc/.venv` (Python 3.13, numpy 2.5.5, scipy 1.18.1, jax 0.12.1),
`PYTHONPATH=~/vqcc/kickmix:~/vqcc/verifiers`.

## First sprint results (2026-11-26)

1. **Bank seeded** with 10 Phase 0 families; bank best `043642216848` at 255,384,410.
2. **Reference compiler is not competitive.** Rebuilding the best family's loader with
   the official `angle_k_reference_circuit.py` yields T = 43,010 vs the public
   optimized loader's T = 13,436. The public circuits use a CX-heavy QROM: circuit 0
   has 39,713 CX, 263 CCX, 79 HMR, 79 CZ and 125 qubits, while the reference builder
   uses 22,977 CX, 1,217 CCX, 1,217 HMR and 56 qubits. Since the score counts only
   Toffoli-like gates, the public loader trades free CX/workspace for ~4.6× fewer
   Toffolis. Reproducing this compiler is now the key missing capability for loader
   co-design.
3. **W-only refit is blocked by the gate cliff.** On both `043642216848` (brot 7) and
   `004d3c3d0ae4` (brot 8, the only family with ~5 mHa error_1 slack), the JAX
   surrogate finds λ_eff improvements (≈47 → 48 in objective terms) but every
   proposal fails the official `error_1 ≤ 20 mHa` gate after eigenvalue re-matching.
   A warm-up phase that minimizes error_1 alone reduces it only from 19.997 mHa to
   19.93 mHa (brot 7) / 13.62 mHa (brot 8) and still leaves no admissible descent
   direction. The entire public frontier is engineered to sit on the 20 mHa gates
   (most families have `error_1`/`error_2` ≈ 19.999999 mHa), so there is effectively
   no slack for coefficient-only refinement.

## Consequence for the plan

G1 ("first co-designed family beats public chemistry best") is **not yet met**.
The next step is one of:

- reconstruct the CX-heavy loader compiler used by the public circuits (analysis of
  `angle_k_0.kmb` above is the starting point), then re-run `search_loader.py`; or
- start from a high-precision factorization (e.g. the 12-bit baseline, error_1 =
  3.2 mHa, error_2 = 18.6 mHa) where the gates have real slack, jointly optimize
  `w`/`W`, and port the result onto an optimized loader; or
- invoke the plan's demotion trigger (G1 failure by Day 35) and run discrete loader
  search as the primary track.

No optimization code was written before Phase 0 was complete — that constraint was
honoured.
