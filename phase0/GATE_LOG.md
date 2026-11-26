# Phase 0 Gate Log

Day 1–3 work. Status as of 2026-11-26.

## Gate 0a — Read the metric implementation: **PASS (source readable)**

The authoritative verifier source is available publicly:

- `verifiers/dfthcsos/SCORING_README.md` (386 lines) — scoring specification
- `verifiers/dfthcsos/verify.py` (744 lines) — `verify()` entry point
- `verifiers/dfthcsos/scoreinfra/{circuits,costs_prepare,dfthc_math,reference}.py`
- Companion paper: arXiv:2610.23832, Eq. 44 / 74–78 and §7

Mirror fetched into `/mnt/data/tmp/opencode/vqcc-mirror/` from
`github.com/Inzainos/quantum-circuit-compilation`. **Still to do:** cross-check this
mirror byte-for-byte against the official competition archive once Kaggle access is set up
(the spec pins SHA-256 checksums for the FCIDUMP and fixed-shift files).

### The metric (lower is better)

```
score = lambda_eff * T_total * Q_total
lambda_eff = sqrt(E_gap * (2*Lambda - E_gap))
E_gap      = E_upper - E_SOS,   E_upper = -22140.27335 Ha (fixed)
```

* `T_total` = simulated ANGLES Toffoli-like gates (forward + reverse) +
  `outer_state_prep` + `inner_state_prep` + `2*(N-1)*(2*brot-1)` Givens.
* `Q_total` = persistent prep qubits + `J*brot` word registers +
  `max(outer_temp, inner_temp, angles_temp)` (shared workspace, **not** capped at 128).
* Only the **ANGLES** circuits have the 128-qubit cap; prep qubits are uncapped.

### Hard gates (as actually implemented — see "Plan corrections" below)

| Quantity | Requirement | Baseline (12-bit words) |
| --- | --- | ---: |
| `error_1` (quantized words) | ≤ 0.020 Ha | 0.003185195 |
| `error_2` (quantized words) | ≤ 0.020 Ha | 0.018639571 |
| `Lambda` | **> 49.0** (strict; ≤ rejects) | 308.455940 |
| `E_gap` | **> 3.914898 Ha** (strict; ≤ rejects) | 3.965477 |
| `0 < E_gap < 2*Lambda` | required | true |
| ANGLES circuit qubits | ≤ 128 each | 119 |

Baseline regression target (empty salt, from a public verifier run):
`score = 3355599588.3155375`, `lambda_eff = 49.30136925057439`,
`T_total = 93237`, `Q_total = 730`, `submission_sha256 = 276d464f…2695`.

### Fidelity details that matter for the oracle

* FCIDUMP two-electron integrals are cast to **float32** before use; `S` is rounded to
  **float32**. Both are deliberate and must be replicated (supports the plan's 2–3%
  tolerance rather than 1%).
* Metrics are recomputed from the **words recovered from the circuits**, not from `U`.
  `H1_I` and `Hsym_lower` are ignored; the baseline symmetry shift `S0, beta_1,0` is fixed.
* `K = ceil(75 / J)` circuits; words are `brot`-bit with `theta/pi = word / 2**(brot-1)`.
* 64 shots per address, forward and reverse; any phase/address/workspace/padding failure
  rejects. Scores are computed from sampled branches, not a proof of exact correctness.

## Gate 0b — Salt semantics: **RESOLVED (contradicts the plan)**

Both the paper and `SCORING_README.md` state that the Fiat-Shamir seed is derived from
the **SHA-256 of the submission bytes combined with a private, unpredictable salt**, which
the grader keeps secret until submissions are fixed.

- Paper: "we also combine the circuit's SHA-256 digest with a **private salt** … so that
  participants cannot perform seed grinding … reproducible once the salt is published."
- Spec: "The optional `--salt` defaults to an empty string, which gives publicly
  reproducible samples. To prevent participants from anticipating grading samples before
  committing their submission, the grader must supply a private, unpredictable salt and
  keep it secret until submissions are fixed."
- Spec: "Repeating verification with the same submission and salt repeats the same
  pseudorandom samples and adds no independent evidence."

**Consequence:** the plan's assumption (content-derived salt only, Jan 4 / Jan 20 question
dissolves) is wrong. Per the plan's own decision rule, proceed with **robustness design**:
assume the salt changes / is unknown. Multi-salt perturbation testing (50–100 perturbed
submissions) remains the right defense, because changing a byte changes the digest and
therefore the seed draw. Confirm the published/frozen status from the official rules page
once Kaggle access is available.

## Plan corrections required (do not execute the plan as written)

1. **Hard-gate inequalities in `PLAN.md` are reversed.** The plan says reject if
   `Λ > 49.0` / `E_gap > 3.914898`; the verifier rejects if `Λ <= 49.0` /
   `E_gap <= 3.914898`. Margin targets must be `Λ ≥ 49 + margin`,
   `E_gap ≥ 3.914898 + margin` — **not** `Λ ≤ 48.0`, `E_gap ≤ 3.90`.
2. **Salt semantics** are the opposite of Gate 0b's "strongly implies" paragraph.
3. **"E_gap physical meaning"** is confirmed: the spec explicitly says the thresholds
   "do not certify a 20 mHa many-electron energy error." Passing the gates is numerical,
   not chemical validity.
4. **Constraint independence** is confirmed structurally: `error_1` and `error_2` are two
   separate Frobenius residuals, each with its own 0.020 Ha gate. No evidence of a shared
   20 mHa budget; the "10 mHa each if additive" scenario can be treated as unlikely but
   should still be checked against the official rules.
5. **Leaderboard scale:** baseline score `3.356e9`; the "33.2" notebook title and "~29.9"
   live top are consistent with `score / 1e8`. Optimize against raw score, not the title.

## Gate 0c — Pareto/ROI check: **PENDING**

Needs the public factorization families / loaders / refits from the Kaggle dataset
(`manwithacat/vqcc-apache-public-circuits`) and leaderboard data.

## Gate 0d — Oracle validation bundle: **SKIPPED**

Gate 0a succeeded (metric implementation readable), so per the plan Gate 0d is skipped.

## Gate 0e — Throughput profile: **PENDING**

Needs `kickmix.so`, the sample submission, and 10 candidates. Environment currently:
Kaggle CLI repaired (`six` installed, CLI 2.2.5) but `~/.kaggle/kaggle.json` is absent.

## Blocking item

Kaggle API token required at `~/.kaggle/kaggle.json` (chmod 600) for
`kaggle competitions download -c google-verifiable-quantum-circuit-compilation`.
Rules must be accepted in a browser first. Until then the oracle cannot be executed.
