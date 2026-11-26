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

### Oracle status: RUNNING LOCALLY — exact baseline match

Official data pulled from Kaggle to `~/vqcc` (competition archive; SHA-256 of
`FeMoCo76o113e.fcidump` = `e5a48af7…de695e1bd` and of
`FeMoCo76o113e.fixed_shifts.npz` = `7b7facf6…851cd87`, both matching the spec).
`kickmix.so` is built for Python 3.13; a `uv` venv with Python 3.13.18 + numpy 2.5.5 +
scipy 1.18.1 at `~/vqcc/.venv` loads it successfully.

Local run of `verifiers/dfthcsos/verify.py` on `sample_submission/dfthcsos`:

```
status  = PASS
score   = 3355599588.3155365   (reference 3355599588.3155375, ratio 1.0)
lambda_eff = 49.30136925057438
toffoli_count = 93237, num_qubits = 730
submission_sha256 = 276d464f…2695
```

Gate 0a deliverable met; Gate 0d stays skipped per the plan.

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

## Gate 0c — Pareto/ROI check: **DONE**

`phase0/audit_public.py`: 10 complete public `dfthcsos` families × 10 independent
random salt draws = 100 verifications, **all PASS**, score spread **0.0000%** for every
family. Best salt-surviving public cost:

```
C_public_verified = 255,384,410.16   (family 043642216848)
```

Runner-ups: 294.6M, 294.8M, 296.0M, 299.7M, 331.0M, 331.1M, 341.0M, 344.8M.
Official baseline is 3,355,599,588 → the best public family is ~13.1× cheaper.

Component breakdown (the score is `λ_eff × T × Q`):

| Family | R,B,C | J | brot | λ_eff | T | Q | score | ANGLES qubits | E_gap |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 043642216848 | 9,127,46 | 5 | 7 | 46.93 | 13,436 | 405 | 255.4M | 128 | 3.9149001 |
| 021550d932c0 | 8,143,44 | 5 | 8 | 46.92 | 14,706 | 427 | 294.6M | 128 | 3.9148980 |
| 04011855be3d | 8,143,60 | 5 | 8 | 48.52 | 16,528 | 430 | 344.8M | 128 | 3.9149022 |

Across all 10 families: λ_eff spans only 46.89–48.63 (+3.7%), T spans
13,436–16,528 (+23%), Q spans 405–430 (+6%). Correlation of `log(score)` with
`log(T)` = **0.987**, with `log(Q)` = 0.797, with `log(λ_eff)` = 0.657. Variance
share in log space: T ≈ 57%, λ ≈ 3%, Q ≈ 3% (remainder cross-terms).

**Joint check:** taking the minimum of each component across all families gives
`46.894 × 13,436 × 405 = 255.17M`, only 0.08% below the best actual family —
there is no free Pareto combination left in the public set.

**Headroom:** λ_eff is pinned near 47 because every family sits at the E_gap floor
(3.914898) with Λ ≈ 283. The verifier's floor pair (Λ=49, E_gap=3.914898) gives
λ_eff = 19.192, i.e. a theoretical 2.44× further score reduction. Loading all public
families have `error_1`/`error_2` at ~0.01999 Ha (just under the 20 mHa gate), so
lowering Λ while keeping both residuals under 20 mHa is the hard part. Loader
compression (brot 8→7, J=5) has already delivered the observed 23% T spread.

**ROI decision:** loader structure is the dominant driver of *observed* public
variance and is cheap to search; factorization has the larger theoretical ceiling
but its payoff is conditional on beating two cliff-adjacent residuals. Per the plan's
ROI rule (not a fixed gap), continue the co-design loop for now but give loader
structure priority in the first co-design sprint; revisit the pivot decision at G1.

The official leaderboard is `10 × (∏ cost_ref/cost_b)^(1/3)` over the three tasks
(Evaluation page), higher is better; Task III cost is the raw `λ_eff·T·Q`. The metric
notebook linked from Evaluation (`google-quantum-ai-metric`) is not readable with this
account (403), but the packaged `verifiers/` source is authoritative for Task III.

## Gate 0d — Oracle validation bundle: **SKIPPED**

Gate 0a succeeded (metric implementation readable), so per the plan Gate 0d is skipped.

## Gate 0e — Throughput profile: **DONE**

Measured over 100 verifications (10 families × 10 salts):

| Quantity | Value |
| --- | --- |
| Cold verification (FCIDUMP parse included) | 19.0 s |
| Cached verification (same family, new salt) | 6.82 s mean / 5.91 s median |
| Peak RSS (`/usr/bin/time -v`) | 1.13 GiB |
| 100 salt draws | 11.6 min wall, single core |

Fits comfortably in 4 cores / 16 GiB — no cloud fallback needed at this stage.
Budget for Phase 1: ~10 salts per candidate = ~68 s/candidate → ~1,200 candidates
per core-day; 50–100 adversarial perturbations per finalist = 6–11 min/finalist
→ ~130–250 finalists per core-day. With 4 cores the $500 ceiling is unlikely to bind
unless the continuous optimizer itself becomes heavy.

Note on Kaggle auth: the CLI (2.2.5) requires a `KGAT_` access token in
`~/.kaggle/access_token`, not in `kaggle.json` — a legacy `kaggle.json` produced the
"Authentication required" error. The bulk dataset download for
`manwithacat/vqcc-apache-public-circuits` 404s, but per-file downloads work;
`scripts/fetch_public_files.py` enumerates and fetches all family directories (filtering
out the dataset's thousands of unrelated `regions/*.json` files).

### Live baseline submission

Submitted the official sample submission as a complete zip to Kaggle:
`submission.zip` containing `invsqrt.kmb`, `dqi.kmb`, `dfthcsos/angle_k_0..9.kmb`,
`dfthcsos/submission.npz`. Submission ref **58940072**, status PENDING at submit time.
The first attempt 400'd because this competition requires the upload filename to be
exactly `submission.zip`; the resubmission with that name succeeded. 4 submissions
remain today. Scoring latency is ~45 min, so the live score is not yet available.
