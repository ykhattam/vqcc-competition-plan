#!/usr/bin/env python3
"""Phase 1 continuous step: lambda_eff-only refit of factorization weights.

Keeps the optimized ANGLE_k circuits, decoded vectors, tree, w and the fixed
symmetry shift unchanged; optimizes only native W = H2_I - sum_b w. Each outer
round re-computes the one-body-row/eigenvector matching (Hungarian), then runs a
JAX L-BFGS inner step on a penalized lambda_eff surrogate. Every proposal is
checked with the official verifier; only PASS candidates are retained.

This is the first, low-risk rung of the Phase 1 co-design loop.
"""

import argparse
import csv
import json
import os
import secrets
import sys
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import linear_sum_assignment, minimize

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, os.path.expanduser("~/vqcc/kickmix"))
sys.path.insert(0, os.path.expanduser("~/vqcc/verifiers"))
from dfthcsos import verify as vmod  # noqa: E402
from dfthcsos.scoreinfra import dfthc_math as dm  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--base", default="043642216848")
p.add_argument("--T", type=float, default=13436.0)
p.add_argument("--Q", type=float, default=405.0)
p.add_argument("--outer", type=int, default=5)
p.add_argument("--warmup", type=int, default=3)
p.add_argument("--iters", type=int, default=60)
p.add_argument("--salts", type=int, default=10)
p.add_argument("--penalty", type=float, default=5000.0)
p.add_argument("--e1-target", type=float, default=0.012)
p.add_argument("--out", default=os.path.expanduser("~/vqcc/phase1_work/refit"))
args = p.parse_args()

base = Path.home() / "vqcc" / "public_circuits" / "circuits" / "chemistry" / args.base
npz = base / "submission.npz"
data = np.load(npz)
inputs = vmod.load_inputs(npz)
edges = np.array(data["tree_edges"])
words = np.array(data["angle_words"])
audit = ROOT.parent / "phase0" / "public_audit.csv"
brot = next(int(r["brot"]) for r in csv.DictReader(audit.open())
            if r["family"] == args.base and r["status"] == "PASS")

u_all = dm.reconstruct_u_vectors(words / (1 << (brot - 1)), edges)
one_body, u_b = dm.split_u_vectors(u_all / np.linalg.norm(u_all, axis=1, keepdims=True), inputs.w)
X = dm.reconstruct_x(u_b, inputs.w)
X = np.tril(X) + np.swapaxes(np.tril(X, k=-1), -1, -2)

oei, tei = np.asarray(inputs.ref["oei"]), np.asarray(inputs.ref["tei"])
eta = int(inputs.ref["electrons"])
h1_thc = oei - 0.5 * np.einsum("prrq->pq", tei)
B = h1_thc - 0.5 * eta * inputs.S + inputs.beta_1 * np.eye(h1_thc.shape[0])
W_cur = np.asarray(inputs.W, dtype=np.float64).copy()

jX, jB, jv, jw = map(jnp.asarray, (X, B, one_body, inputs.w))
ecore, eupper = float(inputs.ref["ecore"]), float(inputs.energy_upper)
R, C = inputs.W.shape


def match(W):
    h1p = B - np.einsum("rc,rcpq->pq", W, X, optimize=True)
    _, vec = np.linalg.eigh(h1p)
    rows, cols = linear_sum_assignment(-np.abs(one_body @ vec))
    out = np.empty(len(vec), dtype=np.int64)
    out[rows] = cols
    return out


def make_objective(matched, error_only=False):
    jm = jnp.asarray(matched)

    @jax.jit
    def objective(x):
        W = jnp.asarray(x).reshape(R, C)
        h1p = jB - jnp.einsum("rc,rcpq->pq", W, jX)
        ev = jnp.linalg.eigvalsh(h1p)
        lam_j = ev[jm]
        h1p_qrom = jnp.einsum("j,jp,jq->pq", lam_j, jv, jv)
        error_1 = jnp.linalg.norm(h1p_qrom - h1p)
        w_B = W + jnp.sum(jw, axis=1)
        lam_h2 = 0.25 * jnp.sum((jnp.sum(jnp.abs(jw), axis=1) + jnp.abs(w_B)) ** 2)
        Lambda = jnp.sum(jnp.abs(ev)) + lam_h2
        E_SOS = ecore - eta * inputs.beta_1 - (jnp.sum(jnp.abs(ev)) - jnp.sum(ev)) - 0.5 * jnp.sum(W ** 2)
        E_gap = eupper - E_SOS
        rad = jnp.maximum(E_gap * (2.0 * Lambda - E_gap), 1e-12)
        if error_only:
            return error_1
        pen = (jnp.maximum(error_1 - args.e1_target, 0.0) ** 2
               + jnp.maximum(50.0 - Lambda, 0.0) ** 2
               + jnp.maximum(3.95 - E_gap, 0.0) ** 2)
        return jnp.sqrt(rad) + args.penalty * pen

    return objective


def candidate_dir(W):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    H2_I = W + np.sum(inputs.w, axis=1)
    np.savez(out / "submission.npz", **{k: (H2_I if k == "H2_I" else data[k]) for k in data.files})
    for c in base.glob("angle_k_*.kmb"):
        (out / c.name).write_bytes(c.read_bytes())
    return out


T_total, Q_total = args.T, args.Q  # fixed: circuits are reused verbatim


def fast_metrics(W):
    """Official DF-THC math, without circuit simulation (same inputs/circuits)."""
    m = dm.evaluate_dfthc_vectors(
        u_all, inputs.w, W, inputs.S, inputs.beta_1, inputs.ref,
        energy_upper=inputs.energy_upper,
    )
    ok = (m["error_1"] <= 0.020 and m["error_2"] <= 0.020
          and m["Lambda"] > 49.0 and m["E_gap"] > 3.914898
          and m["gap_in_domain"])
    m["score"] = m["lambda_eff"] * T_total * Q_total if ok else float("inf")
    return m


base_metrics = fast_metrics(W_cur)
base_result = vmod.verify(str(base))
print("base:", base_metrics["score"], "official", base_result["score"], flush=True)
best_W, best_score = W_cur.copy(), base_metrics["score"]
t0 = time.perf_counter()

# Warm-up: move off the error_1 cliff by minimizing error_1 alone (accept only
# candidates that still pass every hard gate).
for wu in range(args.warmup):
    obj = make_objective(match(W_cur), error_only=True)
    vg = jax.value_and_grad(obj)
    def warm_fun(x):
        val, grad = vg(jnp.asarray(x))
        return float(val), np.asarray(grad)

    res = minimize(warm_fun, W_cur.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": 60, "ftol": 1e-12})
    proposal = res.x.reshape(R, C)
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625):
        trial = W_cur + alpha * (proposal - W_cur)
        m = fast_metrics(trial)
        if m["error_1"] < fast_metrics(W_cur)["error_1"] and m["score"] != float("inf"):
            W_cur = trial.copy()
            print(f"warmup {wu} alpha={alpha}: error_1 {m['error_1']:.5f} "
                  f"score {m['score']:.0f}", flush=True)
            break
    else:
        print(f"warmup {wu}: stuck at error_1 {fast_metrics(W_cur)['error_1']:.5f}", flush=True)
        break

for outer in range(args.outer):
    matched = match(W_cur)
    obj = make_objective(matched)
    vg = jax.value_and_grad(obj)

    def fun(x):
        val, grad = vg(jnp.asarray(x))
        return float(val), np.asarray(grad)

    res = minimize(fun, W_cur.ravel(), jac=True, method="L-BFGS-B",
                   options={"maxiter": args.iters, "ftol": 1e-10, "gtol": 1e-8})
    proposal = res.x.reshape(R, C)
    accepted = False
    for alpha in (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125):
        trial = W_cur + alpha * (proposal - W_cur)
        m = fast_metrics(trial)
        if m["score"] < best_score:
            best_W, best_score, W_cur = trial.copy(), m["score"], trial.copy()
            accepted = True
            print(f"outer {outer} alpha={alpha}: score {best_score:.0f} "
                  f"e1={m['error_1']:.5f} lam={m['lambda_eff']:.3f}", flush=True)
            break
    if not accepted:
        print(f"outer {outer}: no improving step (surrogate "
              f"{obj(jnp.asarray(W_cur)):.4f}->{obj(jnp.asarray(proposal)):.4f})", flush=True)
        break

# Final multi-salt validation of the best candidate.
final = candidate_dir(best_W)
scores = []
for i in range(args.salts):
    r = vmod.verify(str(final), salt="" if i == 0 else f"refit-{secrets.token_hex(8)}")
    scores.append(r["score"])
print(json.dumps({
    "base_score": base_result.get("score"),
    "new_score": min(scores) if scores else None,
    "gain": base_result.get("score", float("inf")) / min(scores) - 1 if scores else None,
    "passes": len(scores),
    "seconds": round(time.perf_counter() - t0, 1),
    "path": str(final),
}, indent=2))
