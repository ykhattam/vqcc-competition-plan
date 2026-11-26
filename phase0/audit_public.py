#!/usr/bin/env python3
"""Audit public dfthcsos families under independent (random) salt draws.

Uses the official verifier from the competition archive. Caches FCIDUMP loading
and circuit parsing so repeated salt draws are cheap. Records pass/fail variance
and the cost of the best salt-surviving family (C_public_verified).
"""

import argparse
import csv
import json
import os
import secrets
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--verifiers", default=os.path.expanduser("~/vqcc/verifiers"))
parser.add_argument("--families", default=os.path.expanduser("~/vqcc/public_circuits/circuits/chemistry"))
parser.add_argument("--salts", type=int, default=10)
parser.add_argument("--out", default="phase0/public_audit.csv")
parser.add_argument("--limit", type=int, default=0, help="audit at most N families (0 = all)")
args = parser.parse_args()

sys.path.insert(0, args.verifiers)
from dfthcsos import verify as vmod  # noqa: E402

# Cache the two expensive, salt-independent parts of verify().
_inputs_cache: dict[str, object] = {}
_circuits_cache: dict[str, object] = {}
_orig_load_inputs = vmod.load_inputs
_orig_load_circuits = vmod.load_circuits


def cached_load_inputs(factorization, *a, **kw):
    key = str(factorization)
    if key not in _inputs_cache:
        _inputs_cache[key] = _orig_load_inputs(factorization, *a, **kw)
    return _inputs_cache[key]


def cached_load_circuits(circuits, *a, **kw):
    key = str(circuits)
    if key not in _circuits_cache:
        _circuits_cache[key] = _orig_load_circuits(circuits, *a, **kw)
    return _circuits_cache[key]


vmod.load_inputs = cached_load_inputs
vmod.load_circuits = cached_load_circuits

families = sorted(
    p for p in Path(args.families).iterdir()
    if p.is_dir() and (p / "submission.npz").is_file() and (p / "angle_k_0.kmb").is_file()
)
if args.limit:
    families = families[: args.limit]

FIELDS = [
    "family", "salt_index", "status", "score", "lambda_eff", "error_1", "error_2",
    "Lambda", "E_gap", "angles_qubits", "toffoli", "qubits", "brot", "batch_size",
    "num_circuits", "seconds", "error",
]

rows = []
for family in families:
    salts = [""] + [f"audit-{secrets.token_hex(8)}" for _ in range(args.salts - 1)]
    for i, salt in enumerate(salts):
        t0 = time.perf_counter()
        row = {k: "" for k in FIELDS}
        row.update(family=family.name, salt_index=i, salt=salt if i else "<empty>")
        try:
            r = vmod.verify(str(family), salt=salt)
            m, res = r["metrics"], r["resources"]
            row.update(
                status="PASS", score=r["score"], lambda_eff=m["lambda_eff"],
                error_1=m["error_1"], error_2=m["error_2"], Lambda=m["Lambda"],
                E_gap=m["E_gap"], angles_qubits=res["angles_qubits"],
                toffoli=r["toffoli_count"], qubits=r["num_qubits"],
                brot=r["brot"], batch_size=r["batch_size"],
                num_circuits=r["num_circuits"],
            )
        except Exception as e:  # rejections are data, not crashes
            row.update(status="FAIL", error=f"{type(e).__name__}: {e}"[:300])
        row["seconds"] = round(time.perf_counter() - t0, 3)
        rows.append(row)
        print(f"{family.name} salt {i}: {row.get('status')} {row.get('score', '')}", flush=True)

out = Path(args.out)
out.parent.mkdir(parents=True, exist_ok=True)
with out.open("w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS + ["salt"], extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

# Summary
summary = {}
for name in {r["family"] for r in rows}:
    fr = [r for r in rows if r["family"] == name and r["status"] == "PASS"]
    summary[name] = {
        "passes": len(fr), "draws": len([r for r in rows if r["family"] == name]),
        "best_score": min((r["score"] for r in fr), default=None),
        "worst_score": max((r["score"] for r in fr), default=None),
        "score_spread_pct": (
            100 * (max(r["score"] for r in fr) - min(r["score"] for r in fr))
            / min(r["score"] for r in fr) if fr else None
        ),
    }
(out.with_suffix(".summary.json")).write_text(json.dumps(summary, indent=2))
print(json.dumps({"families": len(summary), "draws": len(rows)}, indent=2))
