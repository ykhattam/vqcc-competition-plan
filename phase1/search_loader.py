#!/usr/bin/env python3
"""Phase 1 loader-structure search: vary (J, brot), verify under salts, bank winners.

Per Gate 0c this is the high-ROI track. The factorization and Givens tree are held
fixed; only the ANGLE_k loader is recompiled. The full algorithm (JAX continuous
step + demotion triggers) is layered on top of this deterministic inner loop.
"""

import argparse
import json
import os
import secrets
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("PYTHONPATH", str(Path.home() / "vqcc" / "kickmix"))
sys.path.insert(0, os.path.expanduser("~/vqcc/kickmix"))

from bank import add_family, add_rejection, best, connect  # noqa: E402
from compile_loader import compile_family  # noqa: E402

sys.path.insert(0, os.path.expanduser("~/vqcc/verifiers"))
from dfthcsos import verify as vmod  # noqa: E402

_inputs_cache, _circuits_cache = {}, {}
_orig_inputs, _orig_circuits = vmod.load_inputs, vmod.load_circuits


def cached_inputs(f, *a, **kw):
    key = str(f)
    if key not in _inputs_cache:
        _inputs_cache[key] = _orig_inputs(f, *a, **kw)
    return _inputs_cache[key]


def cached_circuits(c, *a, **kw):
    key = str(c)
    if key not in _circuits_cache:
        _circuits_cache[key] = _orig_circuits(c, *a, **kw)
    return _circuits_cache[key]


vmod.load_inputs, vmod.load_circuits = cached_inputs, cached_circuits

p = argparse.ArgumentParser()
p.add_argument("--base-family", required=True)
p.add_argument("--salt-brot", type=int, default=8, help="brot of the base family's angle words")
p.add_argument("--J", type=int, nargs="+", default=[4, 5, 6, 8])
p.add_argument("--brot", type=int, nargs="+", default=[6, 7, 8])
p.add_argument("--salts", type=int, default=10)
p.add_argument("--min-gain", type=float, default=0.01)
p.add_argument("--bank", default=str(ROOT / "family_bank.sqlite"))
p.add_argument("--work", default=os.path.expanduser("~/vqcc/phase1_work"))
args = p.parse_args()

base = Path.home() / "vqcc" / "public_circuits" / "circuits" / "chemistry" / args.base_family
npz = base / "submission.npz"
assert npz.is_file(), npz

conn = connect(args.bank)
baseline = best(conn)
bank_best_score = baseline["score"] if baseline else float("inf")
print(f"bank best: {bank_best_score}")

work = Path(args.work)
work.mkdir(parents=True, exist_ok=True)
results = []
for J in args.J:
    for brot in args.brot:
        name = f"{args.base_family}-J{J}-b{brot}"
        cand = work / name
        t0 = time.perf_counter()
        try:
            compile_family(npz, cand, brot=brot, J=J, src_brot=args.salt_brot, overwrite=True)
        except Exception as e:
            add_rejection(conn, name, J, brot, f"compile: {e}", {"compile": 1.0})
            print(f"{name}: COMPILE FAIL {e}", flush=True)
            continue

        scores, rows = [], []
        salts = [""] + [f"search-{secrets.token_hex(8)}" for _ in range(args.salts - 1)]
        for salt in salts:
            try:
                r = vmod.verify(str(cand), salt=salt)
                m, res = r["metrics"], r["resources"]
                scores.append(r["score"])
                rows.append(r)
            except Exception as e:
                add_rejection(conn, name, J, brot, f"verify: {e}", {"verify_fail": 1.0})
                print(f"{name}: VERIFY FAIL {e}", flush=True)
                break
        if len(rows) != len(salts):
            continue

        best_row = min(rows, key=lambda r: r["score"])
        m, res = best_row["metrics"], best_row["resources"]
        spread = 100 * (max(scores) - min(scores)) / min(scores)
        gain = bank_best_score / min(scores) - 1
        penalties = {
            "error_1_margin": 0.020 - m["error_1"],
            "error_2_margin": 0.020 - m["error_2"],
            "gap_margin": m["E_gap"] - 3.914898,
            "lambda_margin": m["Lambda"] - 49.0,
            "qubit_headroom": 128 - res["angles_qubits"],
        }
        if gain < args.min_gain:
            add_rejection(conn, name, J, brot, f"no gain ({gain:.2%})", penalties)
            print(f"{name}: score {min(scores):.0f} gain {gain:.2%} rejected", flush=True)
            continue

        d = np.load(npz)
        add_family(
            conn, name=name, source="loader-search", R=d["U"].shape[0],
            B=d["U"].shape[1], C=d["W"].shape[2], J=J, brot=brot,
            score=min(scores), lambda_eff=m["lambda_eff"], error_1=m["error_1"],
            error_2=m["error_2"], Lambda=m["Lambda"], E_gap=m["E_gap"],
            T_total=res["toffoli"], Q_total=res["qubits"],
            angles_qubits=res["angles_qubits"], salt_draws=len(salts),
            passes=len(scores), spread_pct=spread, path=str(cand),
        )
        bank_best_score = min(scores)
        print(f"{name}: ACCEPT score {min(scores):.0f} gain {gain:.2%} "
              f"T={res['toffoli']} Q={res['qubits']} time={time.perf_counter()-t0:.1f}s", flush=True)
        results.append(name)

print(json.dumps({"accepted": results, "bank_best": bank_best_score}))
