#!/usr/bin/env python3
"""Seed the family bank with the Phase 0 salt-audited public families."""

import csv
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from bank import add_family, connect  # noqa: E402

AUDIT = ROOT.parent / "phase0" / "public_audit.csv"
conn = connect(ROOT / "family_bank.sqlite")
seen = {}
for r in csv.DictReader(AUDIT.open()):
    if r["status"] == "PASS":
        seen.setdefault(r["family"], []).append(r)

for name, rows in sorted(seen.items()):
    r = rows[0]
    npz = Path.home() / "vqcc" / "public_circuits" / "circuits" / "chemistry" / name / "submission.npz"
    d = np.load(npz)
    scores = [float(x["score"]) for x in rows]
    add_family(
        conn, name=name, source="phase0-public", R=d["U"].shape[0],
        B=d["U"].shape[1], C=d["W"].shape[2], J=int(r["batch_size"]),
        brot=int(r["brot"]), score=min(scores),
        lambda_eff=float(r["lambda_eff"]), error_1=float(r["error_1"]),
        error_2=float(r["error_2"]), Lambda=float(r["Lambda"]),
        E_gap=float(r["E_gap"]), T_total=float(r["toffoli"]),
        Q_total=float(r["qubits"]), angles_qubits=int(r["angles_qubits"]),
        salt_draws=len(rows), passes=len(rows),
        spread_pct=100 * (max(scores) - min(scores)) / min(scores),
        path=str(npz.parent),
    )
print("seeded", len(seen), "families; best", conn.execute(
    "SELECT name,score FROM families ORDER BY score LIMIT 1").fetchone())
