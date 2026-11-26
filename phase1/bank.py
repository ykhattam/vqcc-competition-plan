"""SQLite family bank for Phase 1.

Every accepted candidate is stored with its verified metrics; every rejected
compilation/verification is stored with its structural penalty vector so the
continuous step can consume the feedback (plan Phase 1, steps 2 and 8).
"""

import json
import sqlite3
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS families (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    source TEXT NOT NULL,
    R INTEGER, B INTEGER, C INTEGER,
    J INTEGER, brot INTEGER,
    score REAL NOT NULL,
    lambda_eff REAL NOT NULL,
    error_1 REAL NOT NULL,
    error_2 REAL NOT NULL,
    Lambda REAL NOT NULL,
    E_gap REAL NOT NULL,
    T_total REAL NOT NULL,
    Q_total REAL NOT NULL,
    angles_qubits INTEGER NOT NULL,
    salt_draws INTEGER NOT NULL,
    passes INTEGER NOT NULL,
    spread_pct REAL,
    path TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS rejections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    J INTEGER, brot INTEGER,
    reason TEXT NOT NULL,
    penalties TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_families_score ON families(score);
"""

FIELDS = (
    "name", "source", "R", "B", "C", "J", "brot", "score", "lambda_eff",
    "error_1", "error_2", "Lambda", "E_gap", "T_total", "Q_total",
    "angles_qubits", "salt_draws", "passes", "spread_pct", "path", "created_at",
)


def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(SCHEMA)
    return conn


def add_family(conn: sqlite3.Connection, **kw) -> int:
    kw.setdefault("created_at", time.time())
    cols = ",".join(FIELDS)
    marks = ",".join("?" for _ in FIELDS)
    cur = conn.execute(
        f"INSERT INTO families ({cols}) VALUES ({marks})",
        tuple(kw.get(k) for k in FIELDS),
    )
    conn.commit()
    return int(cur.lastrowid)


def add_rejection(conn: sqlite3.Connection, name: str, J, brot, reason: str,
                  penalties: dict) -> int:
    cur = conn.execute(
        "INSERT INTO rejections (name,J,brot,reason,penalties,created_at) VALUES (?,?,?,?,?,?)",
        (name, J, brot, reason[:500], json.dumps(penalties), time.time()),
    )
    conn.commit()
    return int(cur.lastrowid)


def best(conn: sqlite3.Connection) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM families WHERE passes > 0 ORDER BY score LIMIT 1"
    ).fetchone()
