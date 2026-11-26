#!/usr/bin/env bash
# Phase 0 data pull. Requires ~/.kaggle/kaggle.json (chmod 600) and accepted rules.
set -euo pipefail

TOKEN="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$TOKEN" ]]; then
  echo "ERROR: $TOKEN missing. Create it at kaggle.com/settings -> API -> Create New Token." >&2
  exit 1
fi
chmod 600 "$TOKEN"

# Kaggle CLI 2.2.5 reads KGAT_ access tokens from ~/.kaggle/access_token.
python3 - <<'PY'
import json, os, pathlib
cfg = json.load(open(os.path.expanduser("~/.kaggle/kaggle.json")))
key = cfg.get("key", "")
if key.startswith("KGAT_"):
    p = pathlib.Path.home() / ".kaggle" / "access_token"
    p.write_text(key)
    p.chmod(0o600)
PY

WORK="${1:-$HOME/vqcc}"
mkdir -p "$WORK"
cd "$WORK"

echo "[1/3] competition archive"
kaggle competitions download -c google-verifiable-quantum-circuit-compilation -p "$WORK" --force
unzip -oq google-verifiable-quantum-circuit-compilation.zip -d "$WORK"

echo "[2/3] public circuits dataset (bulk download 404s; fetch per file)"
python3 "$(dirname "$0")/fetch_public_files.py"

echo "[3/3] agent toolkit dataset"
mkdir -p "$WORK/toolkit"
kaggle datasets download -d manwithacat/vqcc-agent-toolkit -p "$WORK/toolkit" --unzip --force || true

echo "[verify] checksums (expected values in phase0/GATE_LOG.md)"
sha256sum \
  "$WORK/verifiers/dfthcsos/FeMoCo76o113e.fcidump" \
  "$WORK/verifiers/dfthcsos/FeMoCo76o113e.fixed_shifts.npz" || true

echo "done: $WORK"
