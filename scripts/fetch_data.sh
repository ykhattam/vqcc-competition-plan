#!/usr/bin/env bash
# Phase 0 data pull. Requires ~/.kaggle/kaggle.json (chmod 600) and accepted rules.
set -euo pipefail

TOKEN="$HOME/.kaggle/kaggle.json"
if [[ ! -f "$TOKEN" ]]; then
  echo "ERROR: $TOKEN missing. Create it at kaggle.com/settings -> API -> Create New Token." >&2
  exit 1
fi
chmod 600 "$TOKEN"

WORK="${1:-$HOME/vqcc}"
mkdir -p "$WORK"
cd "$WORK"

echo "[1/3] competition archive"
kaggle competitions download -c google-verifiable-quantum-circuit-compilation -p "$WORK" --force
unzip -oq google-verifiable-quantum-circuit-compilation.zip -d "$WORK"

echo "[2/3] public circuits dataset"
mkdir -p "$WORK/public_circuits"
kaggle datasets download -d manwithacat/vqcc-apache-public-circuits -p "$WORK/public_circuits" --unzip --force

echo "[3/3] agent toolkit dataset"
mkdir -p "$WORK/toolkit"
kaggle datasets download -d manwithacat/vqcc-agent-toolkit -p "$WORK/toolkit" --unzip --force

echo "[verify] checksums (expected values in phase0/GATE_LOG.md)"
sha256sum \
  "$WORK/verifiers/dfthcsos/FeMoCo76o113e.fcidump" \
  "$WORK/verifiers/dfthcsos/FeMoCo76o113e.fixed_shifts.npz" || true

echo "done: $WORK"
