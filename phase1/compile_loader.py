"""Deterministic ANGLE_k loader compiler (Phase 1 quantize+compile step).

Rebuilds the ANGLE_k circuit family from a submission's angle_word table using
the official reference builder shipped with the verifier. Supports re-quantizing
the words to a different `brot` and re-batching them into a different `J`, which
is the loader-structure search space. Every candidate is deterministic: same
inputs -> identical .kmb bytes -> identical submission SHA-256.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

import numpy as np

VERIFIERS = Path.home() / "vqcc" / "verifiers"
_BUILDER_PATH = VERIFIERS / "dfthcsos" / "example_submission" / "angle_k_reference_circuit.py"


def _builder_module():
    if str(VERIFIERS) not in sys.path:
        sys.path.insert(0, str(VERIFIERS))
    spec = importlib.util.spec_from_file_location("angle_k_reference_circuit", _BUILDER_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def requantize(words: np.ndarray, src_brot: int, dst_brot: int) -> np.ndarray:
    """Map word = theta*pi^-1 * 2^(brot-1) from one precision to another."""
    if dst_brot == src_brot:
        return words.copy()
    scale = 2 ** (dst_brot - src_brot)
    out = np.rint(words.astype(np.float64) * scale).astype(np.int64)
    return out % (1 << dst_brot)


def compile_family(src_npz: Path, out_dir: Path, *, brot: int, J: int,
                   src_brot: int | None = None, overwrite: bool = False) -> Path:
    """Write a complete run directory (angle_k_*.kmb + submission.npz).

    The factorization arrays and tree_edges are copied unchanged; only the
    angle_word table is re-quantized. The verifier ignores angle_words, so it is
    kept only to allow further re-compilation of this candidate.
    """
    mod = _builder_module()
    src = np.load(src_npz)
    words = src["angle_words"]
    if src_brot is None:
        src_brot = max(1, int(words.max()).bit_length())
    w = requantize(words, src_brot, brot)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "submission.npz", **{k: src[k] for k in src.files if k != "angle_words"},
             angle_words=w)
    circuits = mod.build_angle_k_family(w, brot, J)
    mod.write_kmx_family(circuits, out_dir, format="kmb", overwrite=overwrite)
    return out_dir
