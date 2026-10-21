"""Validation against the reference ``fosters_inverse_sss.py`` implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from .basis import compute_basis
from .methods import foster_operator
from .metrics import pearson


def _load_reference():
    """Load the untouched reference implementation from the repository root."""
    path = Path(__file__).resolve().parent.parent / "fosters_inverse_sss.py"
    if not path.exists():
        raise FileNotFoundError(f"Reference implementation not found: {path}")
    spec = importlib.util.spec_from_file_location(
        "fosters_inverse_sss_reference", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference_inverse(raw, N, ntype):
    """Run the reference implementation's inverse on ``raw`` and ``N``.

    The reference file is imported as-is; it is never modified.
    """
    if ntype not in ("E", "OTP"):
        raise ValueError(f"ntype must be 'E' or 'OTP', got {ntype!r}")
    return _load_reference()._do_inverse(raw, N, ntype)


def check_against_reference(raw, N, ntype):
    """Compare this package's ``foster`` variant with the reference implementation.

    The comparison uses the reference implementation's hard-coded basis settings:
    ``origin=(0, 0, 0)``, ``coord_frame="meg"`` and ``regularize=None``.
    ``ntype="E"`` uses the internal basis; ``ntype="OTP"`` uses the full
    basis for the prior and the estimator and reconstructs the internal
    subspace.
    """
    if ntype not in ("E", "OTP"):
        raise ValueError(f"ntype must be 'E' or 'OTP', got {ntype!r}")
    S, pS, _, n_use_in, _ = compute_basis(
        raw,
        origin=(0.0, 0.0, 0.0),
        coord_frame="meg",
        regularize=None,
    )
    if ntype == "E":
        S_use, pS_use = S[:, :n_use_in], pS[:n_use_in]
        scope = "internal"
    else:
        S_use, pS_use = S, pS
        scope = "full"

    ours, _, _ = foster_operator(
        S_use,
        pS_use,
        N,
        raw.get_data(picks="meg"),
        prior="foster",
        affine=True,
        basis_scope=scope,
        n_use_in=n_use_in,
    )
    reference = reference_inverse(raw, N, ntype)
    if reference.shape != ours.shape:
        raise RuntimeError(
            f"Reference shape {reference.shape} does not match {ours.shape}"
        )
    diff = ours - reference
    denom = np.linalg.norm(reference)
    rel_error = np.linalg.norm(diff) / denom if denom > 0 else np.inf
    return {
        "our": ours,
        "reference": reference,
        "max_abs_diff": float(np.max(np.abs(diff))),
        "rel_error": float(rel_error),
        "pearson": float(np.nanmean(pearson(ours, reference))),
        "n_channels": int(ours.shape[0]),
        "n_times": int(ours.shape[1]),
        "n_use_in": int(n_use_in),
        "basis_scope": scope,
    }
