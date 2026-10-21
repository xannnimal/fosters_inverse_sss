"""Foster's inverse and noise-weighted SSS estimators."""

from __future__ import annotations

import numpy as np
from scipy.linalg import eigh, pinv

PRIORS = ("foster", "noise_weighted")


def _validate_inputs(S_in, pS_in, N, data):
    S_in = np.asarray(S_in, dtype=float)
    pS_in = np.asarray(pS_in, dtype=float)
    N = np.asarray(N, dtype=float)
    data = np.asarray(data, dtype=float)
    if S_in.ndim != 2:
        raise ValueError(f"S_in must be 2D, got shape {S_in.shape}")
    if pS_in.shape != (S_in.shape[1], S_in.shape[0]):
        raise ValueError(
            f"pS_in must have shape {(S_in.shape[1], S_in.shape[0])}, "
            f"got {pS_in.shape}"
        )
    if N.shape != (S_in.shape[0], S_in.shape[0]):
        raise ValueError(
            f"N must have shape {(S_in.shape[0], S_in.shape[0])}, got {N.shape}"
        )
    if data.ndim != 2 or data.shape[0] != S_in.shape[0]:
        raise ValueError(
            f"data must have shape {(S_in.shape[0], 'n_times')}, got {data.shape}"
        )
    if data.shape[1] < 2:
        raise ValueError("At least two time samples are needed to estimate a covariance")
    return S_in, pS_in, N, data


def foster_prior(x_hat, pS_in, N, *, prior="noise_weighted", normalize=True, psd=True):
    """Estimate the moment prior ``A``.

    Parameters
    ----------
    x_hat : ndarray, shape (n_mom, n_times)
        Initial (least-squares) internal moments.
    pS_in : ndarray, shape (n_mom, n_ch)
        Least-squares pseudoinverse of the internal basis.
    N : ndarray, shape (n_ch, n_ch)
        Sensor noise covariance in physical units.  Its physical scale is
        preserved because it determines the bias-variance balance.
    prior : "noise_weighted" | "foster"
        ``"noise_weighted"`` removes the noise contribution from the data
        covariance, projects onto the PSD cone, and optionally normalizes the
        trace.  ``"foster"`` uses the raw covariance, as in the original
        script.
    normalize : bool
        Normalize ``A`` to unit trace (``"noise_weighted"`` only).
    psd : bool
        Project the de-biased covariance onto the PSD cone
        (``"noise_weighted"`` only).

    Returns
    -------
    A : ndarray, shape (n_mom, n_mom)
        Moment prior.
    """
    if prior not in PRIORS:
        raise ValueError(
            f"prior must be 'noise_weighted' or 'foster', got {prior!r}"
        )
    moments_cov = np.cov(x_hat)
    if prior == "foster":
        return moments_cov
    X = (moments_cov + moments_cov.T) / 2 - pS_in @ N @ pS_in.T
    if psd:
        eigenvalues, eigenvectors = eigh((X + X.T) / 2)
        X = (eigenvectors * np.clip(eigenvalues, 0, None)) @ eigenvectors.T
    trace = np.trace(X)
    if not np.isfinite(trace) or trace <= 0:
        raise ValueError(
            "The Foster's inverse moment prior is degenerate (trace <= 0); "
            "the noise covariance is too large relative to the data covariance."
        )
    return X / trace if normalize else X


def foster_operator(
    S_in,
    pS_in,
    N,
    data,
    *,
    prior="noise_weighted",
    normalize=True,
    psd=True,
    affine=True,
    basis_scope="internal",
    n_use_in=None,
):
    """Apply Foster's inverse.

    Parameters
    ----------
    S_in : ndarray, shape (n_ch, n_mom)
        Internal SSS basis, or the full (internal + external) basis when
        ``basis_scope="full"``.
    pS_in : ndarray, shape (n_mom, n_ch)
        Least-squares pseudoinverse of ``S_in``.
    N : ndarray, shape (n_ch, n_ch)
        Sensor noise covariance in physical units.
    data : ndarray, shape (n_ch, n_times)
        Sensor data of the segment.
    prior : "noise_weighted" | "foster"
        Prior variant.  ``"noise_weighted"`` is the de-biased, PSD-projected
        and trace-normalized prior; ``"foster"`` is the original raw
        covariance.
    normalize : bool
        Normalize the ``"noise_weighted"`` prior to unit trace.
    psd : bool
        Project the ``"noise_weighted"`` prior onto the PSD cone.
    affine : bool
        Add the affine term that preserves the mean moments.
    basis_scope : "internal" | "full"
        ``"internal"`` uses only internal moments for the prior and the
        estimator.  ``"full"`` builds the prior and the estimator in the
        full moment space and reconstructs the internal subspace, matching
        the reference implementation's ``ntype="OTP"`` path.
    n_use_in : int | None
        Number of internal moments.  Required when ``basis_scope="full"``.

    Returns
    -------
    recon : ndarray, shape (n_ch, n_times)
        Internal reconstruction.
    B : ndarray, shape (n_mom, n_ch)
        Noise-weighted pseudoinverse.
    b : ndarray, shape (n_mom,)
        Affine term (zeros when ``affine=False``).
    """
    S_in, pS_in, N, data = _validate_inputs(S_in, pS_in, N, data)
    if basis_scope not in ("internal", "full"):
        raise ValueError(
            "basis_scope must be 'internal' or 'full', "
            f"got {basis_scope!r}"
        )
    if basis_scope == "full":
        if n_use_in is None:
            raise ValueError("n_use_in is required when basis_scope='full'")
        if not 0 < n_use_in <= S_in.shape[1]:
            raise ValueError(
                f"n_use_in must be in [1, {S_in.shape[1]}], got {n_use_in}"
            )
        n_recon = int(n_use_in)
    else:
        n_recon = S_in.shape[1]
    x_hat = pS_in @ data
    A = foster_prior(x_hat, pS_in, N, prior=prior, normalize=normalize, psd=psd)
    B = A @ S_in.T @ pinv(S_in @ A @ S_in.T + N, check_finite=False)
    if affine:
        mean_moments = x_hat.mean(axis=1)
        b = mean_moments - B @ S_in @ mean_moments
    else:
        b = np.zeros(S_in.shape[1])
    moments = B @ data + b[:, None]
    recon = np.real(S_in[:, :n_recon] @ moments[:n_recon])
    return recon, B, b


def standard_sss(S_in, pS_in, data):
    """Least-squares internal SSS reconstruction (reference method)."""
    S_in = np.asarray(S_in, dtype=float)
    pS_in = np.asarray(pS_in, dtype=float)
    data = np.asarray(data, dtype=float)
    if S_in.ndim != 2:
        raise ValueError(f"S_in must be 2D, got shape {S_in.shape}")
    if pS_in.shape != (S_in.shape[1], S_in.shape[0]):
        raise ValueError(
            f"pS_in must have shape {(S_in.shape[1], S_in.shape[0])}, "
            f"got {pS_in.shape}"
        )
    if data.ndim != 2 or data.shape[0] != S_in.shape[0]:
        raise ValueError(
            f"data must have shape {(S_in.shape[0], 'n_times')}, got {data.shape}"
        )
    return np.real(S_in @ (pS_in @ data))
