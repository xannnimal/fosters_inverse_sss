"""Sensor noise covariance estimation."""

from __future__ import annotations

import numpy as np

import mne


def compute_noise_cov(
    raw,
    *,
    noise_type="E",
    tmin=None,
    tmax=None,
    duration=10.0,
):
    """Compute the sensor noise covariance.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Recording used to estimate the noise covariance.
    noise_type : "E" | "OTP"
        Empirical covariance or oversampled temporal projection.
    tmin, tmax : float | None
        Time window used with ``noise_type="E"``.
    duration : float
        OTP window duration in seconds.

    Returns
    -------
    N : ndarray, shape (n_ch, n_ch)
        Noise covariance in physical units, not normalized.
    """
    if noise_type not in ("E", "OTP"):
        raise ValueError(f"noise_type must be 'E' or 'OTP', got {noise_type!r}")
    raw_meg = raw.copy().pick_types(
        meg=True, eeg=False, eog=False, stim=False, exclude="bads"
    )
    if raw_meg.info["bads"]:
        raise ValueError(
            "Drop bad channels before computing the noise covariance; got "
            f"{raw_meg.info['bads']}"
        )
    if noise_type == "E":
        cov = mne.compute_raw_covariance(
            raw_meg,
            tmin=tmin,
            tmax=tmax,
            rank="info",
            method="empirical",
            verbose=False,
        )
        return np.asarray(cov["data"])

    import sklearn.covariance

    raw_otp = mne.preprocessing.oversampled_temporal_projection(
        raw_meg, duration=duration, verbose=False
    )
    diff = raw_meg.get_data(picks="meg") - raw_otp.get_data(picks="meg")
    return sklearn.covariance.empirical_covariance(diff.T)
