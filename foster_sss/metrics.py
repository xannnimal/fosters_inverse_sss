"""Metrics and plots used to compare reconstructions."""

from __future__ import annotations

import numpy as np

import mne


def _n_fft(n_times):
    n_fft = min(2048, n_times)
    n_fft -= n_fft % 2
    if n_fft < 2:
        raise ValueError("At least two time samples are needed for the PSD")
    return n_fft


def lf_band_power(data, sfreq):
    """Average power in the 0.5-3 Hz band."""
    n_fft = _n_fft(data.shape[1])
    psd, freqs = mne.time_frequency.psd_array_welch(
        data,
        sfreq=sfreq,
        fmin=0.5,
        fmax=40.0,
        n_fft=n_fft,
        verbose=False,
    )
    band = (freqs >= 0.5) & (freqs <= 3.0)
    if not band.any():
        raise ValueError("The 0.5-3 Hz band is empty; provide longer data")
    integrate = getattr(np, "trapezoid", None)
    if integrate is None:
        integrate = np.trapz
    return integrate(np.mean(psd, axis=0)[band], freqs[band])


def pearson(a, b):
    """Per-channel Pearson correlation."""
    a0 = a - a.mean(axis=-1, keepdims=True)
    b0 = b - b.mean(axis=-1, keepdims=True)
    denom = np.sqrt((a0**2).sum(axis=-1) * (b0**2).sum(axis=-1))
    out = np.full(denom.shape, np.nan)
    np.divide((a0 * b0).sum(axis=-1), denom, out=out, where=denom > 0)
    return out


def plot_psd(results, sfreq, path):
    """Write a PSD comparison plot."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4))
    for name, data in results.items():
        psd, freqs = mne.time_frequency.psd_array_welch(
            data,
            sfreq=sfreq,
            fmin=0.5,
            fmax=40.0,
            n_fft=_n_fft(data.shape[1]),
            verbose=False,
        )
        ax.semilogy(freqs, np.mean(psd, axis=0), label=name)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("PSD (mean over channels)")
    ax.set_title("Foster's inverse and noise-weighted SSS")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
