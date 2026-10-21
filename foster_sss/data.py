"""Raw data loading and channel alignment."""

from __future__ import annotations

from pathlib import Path

import mne


def _read_raw(path):
    path = str(path)
    suffix = "".join(Path(path).suffixes).lower()
    if suffix.endswith(".fif") or suffix.endswith(".fif.gz"):
        return mne.io.read_raw_fif(path, preload=True, allow_maxshield=True)
    if suffix.endswith(".edf"):
        return mne.io.read_raw_edf(path, preload=True)
    if suffix.endswith(".bdf"):
        return mne.io.read_raw_bdf(path, preload=True)
    if suffix.endswith(".vhdr"):
        return mne.io.read_raw_brainvision(path, preload=True)
    if suffix.endswith(".set"):
        return mne.io.read_raw_eeglab(path, preload=True)
    if suffix.endswith(".cnt"):
        return mne.io.read_raw_cnt(path, preload=True)
    if suffix.endswith(".ds"):
        return mne.io.read_raw_ctf(path, preload=True)
    raise ValueError(
        f"Unsupported file type {suffix!r} for {path!r}. "
        "Supported: .fif, .fif.gz, .edf, .bdf, .vhdr, .set, .cnt, .ds."
    )


def load_raw(path, l_freq=None, h_freq=None):
    """Load a raw file, keep MEG channels, and optionally filter it."""
    raw = _read_raw(path)
    raw.pick_types(meg=True, eeg=False, eog=False, stim=False, exclude="bads")
    raw.load_data()
    if not raw.ch_names:
        raise ValueError(
            f"No MEG channels found in {path!r}. SSS requires MEG sensors."
        )
    if raw.info["bads"]:
        raise ValueError(f"Bad channels remain in {path}: {raw.info['bads']}")
    if l_freq is not None or h_freq is not None:
        raw.filter(l_freq, h_freq, verbose=False)
    return raw


def align_channels(raw, reference):
    """Reorder or subset ``raw`` channels to match ``reference``."""
    if raw.ch_names != reference.ch_names:
        missing = sorted(set(reference.ch_names) - set(raw.ch_names))
        if missing:
            raise ValueError(
                "Recordings are missing channels from the reference: "
                + ", ".join(missing)
            )
        raw = raw.copy().pick(reference.ch_names, verbose=False)
    return raw


def save_raw(data, info, path):
    """Save an ndarray as a FIF file."""
    raw = mne.io.RawArray(data, info.copy(), verbose=False)
    raw.save(path, overwrite=True, verbose=False)
