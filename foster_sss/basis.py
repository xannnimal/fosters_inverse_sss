"""SSS basis construction shared by all methods."""

from __future__ import annotations

import mne


def compute_basis(
    raw,
    *,
    origin="auto",
    int_order=8,
    ext_order=3,
    regularize="in",
    coord_frame=None,
    ignore_ref=True,
    mag_scale=100.0,
):
    """Compute the SSS basis used by every method.

    Parameters
    ----------
    raw : mne.io.BaseRaw
        Raw data with MEG channels and valid sensor geometry.
    origin : "auto" or tuple of float
        SSS expansion origin.
    int_order : int
        Internal expansion order.
    ext_order : int
        External expansion order.
    regularize : "in" | "in_argmax" | None
        Regularization passed to MNE-Python.
    coord_frame : "head" | "meg" | None
        Coordinate frame used to interpret ``origin``.  ``None`` selects
        ``"head"`` when the raw has a device-to-head transform and ``"meg"``
        otherwise.
    ignore_ref : bool
        Ignore reference channels when computing the basis.
    mag_scale : float
        Magnetometer scaling.

    Returns
    -------
    S : ndarray, shape (n_ch, n_mom)
        SSS basis.
    pS : ndarray, shape (n_mom, n_ch)
        Least-squares pseudoinverse of ``S``.
    reg_moments : ndarray
        Moment indices kept by the regularization.
    n_use_in : int
        Number of internal moments.
    coord_frame : str
        Coordinate frame used by the basis.
    """
    if raw.info["bads"]:
        raise ValueError(
            "Drop bad channels before computing the basis; got "
            f"{raw.info['bads']}"
        )
    if coord_frame is None:
        coord_frame = "head" if raw.info["dev_head_t"] is not None else "meg"
    if coord_frame not in ("head", "meg"):
        raise ValueError(
            f"coord_frame must be 'head' or 'meg', got {coord_frame!r}"
        )
    S, pS, reg_moments, n_use_in = mne.preprocessing.compute_maxwell_basis(
        raw.info,
        origin=origin,
        int_order=int_order,
        ext_order=ext_order,
        calibration=None,
        coord_frame=coord_frame,
        regularize=regularize,
        ignore_ref=ignore_ref,
        bad_condition="error",
        mag_scale=mag_scale,
        extended_proj=(),
        verbose=False,
    )
    return S, pS, reg_moments, n_use_in, coord_frame
