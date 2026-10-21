"""End-to-end comparison pipeline."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

import mne

from .basis import compute_basis
from .data import align_channels, load_raw, save_raw
from .metrics import lf_band_power, pearson, plot_psd
from .methods import foster_operator, standard_sss
from .noise import compute_noise_cov
from .reference import check_against_reference


@dataclass
class ComparisonConfig:
    """Configuration for one comparison run."""

    raw: str
    out: str = "results"
    method: str = "both"
    noise_raw: str | None = None
    noise_cov: str | None = None
    clean: str | None = None
    tmin: float | None = None
    tmax: float | None = None
    noise_tmin: float | None = None
    noise_tmax: float | None = None
    l_freq: float | None = None
    h_freq: float | None = None
    noise_type: str = "E"
    regularize: str | None = "in"
    origin: str | tuple[float, float, float] = "auto"
    coord_frame: str | None = None
    basis_scope: str = "internal"
    int_order: int = 8
    ext_order: int = 3
    duration: float = 10.0
    save_fif: bool = False
    check_reference: bool = False


def selected_priors(method):
    """Return the priors selected by ``--method``."""
    if method == "both":
        return ("foster", "noise_weighted")
    if method in ("foster", "noise_weighted"):
        return (method,)
    return ()


def _load_noise(config, raw):
    if config.noise_cov is not None:
        cov = mne.read_cov(config.noise_cov)
        cov = mne.pick_channels_cov(
            cov, raw.ch_names, exclude=[], ordered=True, copy=True
        )
        N = np.asarray(cov["data"])
        if N.shape != (len(raw.ch_names), len(raw.ch_names)):
            raise RuntimeError(
                f"noise covariance has shape {N.shape}, expected "
                f"({len(raw.ch_names)}, {len(raw.ch_names)})"
            )
        return N

    if config.noise_raw is not None:
        noise_raw = align_channels(
            load_raw(config.noise_raw, config.l_freq, config.h_freq), raw
        )
    else:
        noise_raw = raw
    return compute_noise_cov(
        noise_raw,
        noise_type=config.noise_type,
        tmin=config.noise_tmin,
        tmax=config.noise_tmax,
        duration=config.duration,
    )


def _evaluate_method(
    S_in, pS_in, N, phi, *, prior, basis_scope="internal", n_use_in=None
):
    try:
        recon, _, _ = foster_operator(
            S_in,
            pS_in,
            N,
            phi,
            prior=prior,
            affine=True,
            basis_scope=basis_scope,
            n_use_in=n_use_in,
        )
    except ValueError as exc:
        if "degenerate" not in str(exc):
            raise
        return None, "degenerate", str(exc)
    return recon, "ok", ""


def run_comparison(config):
    """Run standard SSS and the selected Foster variants.

    Parameters
    ----------
    config : ComparisonConfig
        Run configuration.

    Returns
    -------
    result : dict
        Keys ``results`` (method name to reconstructed data), ``rows``
        (metric dictionaries), ``out_dir``, and ``reference_check``
        (validation result against the reference implementation, or
        ``None``).
    """
    if config.method not in ("both", "foster", "noise_weighted", "sss"):
        raise ValueError(f"Unknown method {config.method!r}")
    if config.noise_cov is not None and (
        config.noise_tmin is not None or config.noise_tmax is not None
    ):
        raise ValueError("noise_tmin/noise_tmax cannot be used with noise_cov")

    raw = load_raw(config.raw, config.l_freq, config.h_freq)
    N = _load_noise(config, raw)
    if config.tmin is not None or config.tmax is not None:
        raw.crop(tmin=config.tmin, tmax=config.tmax)

    try:
        S, pS, reg_moments, n_use_in, coord_frame = compute_basis(
            raw,
            origin=config.origin,
            int_order=config.int_order,
            ext_order=config.ext_order,
            regularize=config.regularize,
            coord_frame=config.coord_frame,
        )
    except RuntimeError as exc:
        if config.regularize is None:
            raise RuntimeError(
                "The SSS basis with regularize=None is ill-conditioned for "
                "this file; use regularize='in' (the MNE-Python default)."
            ) from exc
        raise

    S_in = S[:, :n_use_in]
    pS_in = pS[:n_use_in]
    if config.basis_scope == "full":
        S_use, pS_use = S, pS
    elif config.basis_scope == "internal":
        S_use, pS_use = S_in, pS_in
    else:
        raise ValueError(
            f"basis_scope must be 'internal' or 'full', got "
            f"{config.basis_scope!r}"
        )
    phi = raw.get_data()
    if phi.shape[0] != S_in.shape[0]:
        raise RuntimeError(
            f"Data has {phi.shape[0]} channels but the basis has "
            f"{S_in.shape[0]} rows"
        )

    results = {"sss": standard_sss(S_in, pS_in, phi)}
    statuses = {"sss": ("ok", "")}
    for prior in selected_priors(config.method):
        recon, status, note = _evaluate_method(
            S_use,
            pS_use,
            N,
            phi,
            prior=prior,
            basis_scope=config.basis_scope,
            n_use_in=n_use_in,
        )
        statuses[prior] = (status, note)
        if recon is not None:
            results[prior] = recon

    clean_data = None
    if config.clean is not None:
        clean_raw = align_channels(
            load_raw(config.clean, config.l_freq, config.h_freq), raw
        )
        if config.tmin is not None or config.tmax is not None:
            clean_raw.crop(tmin=config.tmin, tmax=config.tmax)
        clean_data = clean_raw.get_data()
        if clean_data.shape != phi.shape:
            raise RuntimeError(
                f"Clean data shape {clean_data.shape} does not match "
                f"data shape {phi.shape}"
            )
        clean_data = standard_sss(S_in, pS_in, clean_data)

    raw_lf = lf_band_power(phi, raw.info["sfreq"])
    rows = []
    for name, (status, note) in statuses.items():
        if name not in results:
            rows.append(
                {
                    "method": name,
                    "lf_gain": "",
                    "lf_power": "",
                    "r_vs_sss": "",
                    "r_vs_clean": "",
                    "status": status,
                    "note": note,
                }
            )
            continue
        data = results[name]
        lf_power = lf_band_power(data, raw.info["sfreq"])
        row = {
            "method": name,
            "lf_power": lf_power,
            "lf_gain": raw_lf / lf_power,
            "r_vs_sss": np.nanmean(pearson(data, results["sss"])),
            "r_vs_clean": "",
            "status": status,
            "note": note,
        }
        if clean_data is not None:
            row["r_vs_clean"] = np.nanmean(pearson(data, clean_data))
        rows.append(row)

    out = Path(config.out)
    out.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "lf_gain",
        "lf_power",
        "r_vs_sss",
        "r_vs_clean",
        "status",
        "note",
    ]
    with (out / "metrics.csv").open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        f"basis: S={S.shape}, n_use_in={n_use_in}, "
        f"coord_frame={coord_frame}, scope={config.basis_scope}, N={N.shape}"
    )
    header = f"{'method':>23} | {'LF gain':>8} | {'r vs SSS':>9} | status"
    if clean_data is not None:
        header = header.replace(" | status", f" | {'r vs clean':>10} | status")
    print(header)
    for row in rows:
        if row["status"] != "ok":
            line = f"{row['method']:>23} | {'n/a':>8} | {'n/a':>9}"
            if clean_data is not None:
                line += f" | {'n/a':>10}"
            line += f" | {row['status']}"
        else:
            line = (
                f"{row['method']:>23} | {row['lf_gain']:8.2f} | "
                f"{row['r_vs_sss']:+9.3f}"
            )
            if clean_data is not None:
                line += f" | {row['r_vs_clean']:+10.3f}"
            line += f" | {row['status']}"
        print(line)

    plot_psd(results, raw.info["sfreq"], out / "psd.png")
    if config.save_fif:
        for name, data in results.items():
            save_raw(data, raw.info, out / f"{name}_meg.fif")
    print(f"wrote {out / 'metrics.csv'} and {out / 'psd.png'}")
    reference_check = None
    if config.check_reference:
        reference_check = _run_reference_check(raw, N, config, out)
    return {
        "results": results,
        "rows": rows,
        "out_dir": out,
        "reference_check": reference_check,
    }


def _run_reference_check(raw, N, config, out):
    """Validate our foster variant against the reference implementation."""
    print(
        "reference check: using the reference implementation's basis settings "
        "(origin=0,0,0, coord_frame=meg, regularize=None)"
    )
    fields = [
        "status",
        "basis_scope",
        "max_abs_diff",
        "rel_error",
        "pearson",
        "n_channels",
        "n_times",
        "n_use_in",
        "note",
    ]
    try:
        result = check_against_reference(raw, N, config.noise_type)
    except Exception as exc:
        row = {
            "status": "error",
            "basis_scope": "",
            "max_abs_diff": "",
            "rel_error": "",
            "pearson": "",
            "n_channels": "",
            "n_times": "",
            "n_use_in": "",
            "note": f"{type(exc).__name__}: {exc}",
        }
        print(f"reference check failed: {row['note']}")
    else:
        row = {
            "status": "ok",
            "basis_scope": result["basis_scope"],
            "max_abs_diff": result["max_abs_diff"],
            "rel_error": result["rel_error"],
            "pearson": result["pearson"],
            "n_channels": result["n_channels"],
            "n_times": result["n_times"],
            "n_use_in": result["n_use_in"],
            "note": "",
        }
        print(
            f"reference check: max|diff|={result['max_abs_diff']:.3e}, "
            f"rel={result['rel_error']:.3e}, "
            f"r={result['pearson']:.12f}, scope={result['basis_scope']}"
        )
    with (out / "reference_check.csv").open("w", newline="") as fid:
        writer = csv.DictWriter(fid, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    print(f"wrote {out / 'reference_check.csv'}")
    return row
