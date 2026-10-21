"""Command-line interface for the comparison pipeline."""

from __future__ import annotations

import argparse

from .pipeline import ComparisonConfig, run_comparison


def parse_origin(value):
    """Parse ``auto`` or ``x,y,z`` into an SSS origin."""
    value = str(value).strip()
    if value.lower() == "auto":
        return "auto"
    try:
        return tuple(float(part) for part in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"origin must be 'auto' or 'x,y,z', got {value!r}"
        ) from exc


def build_parser():
    """Build the argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Compare standard SSS, Foster's inverse, and noise-weighted SSS "
            "on the same MEG data."
        )
    )
    parser.add_argument(
        "--raw",
        required=True,
        help=(
            "Input MEG file. Supported: .fif, .fif.gz, .edf, .bdf, .vhdr, "
            ".set, .cnt, .ds."
        ),
    )
    noise_group = parser.add_mutually_exclusive_group()
    noise_group.add_argument(
        "--noise-raw",
        default=None,
        help="Optional raw file used to estimate N (e.g. empty room).",
    )
    noise_group.add_argument(
        "--noise-cov",
        default=None,
        help="Optional saved covariance used as N.",
    )
    parser.add_argument(
        "--clean", default=None, help="Optional clean raw file for fidelity."
    )
    parser.add_argument("--tmin", type=float, default=None)
    parser.add_argument("--tmax", type=float, default=None)
    parser.add_argument("--noise-tmin", type=float, default=None)
    parser.add_argument("--noise-tmax", type=float, default=None)
    parser.add_argument("--l-freq", type=float, default=None)
    parser.add_argument("--h-freq", type=float, default=None)
    parser.add_argument(
        "--noise-type", choices=("E", "OTP"), default="E", help="Noise estimate."
    )
    parser.add_argument(
        "--method",
        choices=("both", "foster", "noise_weighted", "sss"),
        default="both",
        help="Method(s) to run; 'both' runs foster and noise_weighted.",
    )
    parser.add_argument(
        "--regularize",
        choices=("in", "in_argmax", "None"),
        default="in",
        help="SSS regularization (shared by all methods).",
    )
    parser.add_argument("--origin", type=parse_origin, default="auto")
    parser.add_argument(
        "--coord-frame",
        choices=("auto", "head", "meg"),
        default="auto",
        help=(
            "Frame used to interpret --origin. 'auto' uses head when a "
            "dev_head_t is present, otherwise meg."
        ),
    )
    parser.add_argument(
        "--basis-scope",
        choices=("internal", "full"),
        default="internal",
        help=(
            "Moments used by the prior and estimator. 'full' includes "
            "external moments; reconstruction is always internal."
        ),
    )
    parser.add_argument(
        "--reference-compat",
        action="store_true",
        help=(
            "Use the reference implementation's settings: origin=0,0,0, "
            "coord_frame=meg, regularize=None, and full moments for OTP."
        ),
    )
    parser.add_argument(
        "--check-reference",
        action="store_true",
        help=(
            "Validate the foster variant against fosters_inverse_sss.py "
            "and write reference_check.csv."
        ),
    )
    parser.add_argument("--int-order", type=int, default=8)
    parser.add_argument("--ext-order", type=int, default=3)
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="OTP duration in seconds (only for --noise-type OTP).",
    )
    parser.add_argument(
        "--out", default="results", help="Output directory."
    )
    parser.add_argument(
        "--save-fif", action="store_true", help="Save one FIF per method."
    )
    return parser


def main(argv=None):
    """Run the comparison CLI."""
    args = build_parser().parse_args(argv)
    origin = args.origin
    coord_frame = None if args.coord_frame == "auto" else args.coord_frame
    regularize = None if args.regularize == "None" else args.regularize
    basis_scope = args.basis_scope
    if args.reference_compat:
        origin = (0.0, 0.0, 0.0)
        coord_frame = "meg"
        regularize = None
        basis_scope = "full" if args.noise_type == "OTP" else "internal"
    config = ComparisonConfig(
        raw=args.raw,
        out=args.out,
        method=args.method,
        noise_raw=args.noise_raw,
        noise_cov=args.noise_cov,
        clean=args.clean,
        tmin=args.tmin,
        tmax=args.tmax,
        noise_tmin=args.noise_tmin,
        noise_tmax=args.noise_tmax,
        l_freq=args.l_freq,
        h_freq=args.h_freq,
        noise_type=args.noise_type,
        regularize=regularize,
        origin=origin,
        coord_frame=coord_frame,
        basis_scope=basis_scope,
        int_order=args.int_order,
        ext_order=args.ext_order,
        duration=args.duration,
        save_fif=args.save_fif,
        check_reference=args.check_reference,
    )
    run_comparison(config)
    return 0
