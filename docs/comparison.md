# Comparing Foster's inverse and noise-weighted SSS

This project compares the reference `fosters_inverse_sss.py` estimator
(`foster`) with the MNE-Python integration (`noise_weighted`). The reference
implementation is **left unchanged**.

By default all methods share the same SSS basis, the same sensor data, the
same sensor noise covariance `N`, and the same affine term. Only the moment
prior `A` changes. The `--reference-compat` preset changes the basis settings
to match the reference implementation; see `Matching the reference
implementation` below.

| method | moment prior `A` | `N` |
|---|---|---|
| `sss` | none (least squares) | none |
| `foster` | `np.cov(x_hat)` (raw covariance) | `N` |
| `noise_weighted` | `PSD(np.cov(x_hat) - pS_in @ N @ pS_in.T) / trace(...)` | `N` |

`x_hat = pS_in @ data` are the initial internal moments. `N` is kept in
physical units; only `A` is normalized in `noise_weighted`.

## Equations

The common estimator is the LMMSE/Wiener inverse of the internal SSS model:

```text
phi = S_in @ x + n,    n ~ N(0, N),    x ~ N(mu, A)

B = A @ S_in.T @ (S_in @ A @ S_in.T + N)^-1
b = mu - B @ S_in @ mu
x_hat = B @ phi + b
phi_hat = S_in @ x_hat
```

with `mu = mean(pS_in @ phi)`. Standard SSS is the limit `A^-1 -> 0`.

The only difference between the two original priors is:

```text
foster:          A = Cov(x_hat)

noise_weighted:  X = Cov(x_hat) - pS_in @ N @ pS_in.T
                 X = PSD(X)
                 A = X / trace(X)
```

## Usage

Run both methods and the standard SSS reference:

```bash
python main.py --raw raw.fif --out results/run1
```

Run one method only:

```bash
python main.py --raw raw.fif --method foster --out results/foster
python main.py --raw raw.fif --method noise_weighted --out results/noise_weighted
```

Use a separate recording for `N`, a saved covariance, or a baseline window:

```bash
python main.py --raw raw.fif --noise-raw empty.fif \
    --noise-tmin 0 --noise-tmax 60 --out results/empty_room
python main.py --raw raw.fif --noise-cov empty-cov.fif --out results/cov
```

Options:

- `--noise-type E|OTP`: empirical covariance or oversampled temporal
  projection (ignored with `--noise-cov`).
- `--noise-cov`: saved covariance used as `N`; channels are reordered to match
  the raw file.
- `--regularize in|in_argmax|None`: shared SSS regularization. `None` is
  rejected when the basis is too ill-conditioned; use `in` for those files.
- `--origin auto|x,y,z`: shared SSS origin.
- `--coord-frame auto|head|meg`: frame used to interpret `--origin`. `auto`
  uses head when `dev_head_t` is present, otherwise meg.
- `--basis-scope internal|full`: moments used by the prior and estimator.
  `full` includes external moments; reconstruction is always internal.
- `--reference-compat`: preset for the reference implementation's basis
  (`origin=0,0,0`,
  `coord_frame=meg`, `regularize=None`, full moments for OTP).
- `--check-reference`: validate against `fosters_inverse_sss.py` and write
  `reference_check.csv`.
- `--tmin/--tmax`: segment used for the comparison.
- `--noise-tmin/--noise-tmax`: segment used for `N` (not with `--noise-cov`).
- `--l-freq/--h-freq`: optional filtering before the comparison.
- `--clean`: optional clean recording for a fidelity reference.
- `--save-fif`: save one FIF per method.

Each run writes `metrics.csv` and `psd.png` to the output directory, plus
`reference_check.csv` when `--check-reference` is used.

## Matching the reference implementation

The reference implementation hard-codes a different basis configuration from
the MNE-Python default:

```python
compute_maxwell_basis(
    raw.info,
    origin=(0, 0, 0),
    coord_frame="meg",
    regularize=None,
    ...
)
```

With `--reference-compat`, this package uses exactly those settings. For
`ntype="E"` the prior and estimator use the internal moments; for
`ntype="OTP"` they use the full moment space and reconstruct only the internal
subspace:

```text
E:    S_use = S[:, :n_use_in],   pS_use = pS[:n_use_in, :]
OTP:  S_use = S,                 pS_use = pS

A = Cov(pS_use @ phi)
B = A @ S_use.T @ (S_use @ A @ S_use.T + N)^-1
phi_hat = S[:, :n_use_in] @ (B @ phi + b)[:n_use_in, :]
```

`--check-reference` imports the untouched implementation, calls its
`_do_inverse` with the same data and `N`, and compares the two
reconstructions. It does not call the `fosters_inverse` wrapper, which also
computes `N` and creates Maxwell-filtered info. Use
`--noise-tmin/--noise-tmax` to mirror the wrapper's covariance window. On the
testing fixture (`test_move_anon_raw.fif`, 0-4 s, `N` from the same segment):

| `ntype` | basis scope | max abs diff | relative error | Pearson r |
|---|---|---:|---:|---:|
| `E` | internal | 1.53e-23 | 6.60e-14 | 1.000000000000 |
| `OTP` | full | 4.56e-23 | 5.99e-15 | 1.000000000000 |

The remaining difference is floating-point round-off. If the reference
implementation raises an ill-conditioned basis error, `reference_check.csv`
records `status=error` with the message; that is the same failure the
implementation would produce.

## Example on the testing fixture

`test_move_anon_raw.fif`, 0-4 s, `N` from the same 0-4 s segment, no injected
noise:

| method | LF gain (0.5-3 Hz) | r vs SSS |
|---|---:|---:|
| `sss` | 2.84 | 1.000 |
| `foster` | 42.56 | 0.017 |
| `noise_weighted` | 59.61 | 0.388 |

The raw prior (`foster`) over-suppresses when `N` is estimated from the same
clean data. The de-biased and trace-normalized prior (`noise_weighted`)
suppresses less and keeps a higher correlation with standard SSS.

## Why `N` is not normalized

This tool does not normalize `N`. Replacing it with `N / trace(N)` would remove
its physical scale, which is what sets the bias-variance balance:

```text
B = A @ S_in.T @ (S_in @ A @ S_in.T + N)^-1
```

On real MEG data, `S_in @ A @ S_in.T` is on the order of `1e-24` (T²), while
`N / trace(N)` has trace 1. The normalized noise term therefore dominates: the
raw prior collapses the reconstruction, and the de-biased prior becomes
degenerate because `pS_in @ (N/trace(N)) @ pS_in.T` is much larger than
`Cov(x_hat)`.

Conclusion: `N` stays in physical units. Normalize only `A`, as in
`noise_weighted`.

## Historical fidelity check

The following table comes from the original comparison run with an injected
OPM-like low-frequency artifact. It is kept as a reference for the
unnormalized methods:

| method | LF gain | r vs clean SSS |
|---|---:|---:|
| `sss` | 0.12 | 0.942 |
| `foster` | 133.49 | 0.016 |
| `noise_weighted` | 96.43 | 0.388 |

The original notebook (`run_foster`, without the affine term) reports
92.9 / 0.396 for the same injected artifact. The small difference comes from
the affine term and the MNE-Python version; the prior and `N` handling are the
same.

## Scope and limitations

- The comparison isolates the moment prior. It estimates the data covariance
  with `np.cov` over the selected segment; it does not chunk the data, exclude
  annotated segments, or apply cross-talk correction. The MNE-Python
  integration does those steps separately. The reference implementation also
  does not chunk, exclude annotations, or apply cross-talk correction.
- The default basis follows MNE-Python (`origin="auto"`, head frame when a
  `dev_head_t` is present). This is not the reference implementation's basis.
  Use
  `--reference-compat` for a like-for-like comparison with
  `fosters_inverse_sss.py`.
- `fosters_inverse_sss.py` is not modified. Its `__main__` example currently
  calls `fosters_inverse(raw, Ntmin, Ntmax)` without `ntype`; this project does
  not use that entry point.
- The upstream repository has no license file. This comparison code is
  provided inside this repository; do not copy the reference code elsewhere
  without permission from the copyright holder.
