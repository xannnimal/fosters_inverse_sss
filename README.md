# Foster's Inverse for SSS

Comparison tools for two noise-weighted reconstructions of the internal SSS
subspace:

- **Foster's inverse** (`foster`): the original estimator with the raw moment
  covariance as prior, as in `fosters_inverse_sss.py`.
- **Noise-weighted SSS** (`noise_weighted`): the MNE-Python variant, which
  de-biases the moment prior under the sensor noise model, projects it onto
  the PSD cone, and normalizes it to unit trace.

Both methods use the same SSS basis, the same sensor noise covariance `N`, and
the same affine term. Only the moment prior `A` changes.

The reference `fosters_inverse_sss.py` is **not modified**.

Reference: McPherson et al., *Noise Optimization of Basic Signal Component
Extraction for Cryogenic and On-Scalp Magnetoencephalography (MEG)*, 2026.

## Project structure

```text
foster_sss/                 Python package
  basis.py                  SSS basis construction
  data.py                   Raw loading and channel alignment
  methods.py                standard SSS, Foster's inverse, noise-weighted SSS
  noise.py                  Noise covariance estimation
  metrics.py                PSD, band power, correlation, plots
  pipeline.py               End-to-end comparison
  reference.py              Validation against the reference implementation
  cli.py                    Command-line interface
main.py                     Entry point
tests/                      Unit tests
docs/comparison.md          Method equations and example results
data/                       Put your recordings here (git-ignored)
results/                    Default output directory (git-ignored)
```

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

This project also needs an MNE-Python version that exposes
`mne.preprocessing.compute_maxwell_basis`. The `noise_weighted` method is
validated against `mne.preprocessing.maxwell._foster_inverse`.

## Quick start

Run the comparison on any MEG recording:

```bash
python main.py --raw path/to/recording.fif --out results/run1
```

Run only one estimator:

```bash
python main.py --raw recording.fif --method foster --out results/foster
python main.py --raw recording.fif --method noise_weighted --out results/nw
```

Estimate `N` from a separate empty-room recording and compute fidelity against
a clean recording:

```bash
python main.py \
  --raw recording.fif \
  --noise-raw empty_room.fif \
  --noise-tmin 0 --noise-tmax 60 \
  --clean clean_recording.fif \
  --out results/empty_room
```

Use a saved covariance:

```bash
python main.py --raw recording.fif --noise-cov noise-cov.fif --out results/cov
```

Save one FIF per method:

```bash
python main.py --raw recording.fif --save-fif --out results/with_fif
```

The same CLI is available as a module:

```bash
python -m foster_sss --raw recording.fif
```

Show all options:

```bash
python main.py --help
```

## Comparing against the reference implementation

`--reference-compat` reproduces the reference implementation's hard-coded
basis settings:
`origin=(0, 0, 0)`, `coord_frame="meg"`, `regularize=None`, and the full
moment space when `--noise-type OTP` is selected. The preset overrides
`--origin`, `--coord-frame`, `--regularize`, and `--basis-scope`.
`--check-reference` runs the untouched `fosters_inverse_sss.py` on the same
data and `N`, and writes `reference_check.csv`:

```bash
python main.py \
  --raw recording.fif \
  --tmin 0 --tmax 4 \
  --noise-tmin 0 --noise-tmax 4 \
  --reference-compat --check-reference \
  --out results/reference_check
```

The check calls the reference implementation's `_do_inverse` directly with
the same `N` (use `--noise-tmin/--noise-tmax` to mirror its covariance window).
On the testing fixture it agrees to floating-point precision:

| `ntype` | basis scope | max abs diff | relative error | Pearson r |
|---|---|---:|---:|---:|
| `E` | internal | 1.53e-23 | 6.60e-14 | 1.000000000000 |
| `OTP` | full | 4.56e-23 | 5.99e-15 | 1.000000000000 |

If the reference implementation raises an ill-conditioned basis error with
`regularize=None`, `reference_check.csv` records `status=error` and the
message. That is the same failure the reference implementation would produce
for that file.

Use `--coord-frame head|meg|auto` and `--basis-scope internal|full` directly if
you need those settings without the full preset. Reconstruction is always
limited to the internal subspace.

## Using other data

`--raw`, `--noise-raw`, and `--clean` accept the following formats:

| Extension | Reader |
|---|---|
| `.fif`, `.fif.gz` | `mne.io.read_raw_fif` |
| `.edf` | `mne.io.read_raw_edf` |
| `.bdf` | `mne.io.read_raw_bdf` |
| `.vhdr` | `mne.io.read_raw_brainvision` |
| `.set` | `mne.io.read_raw_eeglab` |
| `.cnt` | `mne.io.read_raw_cnt` |
| `.ds` | `mne.io.read_raw_ctf` |

The recording must contain **MEG channels**; the loader keeps only
`meg=True` and drops bad channels. For SSS, the sensors must have valid
geometry and the file should contain a device-to-head transform when one is
available. If your file uses another MNE reader, add its extension to
`foster_sss/data.py` and return an `mne.io.BaseRaw`.

Recommended workflow for a new dataset:

1. Place or symlink the files under `data/`.
2. Inspect channels and bads in MNE-Python first.
3. Choose whether `N` comes from the same recording, an empty room, or a saved
   covariance.
4. Run `python main.py ... --out results/<dataset>`.
5. Compare `metrics.csv` and `psd.png`; use `--clean` if a clean reference is
   available.

## Methods

| method | moment prior `A` | noise covariance |
|---|---|---|
| `sss` | none (least squares) | none |
| `foster` | `np.cov(x_hat)` | `N` |
| `noise_weighted` | `PSD(np.cov(x_hat) - pS_in @ N @ pS_in.T) / trace(...)` | `N` |

`x_hat = pS_in @ data` are the initial internal moments. See
[docs/comparison.md](docs/comparison.md) for the equations and results.

### Why `N` is not normalized

`N` is kept in physical units because its scale relative to
`S_in @ A @ S_in.T` sets the bias-variance balance. The CLI deliberately does
not normalize it. On real MEG data, `N / trace(N)` dominates the signal term:
the raw prior collapses the reconstruction, and the de-biased prior becomes
degenerate. Normalize only `A`, as in `noise_weighted`.

## Outputs

Each run writes to `--out`:

- `metrics.csv`: `method`, `lf_gain`, `lf_power`, `r_vs_sss`, `r_vs_clean`,
  `status`, `note`.
- `psd.png`: mean PSD per method.
- `reference_check.csv`: only with `--check-reference`; agreement with the
  reference implementation (`status`, `basis_scope`, `max_abs_diff`,
  `rel_error`, `pearson`, `n_channels`, `n_times`, `n_use_in`, `note`).
- `<method>_meg.fif`: only with `--save-fif`.

`lf_gain` is the factor by which the 0.5-3 Hz power is reduced relative to the
raw data. `r_vs_sss` is the mean per-channel correlation with standard SSS.
Methods that cannot be computed are listed with `status=degenerate` and empty
metrics.

## Tests

```bash
python -m pytest tests -q
```

## Reference implementation

`fosters_inverse_sss.py` is the reference implementation and is left unchanged.
Its `__main__` example calls `fosters_inverse(raw, Ntmin, Ntmax)` without the
required `ntype` argument; this project does not use that entry point.
