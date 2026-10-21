from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import pinv

from foster_sss.methods import foster_operator, foster_prior, standard_sss

PRIORS = ("foster", "noise_weighted")


@pytest.fixture
def synthetic():
    rng = np.random.default_rng(0)
    n_ch, n_mom, n_times = 12, 5, 200
    S = rng.normal(size=(n_ch, n_mom))
    pS = np.linalg.pinv(S)
    data = rng.normal(size=(n_ch, n_times))
    noise = rng.normal(size=(n_ch, n_ch))
    N = noise @ noise.T / n_ch + 0.1 * np.eye(n_ch)
    return S, pS, N, data


def test_foster_prior_is_raw_covariance(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="foster")
    assert_allclose(A, np.cov(x_hat))


def test_noise_weighted_prior_is_trace_normalized_and_psd(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="noise_weighted", normalize=True, psd=True)
    assert_allclose(np.trace(A), 1.0)
    assert_allclose(A, A.T)
    eigvals = np.linalg.eigvalsh(A)
    assert eigvals.min() >= -1e-12


def test_noise_weighted_prior_without_normalization(synthetic):
    S, pS, N, data = synthetic
    x_hat = pS @ data
    A = foster_prior(x_hat, pS, N, prior="noise_weighted", normalize=False, psd=True)
    assert not np.isclose(np.trace(A), 1.0)


def test_affine_term_preserves_mean(synthetic):
    S, pS, N, data = synthetic
    _, B, b = foster_operator(S, pS, N, data, prior="noise_weighted", affine=True)
    mean_moments = (pS @ data).mean(axis=1)
    assert_allclose(B @ S @ mean_moments + b, mean_moments)


def test_no_affine_gives_zero_offset(synthetic):
    S, pS, N, data = synthetic
    _, _, b = foster_operator(S, pS, N, data, prior="noise_weighted", affine=False)
    assert_allclose(b, 0.0)


def test_noise_covariance_scale_matters(synthetic):
    S, pS, N, data = synthetic
    _, B1, _ = foster_operator(S, pS, N, data, prior="noise_weighted")
    _, B2, _ = foster_operator(S, pS, 2.0 * N, data, prior="noise_weighted")
    assert not np.allclose(B1, B2)


def test_shapes_and_finiteness(synthetic):
    S, pS, N, data = synthetic
    recon, B, b = foster_operator(S, pS, N, data, prior="noise_weighted")
    assert recon.shape == data.shape
    assert B.shape == (S.shape[1], S.shape[0])
    assert b.shape == (S.shape[1],)
    assert np.isfinite(recon).all()
    assert np.isfinite(B).all()
    assert np.isfinite(b).all()


def test_foster_pseudoinverse_matches_formula(synthetic):
    S, pS, N, data = synthetic
    _, B, _ = foster_operator(S, pS, N, data, prior="foster")
    A = np.cov(pS @ data)
    B_ref = A @ S.T @ pinv(S @ A @ S.T + N, check_finite=False)
    assert_allclose(B, B_ref)


def test_noise_weighted_matches_mne_python(synthetic):
    maxwell = pytest.importorskip("mne.preprocessing.maxwell")
    _foster_inverse = getattr(maxwell, "_foster_inverse", None)
    if _foster_inverse is None:
        pytest.skip("MNE-Python does not expose _foster_inverse")

    S, pS, N, data = synthetic
    _, B, _ = foster_operator(S, pS, N, data, prior="noise_weighted")
    moments_cov = pS @ np.cov(data) @ pS.T
    B_ref = _foster_inverse(S, N, moments_cov, pS)
    assert_allclose(B, B_ref, rtol=1e-6, atol=1e-8)


def test_standard_sss(synthetic):
    S, pS, N, data = synthetic
    assert_allclose(standard_sss(S, pS, data), S @ pS @ data)


def test_degenerate_prior_raises(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="degenerate"):
        foster_prior(pS @ data, pS, 1e6 * N, prior="noise_weighted")


def test_bad_shapes_raise(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="pS_in"):
        foster_operator(S, pS[:, :-1], N, data)
    with pytest.raises(ValueError, match="N must"):
        foster_operator(S, pS, N[:-1, :-1], data)
    with pytest.raises(ValueError, match="data must"):
        foster_operator(S, pS, N, data[:-1])
    with pytest.raises(ValueError, match="At least two"):
        foster_operator(S, pS, N, data[:, :1])


def test_full_scope_matches_reference_formula(synthetic):
    S, pS, N, data = synthetic
    n_use_in = 3
    recon, B, b = foster_operator(
        S,
        pS,
        N,
        data,
        prior="foster",
        basis_scope="full",
        n_use_in=n_use_in,
    )
    x_hat = pS @ data
    A = np.cov(x_hat)
    B_ref = A @ S.T @ pinv(S @ A @ S.T + N, check_finite=False)
    mean_moments = x_hat.mean(axis=1)
    b_ref = mean_moments - B_ref @ S @ mean_moments
    moments = B_ref @ data + b_ref[:, None]
    recon_ref = np.real(S[:, :n_use_in] @ moments[:n_use_in])
    assert_allclose(B, B_ref)
    assert_allclose(b, b_ref)
    assert_allclose(recon, recon_ref)


def test_full_scope_reconstructs_internal_only(synthetic):
    S, pS, N, data = synthetic
    recon, B, _ = foster_operator(
        S, pS, N, data, prior="foster", basis_scope="full", n_use_in=2
    )
    assert recon.shape == data.shape
    assert B.shape == (S.shape[1], S.shape[0])


def test_full_scope_requires_valid_n_use_in(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="n_use_in"):
        foster_operator(S, pS, N, data, basis_scope="full")
    with pytest.raises(ValueError, match="n_use_in"):
        foster_operator(S, pS, N, data, basis_scope="full", n_use_in=99)


def test_invalid_basis_scope_raises(synthetic):
    S, pS, N, data = synthetic
    with pytest.raises(ValueError, match="basis_scope"):
        foster_operator(S, pS, N, data, basis_scope="external")


def test_reference_script_is_available():
    from foster_sss.reference import _load_reference

    assert hasattr(_load_reference(), "_do_inverse")


def test_check_against_reference_on_testing_fixture():
    path = (
        Path.home()
        / "mne_data"
        / "MNE-testing-data"
        / "SSS"
        / "test_move_anon_raw.fif"
    )
    if not path.exists():
        pytest.skip("MNE testing-data fixture is not available")
    from foster_sss.data import load_raw
    from foster_sss.noise import compute_noise_cov
    from foster_sss.reference import check_against_reference

    raw = load_raw(path)
    raw.crop(tmin=0.0, tmax=4.0)
    N = compute_noise_cov(raw, noise_type="E", tmin=0.0, tmax=4.0)
    for ntype in ("E", "OTP"):
        result = check_against_reference(raw, N, ntype)
        assert result["rel_error"] < 1e-10
        assert result["pearson"] > 1.0 - 1e-10
