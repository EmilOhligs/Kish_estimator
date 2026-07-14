"""Tests for Task-1 additions: predicted_neff_gauss, psis_khat, sample_overlap."""
import numpy as np

from uq_mace.reweighting import (
    effective_sample_size,
    khat_threshold,
    predicted_neff_gauss,
    psis_khat,
    reweighting_weights,
    sample_overlap,
)

BETA_300K = 1.0 / (8.617e-5 * 300.0)  # 38.68 / eV


def test_predicted_neff_gauss_matches_kish_for_gaussian_dE():
    # Sanity anchor (KI-Briefing §5): sd=11 meV, N=5000, 300 K -> N_eff ~ 4190
    rng = np.random.default_rng(0)
    dE = rng.normal(0.0, 0.011, 5000)
    w = np.exp(-BETA_300K * (dE - dE.min()))
    neff_kish = effective_sample_size(w)
    neff_pred = predicted_neff_gauss(np.var(dE), BETA_300K, 5000)
    assert abs(neff_pred - neff_kish) / neff_kish < 0.05
    assert 4000 < neff_kish < 4400


def test_predicted_neff_gauss_limits():
    assert predicted_neff_gauss(0.0, BETA_300K, 1000) == 1000.0
    assert predicted_neff_gauss(1.0, BETA_300K, 1000) < 1e-6


def test_psis_khat_light_tail_is_reliable():
    # small dE spread -> near-uniform weights -> light tail, k_hat clearly below 0.7
    rng = np.random.default_rng(0)
    dE = rng.normal(0.0, 0.005, 5000)
    w = reweighting_weights(dE + 1.0, np.ones_like(dE), BETA_300K)  # dE via e_dft-e_model
    k = psis_khat(w)
    assert np.isfinite(k)
    assert k < khat_threshold(5000)


def test_psis_khat_recovers_known_pareto_shape():
    # Weights with an exact Pareto(alpha) tail have GPD shape k = 1/alpha.
    # alpha = 1.25 -> true k = 0.8: the diagnostic must clearly flag (> 0.6).
    rng = np.random.default_rng(0)
    w = (1.0 / rng.uniform(size=20000)) ** (1.0 / 1.25)  # Pareto, k_true = 0.8
    k = psis_khat(w)
    assert 0.6 < k < 1.1


def test_psis_khat_exponential_tail_is_light():
    # Exponential weights: GPD shape k = 0 -> k_hat must stay well below 0.5.
    rng = np.random.default_rng(0)
    w = rng.exponential(size=20000)
    assert psis_khat(w) < 0.3


def test_psis_khat_small_sample_returns_nan():
    assert np.isnan(psis_khat(np.ones(10)))


def test_sample_overlap_limits():
    assert np.isclose(sample_overlap(np.ones(100)), 1.0)
    w = np.full(100, 1e-12)
    w[0] = 1.0
    assert sample_overlap(w) < 0.05


def test_khat_threshold_bounds():
    assert khat_threshold(100) <= 0.7
    assert 0.0 < khat_threshold(50) <= 0.7
