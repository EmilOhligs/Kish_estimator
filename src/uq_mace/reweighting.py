"""Thermodynamic reweighting and effective sample size (N_eff).

    <A>_DFT ~ sum_i A_i * w_i / sum_i w_i
    w_i = exp(-beta * (E_DFT(R_i) - E_MACE(R_i)))
    N_eff = (sum_i w_i)^2 / sum_i w_i^2
"""
from __future__ import annotations

import numpy as np


def reweighting_weights(e_dft: np.ndarray, e_model: np.ndarray, beta: float) -> np.ndarray:
    """Unnormalized reweighting weights for a set of configurations."""
    delta_e = e_dft - e_model
    delta_e = delta_e - delta_e.min()  # numerical stability, does not change N_eff
    return np.exp(-beta * delta_e)


def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = (sum w)^2 / sum(w^2). Ranges from 1 (single dominant sample) to N."""
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def reweighted_average(observable: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(observable * weights) / np.sum(weights))


# ---------------------------------------------------------------------------
# Task 1 (H0): N_eff prediction and reweighting diagnostics
# ---------------------------------------------------------------------------

def predicted_neff_gauss(var_dE: float, beta: float, n: int) -> float:
    """Gaussian-approximation prediction of the effective sample size.

    N_eff = n * exp(-beta^2 * Var(dE)) for dE = E_DFT - E_model ~ Normal.
    Valid for beta * std(dE) up to ~1; beyond that heavy tails dominate and the
    approximation OVERestimates N_eff (check psis_khat!).

    var_dE in eV^2 (extensive, per frame), beta in 1/eV, n = number of frames.
    """
    import numpy as np

    return float(n * np.exp(-(beta ** 2) * var_dE))


def _gpd_fit_khat(x) -> float:
    """Zhang & Stephens (2009) posterior-mean estimate of the generalized-Pareto
    shape parameter k for exceedances x (1D, > 0, any order).

    Follows the profile-likelihood/quadrature scheme also used by arviz/loo
    (Vehtari et al., JMLR 25, 2024). Returns k_hat (larger = heavier tail).
    """
    import numpy as np

    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n < 5 or x[-1] <= 0:
        return float("nan")
    prior_bs, prior_k = 3.0, 10.0
    m_est = 30 + int(np.sqrt(n))
    bs = 1.0 - np.sqrt(m_est / (np.arange(1, m_est + 1) - 0.5))
    bs /= prior_bs * x[int(n / 4 + 0.5) - 1]
    bs += 1.0 / x[-1]
    ks = np.log1p(-bs[:, None] * x[None, :]).mean(axis=1)  # (m_est,)
    logl = n * (np.log(-bs / ks) - ks - 1.0)
    w = 1.0 / np.exp(logl - logl[:, None]).sum(axis=1)
    w /= w.sum()
    b_post = (bs * w).sum()
    k_post = np.log1p(-b_post * x).mean()
    # weakly-informative prior towards k=0.5 (Vehtari et al.)
    return float((n * k_post + prior_k * 0.5) / (n + prior_k))


def psis_khat(weights) -> float:
    """Pareto-k diagnostic for importance weights (Vehtari et al., JMLR 25, 2024).

    Fits a generalized Pareto distribution to the largest weights and returns the
    shape parameter k_hat. Rule of thumb: k_hat < min(1 - 1/log10(S), 0.7) ->
    weight-based estimates (incl. Kish N_eff) are reliable; k_hat > 0.7 ->
    UNRELIABLE no matter how good Kish N_eff looks.

    weights: unnormalized importance weights (> 0), shape (S,).
    """
    import numpy as np

    w = np.asarray(weights, dtype=float)
    w = w[w > 0]
    s = w.size
    if s < 25:
        return float("nan")
    n_tail = int(min(0.2 * s, 3.0 * np.sqrt(s)))
    if n_tail < 5:
        return float("nan")
    w_sorted = np.sort(w)
    cutoff = w_sorted[-n_tail - 1]
    exceed = w_sorted[-n_tail:] - cutoff
    exceed = exceed[exceed > 0]
    if exceed.size < 5:
        return 0.0  # tail is (near-)degenerate -> effectively light
    return _gpd_fit_khat(exceed)


def khat_threshold(n_samples: int) -> float:
    """Sample-size-dependent reliability threshold for psis_khat
    (Vehtari et al. 2024): k_hat below this -> reliable."""
    import numpy as np

    return float(min(1.0 - 1.0 / np.log10(max(n_samples, 11)), 0.7))


def sample_overlap(weights) -> float:
    """Simple distribution-free sample-space overlap between the sampled ensemble
    (uniform weights) and the reweighted target ensemble, in the spirit of the
    phase-space overlap measures of Wu & Kofke (JCP 123, 084109, 2005):

        OVL = sum_i min(1/N, w_i / sum(w))   in (0, 1].

    1 = identical ensembles; -> 0 when a few frames dominate (overlap collapse).
    """
    import numpy as np

    w = np.asarray(weights, dtype=float)
    p = w / w.sum()
    return float(np.minimum(p, 1.0 / w.size).sum())


def neff_leave_one_out(energies, beta: float) -> float:
    """Task 2 (H8): DFT-freie N_eff-Prognose aus der Inter-Member-Streuung.

    Idee (hypothesen_review_runde2 §1.2): Member m spielt "Pseudo-DFT" gegen den
    Mittelwert der uebrigen. Fuer dE_loo = E_m - mean(E_andere) gilt bei rein
    epistemischer, unkorrelierter Streuung Var(dE_loo) = s_eps^2 * (1 + 1/(M-1)).
    Die Zielgroesse Var(E_DFT - E_mean) = s_eps^2 * (1 + 1/M), falls sich E_DFT wie
    ein weiteres Member verhaelt. Blindfleck: gemeinsamer Bias aller Member ist
    unsichtbar -> der Gap zur Kish-Messung MISST den Bias.

    energies: (M, F) Member-Gesamtenergien in eV (M >= 2). Rueckgabe: N_eff (absolut).
    """
    import numpy as np

    E = np.asarray(energies, dtype=float)
    m, n = E.shape
    if m < 2:
        return float("nan")
    var_eps = 0.0
    for i in range(m):
        others = np.delete(E, i, axis=0).mean(axis=0)
        d = E[i] - others
        var_eps += np.var(d, ddof=1) / (1.0 + 1.0 / (m - 1))
    var_eps /= m
    var_pred = var_eps * (1.0 + 1.0 / m)
    return float(n * np.exp(-(beta ** 2) * var_pred))
