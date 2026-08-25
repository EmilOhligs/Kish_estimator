"""Thermodynamic reweighting and effective sample size (N_eff).

    w_i = exp(-beta * (E_DFT(R_i) - E_MACE(R_i)))
    N_eff = (sum_i w_i)^2 / sum_i w_i^2
"""
from __future__ import annotations

import numpy as np


def reweighting_weights(e_dft: np.ndarray, e_model: np.ndarray, beta: float) -> np.ndarray:
    """Unnormalized reweighting weights for a set of configurations."""
    delta_e = e_dft - e_model
    delta_e = delta_e - delta_e.min()  # numerical stability, does not change N_eff because it is a constant offset 
    return np.exp(-beta * delta_e)


def effective_sample_size(weights: np.ndarray) -> float:
    """N_eff = (sum w)^2 / sum(w^2). Ranges from 1 (single dominant sample) to N."""
    return float(weights.sum() ** 2 / np.sum(weights ** 2))


def reweighted_average(values: np.ndarray, weights: np.ndarray) -> float:
    """Weighted average sum(a*w)/sum(w) -- the reweighted estimate of <A>_DFT."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    return float((values * weights).sum() / weights.sum())

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


def running_moments(x) -> dict:
    """Laufende Momente: nach jedem neuen x_k Mittel, Std, Schiefe, Kurtosis.

    Vektorisiert ueber kumulative Potenzsummen, O(n). Fuer dE = E_DFT - E_MACE
    gedacht: liefert die beiden Eingangsgroessen des Gueltigkeitskriteriums des
    Gauss-Praediktors (c = beta*std und gamma_1), als Funktion des
    Stichprobenumfangs.

    Zentrale Momente aus Potenzsummen:
        m2 = <x^2> - mu^2
        m3 = <x^3> - 3 mu <x^2> + 2 mu^3
        m4 = <x^4> - 4 mu <x^3> + 6 mu^2 <x^2> - 3 mu^4
        Schiefe   g1 = m3 / m2^(3/2)          (Populationsversion, wie scipy.stats.skew)
        Exz.-Kurt g2 = m4 / m2^2 - 3          (wie scipy.stats.kurtosis)

    Numerik: Potenzsummen sind schlecht konditioniert, wenn |x| gross gegen die
    Streuung ist. Da alle Momente ab dem zweiten VERSCHIEBUNGSINVARIANT sind,
    wird intern um den Gesamtmittelwert zentriert - das aendert die Ergebnisse in
    exakter Arithmetik nicht, verbessert die Kondition aber drastisch. (Wichtig,
    falls jemand die Funktion versehentlich auf Gesamtenergien statt auf dE
    anwendet: -922 eV mit 8 meV Streuung waere sonst katastrophal.)

    x : 1D-Array in der Reihenfolge, in der die Werte anfallen.
    Rueckgabe: dict mit k, mean, std (ddof=1), skew, kurtosis - jeweils (n,).
    NaN, wo zu wenige Punkte fuer das jeweilige Moment vorliegen.
    """
    import numpy as np

    x = np.asarray(x, dtype=float)
    n = x.size
    shift = x.mean()                 # nur Konditionierung, verschiebungsinvariant
    y = x - shift

    k = np.arange(1, n + 1)
    s1 = np.cumsum(y)
    s2 = np.cumsum(y ** 2)
    s3 = np.cumsum(y ** 3)
    s4 = np.cumsum(y ** 4)

    mu = s1 / k
    m2 = np.maximum(s2 / k - mu ** 2, 0.0)
    m3 = s3 / k - 3.0 * mu * (s2 / k) + 2.0 * mu ** 3
    m4 = s4 / k - 4.0 * mu * (s3 / k) + 6.0 * mu ** 2 * (s2 / k) - 3.0 * mu ** 4

    with np.errstate(invalid="ignore", divide="ignore"):
        std = np.sqrt(np.where(k > 1, m2 * k / np.maximum(k - 1, 1), np.nan))
        skew = np.where(m2 > 0, m3 / np.power(m2, 1.5), np.nan)
        kurt = np.where(m2 > 0, m4 / (m2 ** 2) - 3.0, np.nan)

    # zu wenige Punkte -> undefiniert
    std[:1] = np.nan
    skew[:2] = np.nan
    kurt[:3] = np.nan

    return dict(k=k, mean=mu + shift, std=std, skew=skew, kurtosis=kurt)


def running_neff_cv(weights) -> dict:
    """Iterative N_eff-Schaetzung ueber den CV-Zusammenhang.

    Fuegt die Gewichte w_1, w_2, ... nacheinander hinzu und schaetzt nach jedem
    neuen w_k den effektiven Stichprobenumfang allein aus der bis dahin
    beobachteten Streuung:

        CV_k    = std_pop(w_1..w_k) / mean(w_1..w_k)
        N_eff_k = k / (1 + CV_k^2)

    Diese Schaetzung ist mit der direkten Kish-Formel auf denselben k Gewichten
    IDENTISCH (algebraisch: k/(1+CV^2) = (sum w)^2 / sum w^2), weil
    1 + CV^2 = <w^2>/<w>^2. Der Nutzen der iterativen Sicht ist die KONVERGENZ:
    man sieht, ab welchem Stichprobenumfang sich N_eff/k stabilisiert -> ob das
    Reweighting "klappen" wird (KI-Briefing / notebooks/Neff_first_look).

    Bei schweren Raendern springt die laufende Schaetzung noch spaet (ein einzelnes
    grosses w_k zieht CV hoch) -> genau das Signal fuer Unzuverlaessigkeit.

    weights : 1D-Array positiver Gewichte in der Reihenfolge, in der sie anfallen.
    Rueckgabe: dict mit
        k          : (n,) Stichprobenumfang 1..n
        cv         : (n,) laufender Variationskoeffizient (ddof=0)
        neff       : (n,) laufende N_eff-Schaetzung aus CV
        neff_ratio : (n,) neff / k
    """
    import numpy as np

    w = np.asarray(weights, dtype=float)
    n = w.size
    k = np.arange(1, n + 1)
    s1 = np.cumsum(w)               # sum_{i<=k} w_i
    s2 = np.cumsum(w * w)           # sum_{i<=k} w_i^2
    mean = s1 / k
    var = np.maximum(s2 / k - mean ** 2, 0.0)   # Populations-Varianz (ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        cv = np.where(mean > 0, np.sqrt(var) / mean, np.inf)
    neff = k / (1.0 + cv ** 2)
    return dict(k=k, cv=cv, neff=neff, neff_ratio=neff / k)


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


# ---------------------------------------------------------------------------
# Ensemble-Korrektur: c vom Testsatz auf den Produktionslauf uebertragen
# ---------------------------------------------------------------------------

def neff_ratio_cumulant(c: float, gamma1: float = 0.0, gamma2: float = 0.0) -> float:
    """N_eff/n aus der Kumulantenentwicklung.

        log(N_eff/n) = -c^2 + gamma_1 c^3 - (7/12) gamma_2 c^4 + O(c^5)

    gamma1 = gamma2 = 0 liefert den reinen Gauss-Praediktor exp(-c^2).
    Ab c ~ 1 unbrauchbar (Restterm (2c)^5/5! nicht mehr klein).
    """
    import numpy as np

    return float(np.exp(-c ** 2 + gamma1 * c ** 3 - (7.0 / 12.0) * gamma2 * c ** 4))


def ensemble_shift(delta_e, beta: float) -> dict:
    """c vom DFT-gesampelten Testsatz auf das MACE-Ensemble umrechnen.

    DAS PROBLEM. Der Testsatz wurde (Annahme!) aus p_DFT gezogen, der
    Produktionslauf laeuft dagegen als MD auf dem MACE-Potential, sampelt also
    p_MACE. Die Landschaft dE(R) ist dieselbe, aber ihre STREUUNG haengt davon
    ab, welche Konfigurationen man besucht. Das ist kein Stichprobenfehler - er
    verschwindet nicht mit n -> unendlich.

    DIE BEZIEHUNG. Beide Ensembles haengen ueber genau die Reweighting-Identitaet
    zusammen, die auch die Gewichte definiert:

        p_DFT(R) = w(R) p_MACE(R) / <w>,      w = exp(-beta dE)

    Rueckwaerts, also von den DFT-gesampelten Testframes ins MACE-Ensemble, ist
    demnach mit 1/w = exp(+beta dE) zu gewichten. Das ist hier EXAKT ausgefuehrt
    (Schluessel 'c_exact'), nicht genaehert.

    ERSTE ORDNUNG. Entwickelt man w ~ 1 - beta dE, folgt fuer eine beliebige
    Observable A der uebliche Stoerungsausdruck erster Ordnung

        <A>_DFT ~ <A>_MACE - beta Cov_MACE(A, dE)

    und mit A = (dE - mu)^2 wird Cov(A, dE) = <(dE-mu)^3> = gamma_1 sigma^3, also

        c_prod ~ c_test (1 + gamma_1 c / 2)                     ('c_first_order')

    Die Schiefe taucht auf, weil die Frage "wie verschiebt sich die BREITE" nach
    dem dritten Moment fragt. Bei symmetrischem dE passiert in erster Ordnung
    nichts. Vorzeichen: gamma_1 > 0 -> c_prod > c_test, die Testsatz-Schaetzung
    ist OPTIMISTISCH. Anschaulich: die MACE-MD uebersampelt Konfigurationen mit
    grossem positivem dE, also solche, die MACE fuer guenstiger haelt als sie
    sind - sie laeuft in ihre eigenen blinden Flecken.

    ZWEI VORBEHALTE.
      * Die Praemisse p_test = p_DFT ist UNGEPRUEFT. Wurde der Testsatz mit einem
        anderen Potential erzeugt statt per AIMD, ist die Korrektur nur eine
        Groessenordnung, keine Zahl.
      * Die Rueckwaerts-Umgewichtung hat ihr eigenes N_eff ('neff_backward').
        Ist das klein, ist 'c_exact' selbst verrauscht und die Entwicklung erster
        Ordnung womoeglich der stabilere Wert. Immer mit ausgeben.

    delta_e : (n,) dE = E_DFT - E_MACE je Frame [eV]
    beta    : 1/(k_B T) [1/eV]

    Rueckgabe: dict mit c_test, c_exact, c_first_order, gamma1, gamma2,
               neff_backward, n.
    """
    import numpy as np

    d = np.asarray(delta_e, dtype=float).ravel()
    n = d.size
    mu = d.mean()
    sigma = d.std(ddof=1)
    c_test = beta * sigma
    u = d - mu
    gamma1 = float((u ** 3).mean() / sigma ** 3)
    gamma2 = float((u ** 4).mean() / sigma ** 4 - 3.0)

    # exakte Rueckwaerts-Umgewichtung p_DFT -> p_MACE mit 1/w = exp(+beta dE);
    # Maximum abziehen, sonst ueberlaeuft exp bei grossem c
    log_r = beta * u
    r = np.exp(log_r - log_r.max())
    mean_m = (r * d).sum() / r.sum()
    var_m = (r * (d - mean_m) ** 2).sum() / r.sum()
    c_exact = float(beta * np.sqrt(var_m))
    neff_backward = float(r.sum() ** 2 / (r ** 2).sum())

    return dict(
        n=n, c_test=float(c_test), gamma1=gamma1, gamma2=gamma2,
        c_exact=c_exact,
        c_first_order=float(c_test * (1.0 + 0.5 * gamma1 * c_test)),
        neff_backward=neff_backward,
    )


def scale_to_system_size(c: float, gamma1: float, gamma2: float,
                         n_from: int, n_to: int) -> dict:
    """c, gamma_1, gamma_2 von n_from auf n_to Molekuele skalieren.

    Unter der Annahme, dass dE eine Summe aus N/n_xi UNABHAENGIGEN lokalen
    Beitraegen ist (zentraler Grenzwertsatz), gilt

        c ∝ sqrt(N),    gamma_1 ∝ N^(-1/2),    gamma_2 ∝ N^(-1)

    Die Schiefe faellt also mit, waehrend c steigt - die Skalierung ist damit
    selbstkonsistent und nicht bloss eine Verschiebung entlang c.

    ACHTUNG: die Unabhaengigkeitsannahme ist genau die, die 11_error_correlation
    prueft. Sie ist dort fuer den messbaren Bereich (1-6 A) nicht widerlegt
    worden, aber der langwellige Anteil (k -> 0) bleibt unzugaenglich, weil der
    Kraftfehler als Gradient ein Hochpassfilter ist. Ergebnisse dieser Funktion
    sind daher Prognosen unter einer plausiblen, nicht bewiesenen Annahme.
    """
    import numpy as np

    s = np.sqrt(n_to / n_from)
    return dict(n_molecules=n_to, c=float(c * s),
                gamma1=float(gamma1 / s), gamma2=float(gamma2 / s ** 2))
