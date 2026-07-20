"""Parametrisierte Verteilungen fuer die Regime- und Konvergenzstudien.

Zentral: StdSkewNormal - eine auf Mittel 0 / Std 1 normierte Skew-Normal mit
einstellbarer Schiefe und geschlossener momenterzeugender Funktion. Damit lassen
sich die realen dE-Daten nachbilden UND exakte asymptotische Referenzwerte
berechnen (was mit einer endlichen Stichprobe allein unmoeglich waere).
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm, skewnorm

# Maximale |Schiefe| der Skew-Normal-Familie (Grenzwert a -> +-inf)
MAX_SKEW = 0.9952717


def shape_for_skew(target_skew: float) -> float:
    """Formparameter a der Skew-Normal fuer eine Zielschiefe (|skew| < 0.995)."""
    if abs(target_skew) < 1e-12:
        return 0.0
    if abs(target_skew) >= MAX_SKEW:
        raise ValueError(
            f"Skew-Normal erreicht maximal |Schiefe| = {MAX_SKEW:.4f}; "
            f"angefragt: {target_skew}. Fuer staerkere Schiefe eine andere "
            "Familie waehlen (z.B. Johnson-SU)."
        )

    def g(a: float) -> float:
        d = a / np.sqrt(1.0 + a * a)
        num = (4.0 - np.pi) / 2.0 * (d * np.sqrt(2.0 / np.pi)) ** 3
        den = (1.0 - 2.0 * d * d / np.pi) ** 1.5
        return num / den - target_skew

    lo, hi = (0.0, 200.0) if target_skew > 0 else (-200.0, 0.0)
    return float(brentq(g, lo, hi))


class StdSkewNormal:
    """Skew-Normal, normiert auf Mittel 0 und Std 1.

    Achtung: In dieser Familie ist die Kurtosis durch die Schiefe FESTGELEGT -
    beide lassen sich nicht unabhaengig einstellen. `excess_kurtosis` gibt den
    resultierenden Wert zurueck, damit man pruefen kann, wie gut reale Daten
    getroffen werden.
    """

    def __init__(self, skew: float):
        self.skew = float(skew)
        self.a = shape_for_skew(skew)
        d = self.a / np.sqrt(1.0 + self.a * self.a)
        self.delta = d
        self.mu = d * np.sqrt(2.0 / np.pi)                # Mittel der Rohverteilung
        self.sigma = np.sqrt(1.0 - 2.0 * d * d / np.pi)   # Std der Rohverteilung

    # ------------------------------------------------------------------
    def rvs(self, size, rng) -> np.ndarray:
        """Standardisierte Ziehungen; `size` darf ein Tupel sein."""
        x = skewnorm.rvs(self.a, size=size, random_state=rng)
        return (x - self.mu) / self.sigma

    def excess_kurtosis(self) -> float:
        """Exzess-Kurtosis, die sich aus der Schiefe zwangsweise ergibt."""
        d = self.delta
        t = d * np.sqrt(2.0 / np.pi)
        return float(2.0 * (np.pi - 3.0) * t ** 4 / (1.0 - t ** 2) ** 2)

    # ------------------------------------------------------------------
    def exp_moment(self, c: float) -> float:
        """E[exp(-c*Z)] fuer standardisiertes Z - geschlossene Form.

        MGF der SN(0,1,a): M(t) = 2 exp(t^2/2) Phi(delta*t);
        fuer Z = (X-mu)/sigma gilt E[e^{-cZ}] = e^{c*mu/sigma} * M(-c/sigma).
        Kontrolle a=0: 2*exp(t^2/2)*0.5 = exp(c^2/2) (lognormal).
        """
        t = -c / self.sigma
        return float(np.exp(c * self.mu / self.sigma) * 2.0
                     * np.exp(0.5 * t * t) * norm.cdf(self.delta * t))

    def asymptotic_ratio(self, c: float) -> float:
        """Exaktes N_eff/n im Grenzwert n->oo:  E[w]^2 / E[w^2]."""
        m1 = self.exp_moment(c)
        m2 = self.exp_moment(2.0 * c)
        return float(m1 * m1 / m2)

    def gauss_predictor_error(self, c: float) -> float:
        """Relativer Fehler von exp(-c^2) gegenueber der exakten Wahrheit.

        Referenzwert fuer das Naeherungskriterium |gamma_1|*c^3.
        """
        exact = self.asymptotic_ratio(c)
        return float(np.exp(-c * c) / exact - 1.0)
