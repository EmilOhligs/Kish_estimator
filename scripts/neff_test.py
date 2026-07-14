"""
Simulation von N_eff fuer verschiedene Verteilungen der Gewichte w_i.
============================================================================

Ziehe je 5000 Gewichte w_i > 0 aus 10 Verteilungen (von leicht- bis
schwerschwaenzig) und berechne den effektiven Stichprobenumfang (Kish):

        N_eff = (sum_i w_i)^2 / sum_i w_i^2

Zusaetzlich werden die robusten Overlap-Diagnostiken mitberechnet:
    - N_eff / n                     : Effizienzverhaeltnis
    - CV(w) = std(w)/mean(w) = tan(theta)
    - alpha_max = max(w)/sum(w)     : Konzentration (harte Schranke N_eff <= 1/alpha_max^2)
    - k_hat                         : Pareto-Tail-Index (Zuverlaessigkeit)

Kernaussage, die die Simulation zeigt: unabhaengig von der Verteilungs-FAMILIE
gilt exakt
        N_eff / n = 1 / (1 + CV(w)^2)
d.h. nur die Spreizung der Gewichte zaehlt, nicht die Form. Schwere Raender
erzeugen grosse CV -> kleines N_eff -> und, sichtbar an der Streuung ueber
Wiederholungen, unzuverlaessige Schaetzung.

Ausfuehren:   python neff_distributions.py
Abhaengigkeiten: numpy (Pflicht), scipy (optional, nur fuer k_hat),
                 matplotlib (optional, nur fuer den Plot).
"""

import numpy as np

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
N      = 5000     # Anzahl gezogener Gewichte pro Durchlauf
REPEAT = 300      # Wiederholungen (um Mittelwert UND Streuung von N_eff zu sehen)
SEED   = 0

rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# 10 Verteilungen fuer die Gewichte w_i (alle mit positivem Traeger),
# grob geordnet von leicht- zu schwerschwaenzig.
# Jeder Eintrag ist eine Funktion size -> np.ndarray positiver Gewichte.
# ---------------------------------------------------------------------------
DISTRIBUTIONS = {
    "Konstant (Referenz)":  lambda size: np.ones(size),
    # Gauss: Gewichte muessen positiv sein -> Normal um 1 herum, bei 0 abgeschnitten.
    # Schmale Gauss-Glocke = fast gleiche Gewichte -> N_eff ~ n (guter Overlap).
    "Normal(1, 0.3)":       lambda size: np.clip(rng.normal(1.0, 0.3, size), 1e-9, None),
    "Uniform(0.5, 1.5)":    lambda size: rng.uniform(0.5, 1.5, size),
    "Beta(2, 5)":           lambda size: rng.beta(2.0, 5.0, size),
    "Half-Normal":          lambda size: np.abs(rng.normal(0.0, 1.0, size)),
    "Exponential(1)":       lambda size: rng.exponential(1.0, size),
    "Gamma(k=2)":           lambda size: rng.gamma(2.0, 1.0, size),
    "Chi^2(df=3)":          lambda size: rng.chisquare(3.0, size),
    "Weibull(k=1.5)":       lambda size: rng.weibull(1.5, size),
    "Lognormal(0, 1)":      lambda size: rng.lognormal(0.0, 1.0, size),
    "Pareto(alpha=2)":      lambda size: (rng.pareto(2.0, size) + 1.0),  # inf. Varianz
}


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------
def n_eff(w):
    """Kish effektiver Stichprobenumfang."""
    s1 = w.sum()
    s2 = (w * w).sum()
    return s1 * s1 / s2


def cv(w):
    """Variationskoeffizient std/mean (= tan(theta))."""
    m = w.mean()
    return w.std() / m if m > 0 else np.inf


def alpha_max(w):
    """Groesstes normiertes Gewicht = max(w)/sum(w)."""
    return w.max() / w.sum()


def pareto_khat(w):
    """
    Pareto-Tail-Index k_hat via GPD-Fit an den oberen Tail.
    Nutzt scipy, falls vorhanden; sonst NaN.
    Schwellen: k_hat < 0.5 gut, < 0.7 verlaesslich, > 0.7 unzuverlaessig.
    """
    try:
        from scipy.stats import genpareto
    except ImportError:
        return np.nan
    w = np.sort(w[np.isfinite(w) & (w > 0)])
    S = w.size
    if S < 100:
        return np.nan
    M = int(np.ceil(min(0.2 * S, 3.0 * np.sqrt(S))))  # Groesse des Tails
    u = w[S - M - 1]                                   # Schwelle
    exceed = w[S - M:] - u
    exceed = exceed[exceed > 0]
    if exceed.size < 10:
        return np.nan
    try:
        c, _, _ = genpareto.fit(exceed, floc=0.0)     # c = Formparameter = k_hat
        return c
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------
def simulate():
    header = (f"{'Verteilung':<22} {'N_eff (Mittel)':>16} {'Std(N_eff)':>12} "
              f"{'N_eff/n':>9} {'1/(1+CV^2)':>11} {'CV':>8} "
              f"{'alpha_max':>10} {'k_hat':>7}")
    print(header)
    print("-" * len(header))

    results = {}
    for name, sampler in DISTRIBUTIONS.items():
        neffs, ratios, cvs, amaxs, khats = [], [], [], [], []
        for _ in range(REPEAT):
            w = sampler(N)
            ne = n_eff(w)
            c  = cv(w)
            neffs.append(ne)
            ratios.append(ne / N)
            cvs.append(c)
            amaxs.append(alpha_max(w))
            khats.append(pareto_khat(w))

        ne_mean = np.mean(neffs)
        ne_std  = np.std(neffs)
        r_mean  = np.mean(ratios)
        cv_mean = np.mean(cvs)
        pred    = 1.0 / (1.0 + cv_mean**2)   # theoretisches N_eff/n aus CV
        am_mean = np.mean(amaxs)
        kh_mean = np.nanmean(khats)

        results[name] = dict(neff=ne_mean, neff_std=ne_std, ratio=r_mean,
                             cv=cv_mean, pred=pred, alpha_max=am_mean, khat=kh_mean)

        print(f"{name:<22} {ne_mean:>16.1f} {ne_std:>12.1f} {r_mean:>9.3f} "
              f"{pred:>11.3f} {cv_mean:>8.2f} {am_mean:>10.4f} {kh_mean:>7.2f}")

    print("\nHinweise:")
    print(f"  n = {N} Gewichte pro Durchlauf, {REPEAT} Wiederholungen.")
    print("  N_eff/n stimmt in jeder Zeile mit 1/(1+CV^2) ueberein -> nur die")
    print("  Spreizung zaehlt, nicht die Verteilungsform.")
    print("  Std(N_eff) waechst dramatisch fuer schwere Raender (Lognormal, Pareto)")
    print("  -> genau dort ist die N_eff-Schaetzung selbst unzuverlaessig.")
    print("  k_hat > 0.7 markiert die Faelle mit instabiler Varianz.")
    return results


# ---------------------------------------------------------------------------
# Optionaler Plot
# ---------------------------------------------------------------------------
def make_plot(results, fname="neff_distributions.png"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib nicht verfuegbar - Plot uebersprungen.")
        return

    names  = list(results.keys())
    ratios = [results[n]["ratio"] for n in names]
    khats  = [results[n]["khat"] for n in names]

    fig, ax1 = plt.subplots(figsize=(11, 5))
    x = np.arange(len(names))
    ax1.bar(x, ratios, color="steelblue", alpha=0.8, label="N_eff / n")
    ax1.set_ylabel("N_eff / n", color="steelblue")
    ax1.set_ylim(0, 1.05)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right")

    ax2 = ax1.twinx()
    ax2.plot(x, khats, "o-", color="firebrick", label="k_hat")
    ax2.axhline(0.7, ls="--", color="firebrick", alpha=0.5)
    ax2.set_ylabel("k_hat (Pareto-Tail-Index)", color="firebrick")

    ax1.set_title(f"N_eff/n und k_hat je Verteilung (n={N}, {REPEAT} Wdh.)")
    fig.tight_layout()
    fig.savefig(fname, dpi=120)
    print(f"Plot gespeichert: {fname}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    res = simulate()
    make_plot(res)