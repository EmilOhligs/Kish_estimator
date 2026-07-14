"""
Visualisierung: CV(w) vs. N_eff  und  Var(w) vs. N_eff
============================================================================

"Verteilung" meint immer die Verteilung der Gewichte w_i.

Fuer jede Verteilung/Parameterwahl werden N = 5000 Gewichte gezogen und
    - CV(w)  = std(w)/mean(w)
    - Var(w) = Varianz von w
    - N_eff  = (sum w)^2 / sum w^2   (Kish)
berechnet (ueber mehrere Wiederholungen gemittelt).

Zwei Plots:
  1) CV vs. N_eff  -> ALLE Verteilungsfamilien fallen auf EINE Kurve:
                      N_eff = n / (1 + CV^2)   (exakt, verteilungsfrei)
  2) Var vs. N_eff -> KEIN universeller Zusammenhang: Var haengt zusaetzlich
                      vom Mittelwert ab (CV^2 = Var/mean^2), deshalb streuen
                      Familien mit verschiedenen Mittelwerten auseinander.
                      Die Referenzkurve N_eff = n/(1+Var) gilt nur fuer mean=1.

Das ist die eigentliche Botschaft: CV ist die richtige (skaleninvariante)
Kenngroesse fuer N_eff, Var allein nicht.

Ausfuehren:   python cv_var_vs_neff.py
Abhaengigkeiten: numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Konfiguration
# ---------------------------------------------------------------------------
N      = 5000     # Anzahl Gewichte pro Durchlauf
REPEAT = 40       # Wiederholungen (Mittelung, glaettet vor allem schwere Raender)
SEED   = 0
rng = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Kennzahlen
# ---------------------------------------------------------------------------
def n_eff(w):
    s1 = w.sum()
    return s1 * s1 / (w * w).sum()


def measure(sampler):
    """Mittlere (CV, Var, N_eff) ueber REPEAT Ziehungen."""
    cvs, vars, neffs = [], [], []
    for _ in range(REPEAT):
        w = np.asarray(sampler(N), dtype=float)
        w = w[np.isfinite(w) & (w > 0)]
        if w.size < 2 or w.mean() <= 0:
            continue
        cvs.append(w.std() / w.mean())
        vars.append(w.var())
        neffs.append(n_eff(w))
    return np.mean(cvs), np.mean(vars), np.mean(neffs)


# ---------------------------------------------------------------------------
# Familien mit durchstimmbarem Spreizungs-Parameter
# (jede: Liste von Parameterwerten + Sampler(param, size))
# ---------------------------------------------------------------------------
FAMILIES = {
    "Lognormal(sigma)": (np.linspace(0.1, 2.0, 14),
                         lambda p, size: rng.lognormal(0.0, p, size)),
    "Gamma(k)":         (np.geomspace(0.2, 30.0, 14),
                         lambda p, size: rng.gamma(p, 1.0, size)),
    "Weibull(k)":       (np.geomspace(0.3, 5.0, 14),
                         lambda p, size: rng.weibull(p, size)),
    "Pareto(alpha)":    (np.linspace(2.2, 12.0, 14),
                         lambda p, size: rng.pareto(p, size) + 1.0),
}

# Verteilungen mit FESTEM CV (ein einzelner Punkt) -- zeigen die Universalitaet
SINGLES = {
    "Konstant":     lambda size: np.ones(size),
    "Uniform(0,1)": lambda size: rng.uniform(0.0, 1.0, size),
    "Half-Normal":  lambda size: np.abs(rng.normal(0.0, 1.0, size)),
    "Exponential":  lambda size: rng.exponential(1.0, size),
    "Chi^2(3)":     lambda size: rng.chisquare(3.0, size),
}


# ---------------------------------------------------------------------------
# Datenpunkte sammeln
# ---------------------------------------------------------------------------
def collect():
    fam_data = {}
    for name, (params, sampler) in FAMILIES.items():
        pts = [measure(lambda size, p=p: sampler(p, size)) for p in params]
        fam_data[name] = np.array(pts)          # Spalten: CV, Var, N_eff
    single_data = {name: np.array(measure(s)) for name, s in SINGLES.items()}
    return fam_data, single_data


# ---------------------------------------------------------------------------
# Plot 1: CV vs. N_eff
# ---------------------------------------------------------------------------
def plot_cv(fam_data, single_data, fname="cv_vs_neff.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # exakte theoretische Kurve
    cv_grid = np.linspace(0, max(6, np.sqrt(N)), 400)
    ax.plot(cv_grid, N / (1 + cv_grid**2), "k--", lw=2,
            label=r"$N_\mathrm{eff}=n/(1+CV^2)$ (exakt)")

    for name, d in fam_data.items():
        ax.plot(d[:, 0], d[:, 2], "o-", ms=4, alpha=0.85, label=name)
    for name, p in single_data.items():
        ax.plot(p[0], p[2], "s", ms=9, alpha=0.9, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("CV(w)  (= std/mean = tan θ)")
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_title(f"CV vs. N_eff  (n={N}) — alle Familien fallen auf eine Kurve")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"gespeichert: {fname}")


# ---------------------------------------------------------------------------
# Plot 2: Var vs. N_eff
# ---------------------------------------------------------------------------
def plot_var(fam_data, single_data, fname="var_vs_neff.png"):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    # Referenzkurve, die NUR fuer Mittelwert = 1 gilt (dann Var = CV^2)
    var_grid = np.geomspace(1e-3, 1e3, 400)
    ax.plot(var_grid, N / (1 + var_grid), color="gray", ls="--", lw=1.5,
            label=r"$N_\mathrm{eff}=n/(1+\mathrm{Var})$  (nur fuer mean=1)")

    for name, d in fam_data.items():
        ax.plot(d[:, 1], d[:, 2], "o-", ms=4, alpha=0.85, label=name)
    for name, p in single_data.items():
        ax.plot(p[1], p[2], "s", ms=9, alpha=0.9, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Var(w)")
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_title(f"Var vs. N_eff  (n={N}) — KEIN universeller Zusammenhang")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(fname, dpi=130)
    print(f"gespeichert: {fname}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fam_data, single_data = collect()
    plot_cv(fam_data, single_data)
    plot_var(fam_data, single_data)
    print("Fertig. Zwei PNGs im aktuellen Arbeitsverzeichnis.")