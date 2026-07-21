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
from pathlib import Path

HERE = Path(__file__).resolve().parent   # Ausgaben landen neben dem Skript
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
        if p[0] <= 0:      # CV=0 (Konstant) -> auf Log-Achse nicht darstellbar
            ax.axhline(p[2], color="green", ls=":", lw=1.5,
                       label=f"{name} (CV=0 → N_eff=n)")
            continue
        ax.plot(p[0], p[2], "s", ms=9, alpha=0.9, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("CV(w)  (= std/mean = tan θ)")
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_title(f"CV vs. N_eff  (n={N}) — alle Familien fallen auf eine Kurve")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=130)
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
        if p[1] <= 0:      # Var=0 (Konstant) -> auf Log-Achse nicht darstellbar
            ax.axhline(p[2], color="green", ls=":", lw=1.5,
                       label=f"{name} (Var=0 → N_eff=n)")
            continue
        ax.plot(p[1], p[2], "s", ms=9, alpha=0.9, label=name)

    ax.set_xscale("log")
    ax.set_xlabel("Var(w)")
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_title(f"Var vs. N_eff  (n={N}) — KEIN universeller Zusammenhang")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=130)
    print(f"gespeichert: {fname}")


# ---------------------------------------------------------------------------
# Plot 3: CV vs. N_eff — je Verteilung EINZELN (ein Subplot pro Familie),
#         alle Verteilungen mit festem CV gesammelt in EINEM Subplot.
# ---------------------------------------------------------------------------
def plot_cv_grid(fam_data, single_data, fname="cv_vs_neff_einzeln.png"):
    cv_grid = np.linspace(0, max(6, np.sqrt(N)), 400)
    theory  = N / (1 + cv_grid**2)

    panels   = list(fam_data.items())        # eine Familie = ein Subplot
    n_panels = len(panels) + 1               # + 1 Sammel-Subplot fuer feste CV
    ncol = 3
    nrow = int(np.ceil(n_panels / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(6 * ncol, 4.5 * nrow))
    axes = np.atleast_1d(axes).ravel()

    # je Familie ein Subplot (mit theoretischer Kurve als Referenz)
    for ax, (name, d) in zip(axes, panels):
        ax.plot(cv_grid, theory, "k--", lw=1.5, label=r"$n/(1+CV^2)$")
        ax.plot(d[:, 0], d[:, 2], "o-", ms=5, color="C0", label=name)
        ax.set_xscale("log")
        ax.set_xlabel("CV(w)")
        ax.set_ylabel(r"$N_\mathrm{eff}$")
        ax.set_title(name)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, which="both")

    # letzter genutzter Subplot: alle Verteilungen mit festem CV gemeinsam
    ax = axes[len(panels)]
    ax.plot(cv_grid, theory, "k--", lw=1.5, label=r"$n/(1+CV^2)$")
    for sname, p in single_data.items():
        if p[0] <= 0:      # Konstante (CV=0) -> Horizontale
            ax.axhline(p[2], color="green", ls=":", lw=1.5, label=f"{sname} (CV=0)")
            continue
        ax.plot(p[0], p[2], "s", ms=9, label=sname)
    ax.set_xscale("log")
    ax.set_xlabel("CV(w)")
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_title("Verteilungen mit festem CV")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    for ax in axes[n_panels:]:               # ungenutzte Achsen ausblenden
        ax.axis("off")

    fig.suptitle(f"CV vs. N_eff — je Verteilung einzeln (n={N})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(HERE / fname, dpi=130)
    print(f"gespeichert: {fname}")


# ---------------------------------------------------------------------------
# Plot 4: CV vs. N_eff mit LINEARER x-Achse
#         -> zeigt die eigentliche Funktionsform 1/(1+CV^2) (Hexe von Agnesi).
#         Ausschnitt bis xmax, weil grosse CV (Pareto/Lognormal) sonst alles
#         nach rechts ziehen. Konstante (CV=0) ist hier als Punkt darstellbar.
# ---------------------------------------------------------------------------
def plot_cv_linear(fam_data, single_data, fname="cv_vs_neff_linear.png", xmax=6.0):
    fig, ax = plt.subplots(figsize=(8, 5.5))

    cv_grid = np.linspace(0, xmax, 400)
    ax.plot(cv_grid, N / (1 + cv_grid**2), "k--", lw=2,
            label=r"$N_\mathrm{eff}=n/(1+CV^2)$ (exakt)")

    for name, d in fam_data.items():
        ax.plot(d[:, 0], d[:, 2], "o-", ms=4, alpha=0.85, label=name)
    for name, p in single_data.items():
        ax.plot(p[0], p[2], "s", ms=9, alpha=0.9, label=name)  # inkl. Konstante bei CV=0

    ax.set_xlim(0, xmax)          # LINEAR (kein set_xscale) + Ausschnitt
    ax.set_xlabel("CV(w)  (lineare Achse)")
    ax.set_ylabel(r"$N_\mathrm{eff}$")
    ax.set_title(f"CV vs. N_eff  (n={N}, linear) — Form $1/(1+CV^2)$ sichtbar")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / fname, dpi=130)
    print(f"gespeichert: {fname}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    fam_data, single_data = collect()
    plot_cv(fam_data, single_data)          # kombinierter CV-Plot (log)
    plot_cv_grid(fam_data, single_data)     # ein Subplot pro Verteilung (log)
    plot_cv_linear(fam_data, single_data)   # NEU: CV linear -> 1/(1+CV^2)-Form
    plot_var(fam_data, single_data)         # Var-Plot (log)
    print("Fertig. Vier PNGs im aktuellen Arbeitsverzeichnis.")