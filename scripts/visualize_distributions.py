"""
Visualisierung der getesteten Gewichts-Verteilungen w_i
============================================================================

Zeigt die Form jeder Verteilung, die in cv_var_vs_neff.py verwendet wird:
je 5000 Ziehungen als Histogramm (Dichte).

  - Familien mit Spreizungs-Parameter: 3 repraesentative Werte ueberlagert,
    jeweils mit dem resultierenden CV in der Legende.
  - Verteilungen mit festem CV: ein Histogramm.

Schwere Raender (Lognormal, Pareto) werden zur besseren Lesbarkeit am
99%-Quantil abgeschnitten.

Ausfuehren:   python plot_verteilungen.py
Abhaengigkeiten: numpy, matplotlib
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

N    = 5000
SEED = 0
rng  = np.random.default_rng(SEED)


# ---------------------------------------------------------------------------
# Getestete Verteilungen (identisch zu cv_var_vs_neff.py)
# Familien: 3 repraesentative Parameterwerte (klein / mittel / gross)
# ---------------------------------------------------------------------------
FAMILIES = {
    "Lognormal(sigma)": ([0.3, 1.0, 2.0],  lambda p, size: rng.lognormal(0.0, p, size)),
    "Gamma(k)":         ([0.5, 2.0, 20.0], lambda p, size: rng.gamma(p, 1.0, size)),
    "Weibull(k)":       ([0.5, 1.5, 5.0],  lambda p, size: rng.weibull(p, size)),
    "Pareto(alpha)":    ([2.5, 4.0, 10.0], lambda p, size: rng.pareto(p, size) + 1.0),
}

SINGLES = {
    "Konstant":     lambda size: np.ones(size),
    "Uniform(0,1)": lambda size: rng.uniform(0.0, 1.0, size),
    "Half-Normal":  lambda size: np.abs(rng.normal(0.0, 1.0, size)),
    "Exponential":  lambda size: rng.exponential(1.0, size),
    "Chi^2(3)":     lambda size: rng.chisquare(3.0, size),
}


def cv(w):
    return w.std() / w.mean()


# ---------------------------------------------------------------------------
# Panels vorbereiten: (Titel, [(Label, samples), ...])
# ---------------------------------------------------------------------------
def build_panels():
    panels = []
    for name, (params, sampler) in FAMILIES.items():
        series = []
        for p in params:
            w = sampler(p, N)
            series.append((f"{name.split('(')[1][:-1]}={p:g}  (CV={cv(w):.2f})", w))
        panels.append((name, series))
    for name, sampler in SINGLES.items():
        w = sampler(N)
        panels.append((name, [(f"CV={cv(w):.2f}", w)]))
    return panels


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def main(fname="verteilungen.png"):
    panels = build_panels()
    ncol = 3
    nrow = int(np.ceil(len(panels) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 4.2 * nrow))
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, series) in zip(axes, panels):
        all_w = np.concatenate([w for _, w in series])
        hi = np.quantile(all_w, 0.99)   # Tail fuer Lesbarkeit abschneiden

        for label, w in series:
            if np.allclose(w, w.flat[0]):          # Konstante -> senkrechte Linie
                ax.axvline(w.flat[0], color="C2", lw=2, label=label)
                ax.set_xlim(w.flat[0] - 1, w.flat[0] + 1)
            else:
                wc = w[w <= hi] if hi > 0 else w
                ax.hist(wc, bins=60, density=True, alpha=0.55, label=label)

        ax.set_title(title)
        ax.set_xlabel("w")
        ax.set_ylabel("Dichte")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    for ax in axes[len(panels):]:            # ungenutzte Achsen ausblenden
        ax.axis("off")

    fig.suptitle(f"Getestete Verteilungen der Gewichte w  "
                 f"(je {N} Ziehungen; Tail am 99%-Quantil abgeschnitten)",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(fname, dpi=130)
    print(f"gespeichert: {fname}")


if __name__ == "__main__":
    main()