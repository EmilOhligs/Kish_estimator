"""
Gewichtsverteilungen und Kumulanten-Parameter im Vergleich
============================================================================

Stellt fuer alle vorhandenen Energie-Caches nebeneinander:

  (a) die Verteilung der Gewichte w_i / <w>            (log-x, weil die Spannweite
                                                        ueber Modellqualitaeten
                                                        Groessenordnungen umfasst)
  (b) die standardisierte dE-Verteilung z = (dE-mu)/sigma gegen N(0,1)
      -> zeigt die FORM, unabhaengig von der Skala
  (c) die Kumulanten-Parameter c, gamma_1, gamma_2
  (d) die einzelnen TERME der Entwicklung

        log(N_eff/n) = -c^2 + gamma_1 c^3 - (7/12) gamma_2 c^4 + O(c^5)

      als Betraege auf log-Skala, plus die Restabschaetzung (2c)^5/5!.
      Fallen die Terme schnell, ist die Reihe brauchbar; tun sie es nicht,
      ist der Gauss-Praediktor (und auch seine Korrektur) unbrauchbar.
      Der effektive Entwicklungsparameter ist 2c, nicht c - weil K auch bei
      t = -2c ausgewertet wird.

Ausfuehren:
    python analyses/01_weight_distribution/compare_weight_distributions.py
    python analyses/01_weight_distribution/compare_weight_distributions.py --temperature 298
"""
from __future__ import annotations

import argparse
from math import factorial
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, norm, skew

from uq_mace.predictions import load_energies
from uq_mace.reweighting import effective_sample_size, psis_khat, reweighting_weights

HERE = Path(__file__).resolve().parent
CACHE = Path(__file__).resolve().parents[2] / "cache"
K_B = 8.617333262e-5  # eV/K


def analyse(path: Path, beta: float) -> dict:
    e_dft, e_mace = load_energies(path)
    dE = e_dft - e_mace
    n = dE.size
    sigma = dE.std(ddof=1)
    c = beta * sigma
    g1, g2 = float(skew(dE)), float(kurtosis(dE))
    w = reweighting_weights(e_dft, e_mace, beta)
    neff = effective_sample_size(w)

    name = (path.stem.replace("single_", "").replace("mace_energies_", "")
                     .replace("_testbig", "").replace("_testsmall", ""))
    return dict(
        name=name, n=n, c=c, gamma1=g1, gamma2=g2,
        z=(dE - dE.mean()) / sigma, w_norm=w / w.mean(),
        ratio=neff / n, khat=psis_khat(w),
        t2=c ** 2,                                   # |Gauss-Term|
        t3=abs(g1) * c ** 3,                         # |Schiefe-Term|
        t4=abs(7.0 / 12.0 * g2 * c ** 4),            # |Kurtosis-Term|
        t5=(2 * c) ** 5 / factorial(5),              # Restabschaetzung, kappa_5 ~ 1
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", default=None)
    ap.add_argument("--temperature", type=float, default=300.0)
    args = ap.parse_args()

    if args.caches:
        paths = [Path(p) for p in args.caches]
    else:
        paths = [p for p in sorted(CACHE.glob("*.npz"))
                 if "e_dft" in np.load(p, allow_pickle=True).files]
    if not paths:
        raise SystemExit(f"Keine Caches in {CACHE}.")

    beta = 1.0 / (K_B * args.temperature)
    rows = sorted((analyse(p, beta) for p in paths), key=lambda r: r["c"])
    cols = plt.cm.viridis(np.linspace(0.1, 0.85, len(rows)))

    # ---- Tabelle ----
    print(f"\nT = {args.temperature:.0f} K   (nach c sortiert)")
    print("=" * 104)
    print(f"{'Datensatz':<24}{'c':>7}{'gamma1':>9}{'gamma2':>9}{'N_eff/n':>9}{'khat':>8}"
          f"{'|c²|':>9}{'|g1c³|':>9}{'|g2c⁴|':>9}{'Rest':>9}")
    print("-" * 104)
    for r in rows:
        print(f"{r['name']:<24}{r['c']:>7.3f}{r['gamma1']:>+9.3f}{r['gamma2']:>+9.3f}"
              f"{r['ratio']:>9.3f}{r['khat']:>8.3f}"
              f"{r['t2']:>9.4f}{r['t3']:>9.4f}{r['t4']:>9.4f}{r['t5']:>9.4f}")
    print("=" * 104)
    print("\nKonvergenz der Reihe (Verhaeltnis aufeinanderfolgender Terme):")
    for r in rows:
        q1 = r["t3"] / r["t2"] if r["t2"] else np.nan
        q2 = r["t4"] / r["t3"] if r["t3"] else np.nan
        flag = "ok" if max(q1, q2) < 0.3 and r["t5"] < 0.05 else "REIHE UNBRAUCHBAR"
        print(f"  {r['name']:<24} t3/t2 = {q1:5.2f}   t4/t3 = {q2:5.2f}   "
              f"Rest = {r['t5']:6.3f}   -> {flag}")

    # ---- Plot ----
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    (a1, a2), (a3, a4) = ax

    # (a) Gewichtsverteilungen, log-x
    for r, col in zip(rows, cols):
        wn = r["w_norm"][r["w_norm"] > 0]
        bins = np.logspace(np.log10(max(wn.min(), 1e-8)), np.log10(wn.max()), 45)
        a1.hist(wn, bins=bins, density=True, histtype="step", lw=2, color=col,
                label=f"{r['name']}  (c={r['c']:.2f})")
    a1.set_xscale("log")
    a1.set_xlabel("$w_i / \\langle w\\rangle$")
    a1.set_ylabel("Dichte")
    a1.set_title("(a) Gewichtsverteilungen")
    a1.legend(fontsize=8); a1.grid(alpha=0.3, which="both")

    # (b) Form von dE, standardisiert
    xs = np.linspace(-4, 4, 300)
    a2.plot(xs, norm.pdf(xs), "k--", lw=1.6, label="$\\mathcal{N}(0,1)$")
    for r, col in zip(rows, cols):
        a2.hist(r["z"], bins=40, density=True, histtype="step", lw=2, color=col,
                label=f"{r['name']}  ($\\gamma_1$={r['gamma1']:+.2f})")
    a2.set_xlabel("$z = (\\Delta E - \\mu)/\\sigma$")
    a2.set_ylabel("Dichte")
    a2.set_title("(b) Form von $\\Delta E$ — skalenfrei")
    a2.legend(fontsize=8); a2.grid(alpha=0.3)

    # (c) Kumulanten-Parameter
    x = np.arange(len(rows)); wdt = 0.27
    a3.bar(x - wdt, [r["c"] for r in rows], wdt, label="$c=\\beta\\sigma$", color="steelblue")
    a3.bar(x, [r["gamma1"] for r in rows], wdt, label="$\\gamma_1$", color="darkorange")
    a3.bar(x + wdt, [r["gamma2"] for r in rows], wdt, label="$\\gamma_2$", color="seagreen")
    a3.axhline(0, color="k", lw=0.8)
    a3.set_xticks(x); a3.set_xticklabels([r["name"] for r in rows], rotation=20, ha="right")
    a3.set_title("(c) Kumulanten-Parameter")
    a3.legend(fontsize=9); a3.grid(alpha=0.3, axis="y")

    # (d) Terme der Entwicklung, log-Skala
    labels = [("$c^2$", "t2", "steelblue"), ("$|\\gamma_1c^3|$", "t3", "darkorange"),
              ("$|\\frac{7}{12}\\gamma_2c^4|$", "t4", "seagreen"),
              ("Rest $(2c)^5/5!$", "t5", "firebrick")]
    wdt = 0.2
    for j, (lbl, key, col) in enumerate(labels):
        a4.bar(x + (j - 1.5) * wdt, [max(r[key], 1e-6) for r in rows], wdt,
               label=lbl, color=col)
    a4.axhline(0.05, color="k", ls=":", lw=1.2, label="5 %-Marke")
    a4.set_yscale("log")
    a4.set_xticks(x); a4.set_xticklabels([r["name"] for r in rows], rotation=20, ha="right")
    a4.set_ylabel("Betrag des Terms (im log-Verhältnis)")
    a4.set_title("(d) Terme der Entwicklung — fallen sie schnell genug?")
    a4.legend(fontsize=8, ncol=2); a4.grid(alpha=0.3, axis="y", which="both")

    fig.suptitle(f"Gewichtsverteilungen und Kumulanten-Parameter   |   "
                 f"T = {args.temperature:.0f} K,  n = {rows[0]['n']}", fontsize=13)
    fig.subplots_adjust(top=0.92, hspace=0.32, wspace=0.2,
                        left=0.07, right=0.97, bottom=0.09)
    out = HERE / "compare_weight_distributions.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")


if __name__ == "__main__":
    main()
