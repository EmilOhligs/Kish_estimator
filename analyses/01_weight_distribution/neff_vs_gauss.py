"""
N_eff exakt (Kish) vs. Gauß-Prädiktor — und woher die Differenz kommt
============================================================================

Fuer jeden vorhandenen Energie-Cache werden nebeneinander gestellt:

    N_eff (Kish)   = (sum w)^2 / sum w^2            exakt, keine Annahme
    N_eff (Gauss)  = n * exp(-c^2)                  setzt normalverteiltes dE voraus

Interessant ist die DIFFERENZ. Laut Kumulantenentwicklung

    log(N_eff/n) = -c^2 + gamma_1 c^3 - (7/12) gamma_2 c^4 + O(c^5)

ist der Gauss-Wert nur der fuehrende Term. Die relative Abweichung sollte also

    (Gauss - Kish)/Kish  =  exp(-[gamma_1 c^3 - (7/12) gamma_2 c^4]) - 1

betragen. Das Skript rechnet beides aus und vergleicht - stimmen gemessene und
vorhergesagte Abweichung ueberein, ist die Entwicklung unabhaengig bestaetigt.

Vorzeichen: gamma_1 > 0 (Rechtsschiefe) macht den Gauss-Praediktor KONSERVATIV,
er unterschaetzt N_eff dann.

Arbeitet auf allen cache/*.npz, die e_dft enthalten - also sowohl den
Einzelmember-Caches (single_*) als auch den Ensemble-Caches.

Ausfuehren:
    python analyses/01_weight_distribution/neff_vs_gauss.py
    python analyses/01_weight_distribution/neff_vs_gauss.py --temperature 298
    python analyses/01_weight_distribution/neff_vs_gauss.py --caches cache/single_mace-L0-01_testbig.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from uq_mace.predictions import load_energies
from uq_mace.reweighting import effective_sample_size, psis_khat, reweighting_weights

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
K_B = 8.617333262e-5  # eV/K


def analyse(path: Path, beta: float) -> dict:
    e_dft, e_mace = load_energies(path)
    dE = e_dft - e_mace
    n = dE.size
    sigma = dE.std(ddof=1)
    c = beta * sigma
    g1, g2 = float(skew(dE)), float(kurtosis(dE))

    neff_kish = effective_sample_size(reweighting_weights(e_dft, e_mace, beta))
    neff_gauss = n * np.exp(-c * c)

    # gemessene und vorhergesagte relative Abweichung des Gauss-Werts
    dev_meas = neff_gauss / neff_kish - 1.0
    corr = g1 * c ** 3 - (7.0 / 12.0) * g2 * c ** 4      # Zusatzterme der Entwicklung
    dev_pred = np.exp(-corr) - 1.0

    return dict(
        name=path.stem.replace("single_", "").replace("_testbig", "")
                      .replace("mace_energies_", ""),
        n=n, c=c, gamma1=g1, gamma2=g2,
        neff_kish=neff_kish, neff_gauss=neff_gauss,
        diff_abs=neff_gauss - neff_kish,
        dev_meas=dev_meas * 100, dev_pred=dev_pred * 100,
        term3=g1 * c ** 3 * 100, term4=-(7.0 / 12.0) * g2 * c ** 4 * 100,
        khat=psis_khat(reweighting_weights(e_dft, e_mace, beta)),
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", default=None,
                    help="npz-Dateien (Default: alle in cache/ mit e_dft)")
    ap.add_argument("--temperature", type=float, default=292.0)
    args = ap.parse_args()

    if args.caches:
        paths = [Path(p) for p in args.caches]
    else:
        paths = []
        for p in sorted(CACHE.glob("*.npz")):
            try:
                if "e_dft" in np.load(p, allow_pickle=True).files:
                    paths.append(p)
            except Exception:
                pass
    if not paths:
        raise SystemExit(f"Keine auswertbaren Caches in {CACHE}. "
                         "Erst single_member_weights.py oder plot_weight_distribution.py laufen lassen.")

    beta = 1.0 / (K_B * args.temperature)
    rows = [analyse(p, beta) for p in paths]

    # ---- Tabelle ----
    print(f"\nT = {args.temperature:.0f} K,  beta = {beta:.2f} eV^-1")
    print("=" * 104)
    print(f"{'Datensatz':<26}{'n':>5}{'c':>7}{'N_eff Kish':>12}{'N_eff Gauß':>12}"
          f"{'Differenz':>11}{'gemessen':>10}{'erwartet':>10}")
    print(f"{'':26}{'':>5}{'':>7}{'exakt':>12}{'e^-c²':>12}{'absolut':>11}"
          f"{'[%]':>10}{'[%]':>10}")
    print("-" * 104)
    for r in rows:
        print(f"{r['name']:<26}{r['n']:>5}{r['c']:>7.3f}{r['neff_kish']:>12.1f}"
              f"{r['neff_gauss']:>12.1f}{r['diff_abs']:>+11.1f}"
              f"{r['dev_meas']:>+10.2f}{r['dev_pred']:>+10.2f}")
    print("=" * 104)

    print("\nZerlegung der erwarteten Abweichung (Beitraege zum log-Verhaeltnis, in %):")
    print(f"  {'Datensatz':<26}{'gamma1*c^3':>12}{'-(7/12)g2*c^4':>16}{'Summe':>9}"
          f"{'Rest (gem.-erw.)':>18}")
    for r in rows:
        rest = r["dev_meas"] - r["dev_pred"]
        print(f"  {r['name']:<26}{r['term3']:>+12.3f}{r['term4']:>+16.3f}"
              f"{r['term3']+r['term4']:>+9.3f}{rest:>+18.3f}")
    print("\n  Kleiner Rest = die Entwicklung erklaert die Abweichung vollstaendig.")

    # ---- Plot ----
    x = np.arange(len(rows))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    wdt = 0.36
    a1.bar(x - wdt / 2, [r["neff_kish"] for r in rows], wdt,
           label="Kish (exakt)", color="steelblue")
    a1.bar(x + wdt / 2, [r["neff_gauss"] for r in rows], wdt,
           label="Gauß  $n\\,e^{-c^2}$", color="darkorange")
    for i, r in enumerate(rows):
        a1.text(i, max(r["neff_kish"], r["neff_gauss"]) * 1.01,
                f"{r['dev_meas']:+.2f} %", ha="center", fontsize=9, fontweight="bold")
    a1.set_xticks(x); a1.set_xticklabels([r["name"] for r in rows], rotation=20, ha="right")
    a1.set_ylabel("$N_{eff}$")
    a1.set_title("(a) Exakt vs. Gauß-Prädiktor")
    a1.legend(fontsize=9); a1.grid(alpha=0.3, axis="y")

    a2.bar(x - wdt / 2, [r["dev_meas"] for r in rows], wdt,
           label="gemessen", color="steelblue")
    a2.bar(x + wdt / 2, [r["dev_pred"] for r in rows], wdt,
           label="aus $\\gamma_1c^3-\\frac{7}{12}\\gamma_2c^4$", color="seagreen")
    a2.axhline(0, color="k", lw=0.8)
    a2.set_xticks(x); a2.set_xticklabels([r["name"] for r in rows], rotation=20, ha="right")
    a2.set_ylabel("relative Abweichung des Gauß-Werts [%]")
    a2.set_title("(b) Erklärt die Kumulantenentwicklung die Differenz?")
    a2.legend(fontsize=9); a2.grid(alpha=0.3, axis="y")

    fig.suptitle(f"$N_{{eff}}$: Kish vs. Gauß   |   T = {args.temperature:.0f} K", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = HERE / "neff_vs_gauss.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")


if __name__ == "__main__":
    main()
