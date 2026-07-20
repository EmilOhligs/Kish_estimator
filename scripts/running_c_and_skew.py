"""
Laufende Schaetzung von c = beta*std(dE) und der Schiefe gamma_1
============================================================================

Berechnet bei jedem hinzukommenden Frame die beiden Eingangsgroessen des
Gueltigkeitskriteriums fuer den Gauss-Praediktor (siehe
notebooks/gauss_naeherung_gueltigkeit.md):

    c(k)       = beta * std(dE_1..dE_k)
    gamma_1(k) = Schiefe(dE_1..dE_k)
    Fehler(k)  ~ |gamma_1(k)| * c(k)^3        (erwarteter relativer Fehler von
                                               predicted_neff_gauss)

Die Frage, die das beantwortet: ab wie vielen Frames sind c und gamma_1 stabil
genug, dass man dem Kriterium trauen kann? Erfahrungsgemaess konvergiert die
Schiefe deutlich langsamer als die Streuung - sie ist ein drittes Moment und
reagiert empfindlich auf einzelne Ausreisser.

Wie in den anderen Running-Skripten wird ueber mehrere Reihenfolgen gemittelt
(Median + 10/90-%-Band), damit die Streuung nicht von einer zufaelligen
Frame-Reihenfolge abhaengt. Bei k = n laufen alle Kurven zusammen.

Ausfuehren:
    python scripts/running_c_and_skew.py \
        --energies results/mace_energies_ensemble_L2c_testbig.npz
    python scripts/running_c_and_skew.py --shuffles 200 --temperature 300
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.predictions import load_energies
from uq_mace.reweighting import running_moments

K_B = 8.617333262e-5  # eV/K


def running_over_shuffles(dE: np.ndarray, beta: float, shuffles: int, rng):
    """Median + 10/90-%-Band von c(k), gamma_1(k) und |gamma_1|c^3 ueber Permutationen."""
    n = dE.size
    cs = np.empty((shuffles, n))
    gs = np.empty((shuffles, n))
    for s in range(shuffles):
        perm = rng.permutation(dE) if s > 0 else dE
        mom = running_moments(perm)
        cs[s] = beta * mom["std"]
        gs[s] = mom["skew"]
    err = np.abs(gs) * cs ** 3

    def summ(a):
        # die ersten Spalten sind per Definition NaN (zu wenige Punkte fuer das
        # jeweilige Moment) -> All-NaN-Warnung unterdruecken
        import warnings
        with np.errstate(invalid="ignore"), warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return (np.nanmedian(a, axis=0),
                    np.nanpercentile(a, 10, axis=0),
                    np.nanpercentile(a, 90, axis=0))

    return np.arange(1, n + 1), summ(cs), summ(gs), summ(err)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energies", default="results/mace_energies_ensemble_L2c_testbig.npz",
                    help="npz mit e_dft und e_mace/energies")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--shuffles", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    path = args.energies if Path(args.energies).is_absolute() else root / args.energies
    e_dft, e_mace = load_energies(path)
    dE = e_dft - e_mace
    n = dE.size
    beta = 1.0 / (K_B * args.temperature)
    rng = np.random.default_rng(args.seed)

    k, (c_med, c_lo, c_hi), (g_med, g_lo, g_hi), (e_med, e_lo, e_hi) = \
        running_over_shuffles(dE, beta, args.shuffles, rng)

    c_fin, g_fin = c_med[-1], g_med[-1]
    err_fin = abs(g_fin) * c_fin ** 3
    print(f"\nn = {n},  T = {args.temperature:.0f} K,  beta = {beta:.2f} eV^-1")
    print(f"  Endwerte:  c = {c_fin:.4f},  gamma_1 = {g_fin:+.4f}")
    print(f"  -> erwarteter Fehler des Gauss-Praediktors ~ {err_fin*100:.2f} %\n")

    # Konvergenz: ab wann liegt der Median innerhalb von 10 % des Endwerts?
    def first_stable(med, final, tol=0.10):
        ok = np.abs(med - final) <= tol * abs(final)
        # letzter Ausreisser bestimmt den Einschwingpunkt
        bad = np.where(~ok)[0]
        return int(bad[-1] + 2) if bad.size else 1

    print(f"  c stabil (+-10 %) ab k = {first_stable(c_med, c_fin)}")
    print(f"  gamma_1 stabil (+-10 %) ab k = {first_stable(g_med, g_fin)}")
    print(f"  Bandbreite bei k=100:  c = [{c_lo[99]:.3f}, {c_hi[99]:.3f}], "
          f"gamma_1 = [{g_lo[99]:+.3f}, {g_hi[99]:+.3f}]")

    # ---------------------------------------------------------------
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 11), sharex=True)

    ax1.fill_between(k, c_lo, c_hi, color="steelblue", alpha=0.18,
                     label="10-90 % ueber Reihenfolgen")
    ax1.plot(k, c_med, color="steelblue", lw=1.8, label="laufendes $c$")
    ax1.axhline(c_fin, color="firebrick", ls="--", lw=1.3, label=f"Endwert = {c_fin:.3f}")
    ax1.set_ylabel("$c = \\beta\\cdot$std$(\\Delta E)$")
    ax1.set_title("(a) Laufende Streuung, in thermischen Einheiten")
    ax1.legend(fontsize=8.5); ax1.grid(alpha=0.3)

    ax2.fill_between(k, g_lo, g_hi, color="darkorange", alpha=0.18,
                     label="10-90 % ueber Reihenfolgen")
    ax2.plot(k, g_med, color="darkorange", lw=1.8, label="laufende Schiefe $\\gamma_1$")
    ax2.axhline(g_fin, color="firebrick", ls="--", lw=1.3, label=f"Endwert = {g_fin:+.3f}")
    ax2.axhline(0.0, color="k", ls=":", lw=1.0)
    # Streuung der Schiefe-Schaetzung bei Normalitaet: sqrt(6/k)
    ax2.plot(k[4:], g_fin + 1.96 * np.sqrt(6.0 / k[4:]), color="gray", ls=":", lw=1.0,
             label="$\\pm1.96\\sqrt{6/k}$ (Normalfall)")
    ax2.plot(k[4:], g_fin - 1.96 * np.sqrt(6.0 / k[4:]), color="gray", ls=":", lw=1.0)
    ax2.set_ylabel("$\\gamma_1$")
    ax2.set_title("(b) Laufende Schiefe - konvergiert deutlich langsamer")
    ax2.legend(fontsize=8.5); ax2.grid(alpha=0.3)

    ax3.fill_between(k, e_lo * 100, e_hi * 100, color="seagreen", alpha=0.18)
    ax3.plot(k, e_med * 100, color="seagreen", lw=1.8, label="$|\\gamma_1|c^3$")
    for tol, col in ((1, "gray"), (5, "crimson")):
        ax3.axhline(tol, color=col, ls=":", lw=1.1, label=f"{tol} % Toleranz")
    ax3.set_ylabel("erwarteter Fehler [%]")
    ax3.set_xlabel("Stichprobenumfang  k  (Zahl der Frames)")
    ax3.set_title("(c) Daraus abgeleiteter Fehler des Gauss-Praediktors")
    ax3.legend(fontsize=8.5); ax3.grid(alpha=0.3)

    fig.suptitle(f"Laufende Eingangsgroessen des Gauss-Kriteriums   |   "
                 f"n={n}, T={args.temperature:.0f} K   |   "
                 f"final: c={c_fin:.3f}, $\\gamma_1$={g_fin:+.3f}, "
                 f"Fehler~{err_fin*100:.1f} %", fontsize=11)
    fig.subplots_adjust(top=0.93, hspace=0.25, left=0.09, right=0.97, bottom=0.06)

    out = (root / args.outdir / "running_c_and_skew.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] gespeichert -> {out}")


if __name__ == "__main__":
    main()
