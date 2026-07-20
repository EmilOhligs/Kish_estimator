"""
Wann warnt k_hat, wo N_eff noch harmlos aussieht?  (Regime-Studie)
============================================================================

Modell der realen Situation: die Gewichte sind w = exp(-beta*dE) mit dE der
Energiedifferenz DFT-MACE. Schreibt man dE = std(dE) * z mit standardisiertem z,
haengt ALLES nur von einem Parameter ab:

        c = beta * std(dE)          (bei dir gemessen: c = 0.318)

denn  w = exp(-c*z).  Ein schlechteres Modell (groesseres std) oder eine tiefere
Temperatur (groesseres beta) schiebt c nach oben.

Untersucht werden drei Formen von z, um die ROLLE DER SCHIEFE sauber zu trennen:

    skew = 0     -> w ist exakt lognormal (Referenzfall)
    skew = +0.5  -> die bei dir GEMESSENE Rechtsschiefe von dE
    skew = -0.5  -> gespiegelt, zur Kontrolle

WICHTIG zum Vorzeichen: grosse w entstehen aus stark NEGATIVEN dE, der obere
w-Rand kommt also aus dem LINKEN Rand von dE. Rechtsschiefe (langer Rand nach
rechts, gestauchter nach links) macht den oberen w-Rand daher LEICHTER als
lognormal - sie wirkt schuetzend. Linksschiefe waere der gefaehrliche Fall.
Das Skript testet diese Behauptung, statt sie vorauszusetzen.

Der eigentliche Punkt (Panels c und d): bei wachsendem c wird das aus einer
ENDLICHEN Stichprobe berechnete N_eff systematisch zu OPTIMISTISCH - man hat den
Rand schlicht noch nicht gezogen - und gleichzeitig instabil. Verglichen wird
deshalb gegen den exakten asymptotischen Wert, der analytisch bekannt ist
(Momente der Skew-Normal, geschlossene Form; fuer skew=0 exakt exp(-c^2)).
k_hat markiert genau den Bereich, in dem diese Ueberschaetzung einsetzt.

Ausgabe:
    results/regime/khat_vs_neff_regime.png    Hauptabbildung (4 Panels)
    results/regime/finite_size_effect.png     k_hat und Bias als Funktion von n
    results/regime/regime_table.csv           alle Zahlen

Ausfuehren:
    python scripts/khat_vs_neff_regime.py
    python scripts/khat_vs_neff_regime.py --n 5000 --reps 40 --cmax 3.0
    python scripts/khat_vs_neff_regime.py --real results/mace_energies_ensemble_L2c_testbig.npz
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from uq_mace.distributions import StdSkewNormal, shape_for_skew  # noqa: F401
from uq_mace.reweighting import effective_sample_size, khat_threshold, psis_khat

K_B = 8.617333262e-5  # eV/K


# StdSkewNormal / shape_for_skew liegen jetzt in uq_mace.distributions
# (gemeinsam genutzt mit scripts/convergence_simulation.py).


# ---------------------------------------------------------------------------
# Endliche Stichproben
# ---------------------------------------------------------------------------
def finite_sample_stats(dist: StdSkewNormal, c: float, n: int, reps: int, rng):
    """N_eff/n und k_hat aus reps unabhaengigen Stichproben der Groesse n."""
    ratios = np.empty(reps)
    khats = np.empty(reps)
    for r in range(reps):
        z = dist.rvs(n, rng)
        w = np.exp(-c * z)
        w /= w.max()                     # nur numerische Stabilitaet, aendert nichts
        ratios[r] = effective_sample_size(w) / n
        khats[r] = psis_khat(w)
    return ratios, khats


def summarize(x: np.ndarray) -> tuple[float, float, float]:
    return (float(np.nanmedian(x)),
            float(np.nanpercentile(x, 10)),
            float(np.nanpercentile(x, 90)))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=5000, help="Stichprobengroesse je Ziehung")
    ap.add_argument("--reps", type=int, default=40, help="unabhaengige Ziehungen je c")
    ap.add_argument("--cmin", type=float, default=0.1)
    ap.add_argument("--cmax", type=float, default=3.0)
    ap.add_argument("--csteps", type=int, default=26)
    ap.add_argument("--skews", default="0.5,0.0,-0.5",
                    help="zu vergleichende Schiefen von dE (Komma-Liste)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--real", default=None,
                    help="npz mit e_dft/e_mace: misst c und Schiefe der echten Daten "
                         "und markiert den Punkt in den Plots")
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--outdir", default="results/regime")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    skews = [float(s) for s in args.skews.split(",")]
    cs = np.linspace(args.cmin, args.cmax, args.csteps)
    thr = khat_threshold(args.n)

    # ---- echte Daten (optional) ---------------------------------------
    c_real = skew_real = ratio_real = khat_real = None
    if args.real:
        from scipy.stats import skew as sp_skew

        from uq_mace.predictions import load_energies
        from uq_mace.reweighting import reweighting_weights

        e_dft, e_mace = load_energies(args.real)
        dE = e_dft - e_mace
        beta = 1.0 / (K_B * args.temperature)
        c_real = float(beta * dE.std(ddof=1))
        skew_real = float(sp_skew(dE))
        w_real = reweighting_weights(e_dft, e_mace, beta)
        ratio_real = float(effective_sample_size(w_real) / w_real.size)
        khat_real = float(psis_khat(w_real))
        print(f"[real ] n={dE.size}  c = beta*std = {c_real:.3f}  Schiefe(dE) = {skew_real:+.3f}")
        print(f"[real ] N_eff/n = {ratio_real:.3f}   k_hat = {khat_real:.3f}\n")

    # ---- Hauptsweep ----------------------------------------------------
    results: dict[float, dict] = {}
    for sk in skews:
        dist = StdSkewNormal(sk)
        asym = np.array([dist.asymptotic_ratio(c) for c in cs])
        r_med = np.empty_like(cs); r_lo = np.empty_like(cs); r_hi = np.empty_like(cs)
        k_med = np.empty_like(cs); k_lo = np.empty_like(cs); k_hi = np.empty_like(cs)
        print(f"[sweep] Schiefe {sk:+.2f} (Skew-Normal a={dist.a:.3f}) ...")
        for j, c in enumerate(cs):
            ratios, khats = finite_sample_stats(dist, c, args.n, args.reps, rng)
            r_med[j], r_lo[j], r_hi[j] = summarize(ratios)
            k_med[j], k_lo[j], k_hi[j] = summarize(khats)
        results[sk] = dict(asym=asym, r_med=r_med, r_lo=r_lo, r_hi=r_hi,
                           k_med=k_med, k_lo=k_lo, k_hi=k_hi)

    colors = {skews[i]: c for i, c in enumerate(["darkorange", "steelblue", "firebrick"][:len(skews)])}

    # ---- Hauptabbildung ------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    (axA, axB), (axC, axD) = axes

    for sk in skews:
        R = results[sk]
        col = colors[sk]
        lbl = f"Schiefe {sk:+.1f}" + (" (lognormal)" if sk == 0 else "")
        # (a) N_eff/n: endlich vs. exakt
        axA.plot(cs, R["asym"], color=col, lw=2.0, label=f"{lbl} - exakt (n$\\to\\infty$)")
        axA.plot(cs, R["r_med"], color=col, lw=1.2, ls="--",
                 label=f"{lbl} - gemessen (n={args.n})")
        axA.fill_between(cs, R["r_lo"], R["r_hi"], color=col, alpha=0.13)
        # (b) k_hat
        axB.plot(cs, R["k_med"], color=col, lw=1.8, label=lbl)
        axB.fill_between(cs, R["k_lo"], R["k_hi"], color=col, alpha=0.13)
        # (c) Ueberschaetzung
        axC.plot(cs, R["r_med"] / R["asym"], color=col, lw=1.8, label=lbl)
        # (d) relative Streuung
        axD.plot(cs, (R["r_hi"] - R["r_lo"]) / np.maximum(R["r_med"], 1e-12),
                 color=col, lw=1.8, label=lbl)

    axA.set_xlabel("$c = \\beta\\cdot\\mathrm{std}(\\Delta E)$")
    axA.set_ylabel("$N_{eff}/n$")
    axA.set_title("(a) $N_{eff}$: endliche Stichprobe vs. Wahrheit")
    axA.set_ylim(0, 1.02); axA.grid(alpha=0.3); axA.legend(fontsize=8)

    axB.axhline(0.7, color="k", ls="--", lw=1.3, label="$\\hat{k}=0.7$")
    axB.axhline(thr, color="gray", ls=":", lw=1.2, label=f"Schwelle (n={args.n}) = {thr:.2f}")
    axB.axhline(0.5, color="seagreen", ls=":", lw=1.0, label="$\\hat{k}=0.5$")
    axB.set_xlabel("$c = \\beta\\cdot\\mathrm{std}(\\Delta E)$")
    axB.set_ylabel("$\\hat{k}$")
    axB.set_title("(b) Pareto-Tail-Index")
    axB.grid(alpha=0.3); axB.legend(fontsize=8)

    axC.axhline(1.0, color="k", ls="--", lw=1.2, label="unverzerrt")
    axC.axhline(1.1, color="crimson", ls=":", lw=1.1, label="+10 % Ueberschaetzung")
    axC.set_xlabel("$c = \\beta\\cdot\\mathrm{std}(\\Delta E)$")
    axC.set_ylabel("gemessenes / wahres  $N_{eff}$")
    axC.set_title("(c) Wie optimistisch ist das gemessene $N_{eff}$?")
    axC.grid(alpha=0.3); axC.legend(fontsize=8)

    axD.set_xlabel("$c = \\beta\\cdot\\mathrm{std}(\\Delta E)$")
    axD.set_ylabel("(P90-P10) / Median")
    axD.set_title("(d) Instabilitaet von $N_{eff}$ ueber Wiederholungen")
    axD.grid(alpha=0.3); axD.legend(fontsize=8)

    for ax in (axA, axB, axC, axD):
        if c_real is not None:
            ax.axvline(c_real, color="purple", ls="-.", lw=1.5, alpha=0.8)
    if c_real is not None:
        axA.plot([c_real], [ratio_real], "*", ms=16, color="purple",
                 label=f"deine Daten (c={c_real:.2f})", zorder=5)
        axB.plot([c_real], [khat_real], "*", ms=16, color="purple", zorder=5)
        axA.legend(fontsize=8)

    fig.suptitle(
        f"Regime-Studie: $w=\\exp(-c\\,z)$, $c=\\beta\\cdot$std$(\\Delta E)$   |   "
        f"n = {args.n}, {args.reps} Wiederholungen je Punkt\n"
        "Kernfrage: ab welchem c ueberschaetzt das gemessene $N_{eff}$ die Wahrheit "
        "- und warnt $\\hat{k}$ dort bereits?",
        fontsize=12)
    fig.subplots_adjust(top=0.88, hspace=0.28, wspace=0.22,
                        left=0.07, right=0.97, bottom=0.07)
    outdir = Path(__file__).resolve().parents[1] / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "khat_vs_neff_regime.png", dpi=140)
    plt.close(fig)
    print(f"[plot ] {outdir/'khat_vs_neff_regime.png'}")

    # ---- Endlichkeits-Effekt: k_hat und Bias als Funktion von n --------
    ns = np.array([200, 500, 1000, 2000, 5000, 10000, 20000])
    c_fixed = [0.5, 1.0, 1.5]
    dist0 = StdSkewNormal(0.0)          # lognormal-Referenz
    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for c in c_fixed:
        km, bias = [], []
        asym = dist0.asymptotic_ratio(c)
        for n_ in ns:
            ratios, khats = finite_sample_stats(dist0, c, int(n_),
                                                max(10, args.reps // 2), rng)
            km.append(np.nanmedian(khats))
            bias.append(np.nanmedian(ratios) / asym)
        ax1.plot(ns, km, "o-", lw=1.6, label=f"c = {c}")
        ax2.plot(ns, bias, "o-", lw=1.6, label=f"c = {c}")
    ax1.axhline(0.7, color="k", ls="--", lw=1.2)
    ax1.plot(ns, [khat_threshold(int(x)) for x in ns], color="gray", ls=":", lw=1.2,
             label="Schwelle(n)")
    ax1.set_xscale("log"); ax1.set_xlabel("n"); ax1.set_ylabel("$\\hat{k}$")
    ax1.set_title("$\\hat{k}$ von lognormalen Gewichten vs. n\n"
                  "(asymptotisch 0 - aber extrem langsam)")
    ax1.grid(alpha=0.3, which="both"); ax1.legend(fontsize=9)
    ax2.axhline(1.0, color="k", ls="--", lw=1.2)
    ax2.set_xscale("log"); ax2.set_xlabel("n")
    ax2.set_ylabel("gemessenes / wahres $N_{eff}$")
    ax2.set_title("Optimismus-Bias von $N_{eff}$ vs. n")
    ax2.grid(alpha=0.3, which="both"); ax2.legend(fontsize=9)
    fig2.tight_layout()
    fig2.savefig(outdir / "finite_size_effect.png", dpi=140)
    plt.close(fig2)
    print(f"[plot ] {outdir/'finite_size_effect.png'}")

    # ---- Tabelle + Gefahrenzone ---------------------------------------
    csv_path = outdir / "regime_table.csv"
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.writer(fh)
        wtr.writerow(["skew", "c", "neff_ratio_exact", "neff_ratio_measured",
                      "overestimation", "khat", "khat_threshold"])
        for sk in skews:
            R = results[sk]
            for j, c in enumerate(cs):
                wtr.writerow([sk, round(float(c), 4), round(float(R["asym"][j]), 5),
                              round(float(R["r_med"][j]), 5),
                              round(float(R["r_med"][j] / R["asym"][j]), 4),
                              round(float(R["k_med"][j]), 4), round(thr, 4)])
    print(f"[csv  ] {csv_path}\n")

    print("=" * 96)
    print(f"{'skew':>6}{'c':>8}{'N_eff/n exakt':>15}{'gemessen':>11}"
          f"{'Ueberschaetzung':>17}{'k_hat':>9}{'Urteil':>16}")
    print("-" * 96)
    for sk in skews:
        R = results[sk]
        for j, c in enumerate(cs):
            if j % 4 and j != len(cs) - 1:
                continue
            over = R["r_med"][j] / R["asym"][j]
            verdict = "k_hat WARNT" if R["k_med"][j] > thr else "ok"
            print(f"{sk:>6.1f}{c:>8.2f}{R['asym'][j]:>15.3f}{R['r_med'][j]:>11.3f}"
                  f"{over:>16.2f}x{R['k_med'][j]:>9.3f}{verdict:>16}")
        print("-" * 96)

    print("\nGefahrenzone je Schiefe (k_hat ueber Schwelle UND N_eff um >10% zu optimistisch):")
    for sk in skews:
        R = results[sk]
        mask = (R["k_med"] > thr) & (R["r_med"] / R["asym"] > 1.10)
        if mask.any():
            print(f"  Schiefe {sk:+.1f}:  c von {cs[mask].min():.2f} bis {cs[mask].max():.2f}"
                  f"   (dort gemessenes N_eff/n bis {R['r_med'][mask].max():.2f} - sieht harmlos aus)")
        else:
            print(f"  Schiefe {sk:+.1f}:  keine - k_hat und N_eff verschlechtern sich hier gemeinsam")

    print("\nZur Schiefe: grosse w entstehen aus stark NEGATIVEN dE. Rechtsschiefe (+)")
    print("staucht den linken dE-Rand und macht den oberen w-Rand LEICHTER; Linksschiefe (-)")
    print("ist der gefaehrliche Fall. Vergleiche dazu die Kurven in Panel (b).")


if __name__ == "__main__":
    main()
