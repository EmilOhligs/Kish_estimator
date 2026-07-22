"""
Konvergenz von Var(ΔE) und Schiefe — und warum die Prognose γ₁ braucht
============================================================================

MOTIVATION. Die Entscheidungsschranke unter reiner Gauß-Annahme ist
c_max = sqrt(-ln R). Bei RECHTSSCHIEFE (γ₁ > 0, hier gemessen +0.50) unterschätzt
der Gauß-Prädiktor aber N_eff — der wahre Wert liegt HÖHER:

    N_eff/n = exp(-c² + γ₁c³ - 7/12 γ₂c⁴)  >  exp(-c²)   für γ₁ > 0.

Folge: nahe der Schwelle kann das reine Gauß-Kriterium ein Modell zu FALSCH-FAIL
verurteilen, obwohl sein wahres N_eff/n ≥ R ist. Um das auszuschließen, muss die
Prognose γ₁ (und γ₂) mitrechnen und die schiefe-korrigierte Schranke verwenden:

    c_max^korr : löse  c² - γ₁c³ + 7/12 γ₂c⁴ = -ln R.

DIESES SKRIPT belegt zweierlei:
  (1) wie schnell c = β·std(ΔE) und γ₁ konvergieren (Bootstrap-Bänder, gegen die
      analytische SE), und dass γ₁ bei kleinem k nach unten VERZERRT ist —
      also die Korrektur dort zu schwach greift (konservativ, sichere Richtung);
  (2) die Entscheidungskonsequenz: die FALSCH-FAIL-Zone zwischen der Gauß- und
      der schiefe-korrigierten Schranke.

SE-Theorie (verwendet als Referenzkurven):
    SE(c)/c  = sqrt((γ₂+2)/4k)
    SE(γ₁)  ≈ sqrt(6/k)          (exakt für Normalverteilung)
    SE(γ₂)  ≈ sqrt(24/k)

Ausführen:
    python analyses/13_sequential_screening/moment_convergence.py
    python analyses/13_sequential_screening/moment_convergence.py --model mace-L0-01
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import brentq

from uq_mace.predictions import load_energies

HERE = Path(__file__).resolve().parent
CACHE = Path(__file__).resolve().parents[2] / "cache"
K_B = 8.617333262e-5

MODELS = {
    "ensemble_L2c": "mace_energies_ensemble_L2c_testbig.npz",
    "mace-L2-c-01": "single_mace-L2-c-01_testbig.npz",
    "mace-L0-c-01": "single_mace-L0-c-01_testbig.npz",
    "mace-L0-01": "single_mace-L0-01_testbig.npz",
}


def moments(d, beta):
    s = d.std(ddof=1)
    u = d - d.mean()
    g1 = float((u ** 3).mean() / s ** 3)
    g2 = float((u ** 4).mean() / s ** 4 - 3.0)
    return beta * s, g1, g2


def cmax_gauss(R):
    return float(np.sqrt(-np.log(R)))


def cmax_skew(R, g1, g2):
    """Loest c² - γ₁c³ + 7/12 γ₂c⁴ = -ln R nach c (kleinste positive Wurzel)."""
    target = -np.log(R)
    f = lambda c: c ** 2 - g1 * c ** 3 + (7.0 / 12.0) * g2 * c ** 4 - target
    try:
        return float(brentq(f, 1e-4, 3.0))
    except ValueError:
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default="ensemble_L2c")
    ap.add_argument("--R", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--boot", type=int, default=800)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    beta = 1.0 / (K_B * args.temperature)
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    e_dft, e_mace = load_energies(CACHE / MODELS[args.model])
    dE = e_dft - e_mace
    n = dE.size
    c_true, g1_true, g2_true = moments(dE, beta)

    ks = np.unique(np.r_[np.arange(15, 100, 5), np.arange(100, n + 1, 25)])
    stat = {q: {m: np.empty(ks.size) for m in ("c", "g1", "g2")}
            for q in ("med", "lo", "hi")}
    for i, k in enumerate(ks):
        C = np.empty(args.boot); G1 = np.empty(args.boot); G2 = np.empty(args.boot)
        for b in range(args.boot):
            d = dE[rng.integers(0, n, k)]
            C[b], G1[b], G2[b] = moments(d, beta)
        for m, arr in (("c", C), ("g1", G1), ("g2", G2)):
            stat["med"][m][i] = np.median(arr)
            stat["lo"][m][i], stat["hi"][m][i] = np.percentile(arr, [5, 95])

    cg = cmax_gauss(args.R)
    cs = cmax_skew(args.R, g1_true, g2_true)

    # ---- Konsole ----
    print(f"\n{args.model}, T = {args.temperature:.0f} K, R = {args.R}")
    print("=" * 74)
    print(f"volle Daten (n={n}):  c = {c_true:.3f},  γ₁ = {g1_true:+.3f},  γ₂ = {g2_true:+.3f}")
    print(f"Schranke  Gauß:            c_max = {cg:.3f}")
    print(f"Schranke  schiefe-korr.:   c_max = {cs:.3f}   "
          f"(+{100*(cs-cg)/cg:.0f} % mehr c erlaubt)")
    print("-" * 74)
    print(f"{'k':>5}{'c':>9}{'rel.SE':>9}{'γ₁':>9}{'SE(γ₁)':>9}{'γ₁-Bias':>10}{'γ₂':>9}")
    for i, k in enumerate(ks):
        if k not in (15, 25, 50, 100, 200, n) and k != ks[-1]:
            continue
        rse = (stat["hi"]["c"][i] - stat["lo"]["c"][i]) / (2 * 1.645 * stat["med"]["c"][i])
        seg1 = (stat["hi"]["g1"][i] - stat["lo"]["g1"][i]) / (2 * 1.645)
        bias = stat["med"]["g1"][i] - g1_true
        print(f"{k:>5}{stat['med']['c'][i]:>9.3f}{100*rse:>8.1f}%"
              f"{stat['med']['g1'][i]:>9.3f}{seg1:>9.3f}{bias:>+10.3f}"
              f"{stat['med']['g2'][i]:>9.3f}")
    print("=" * 74)
    print("  γ₁-Bias < 0 bei kleinem k: Stichproben-Schiefe unterschätzt → Korrektur")
    print("  zu schwach → Schranke näher an Gauß → sichere (konservative) Richtung.")

    # ---- CSV ----
    f1 = args.outdir / f"moment_convergence_{args.model}.csv"
    with open(f1, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["k", "c_med", "c_lo", "c_hi", "g1_med", "g1_lo", "g1_hi",
                     "g2_med", "g2_lo", "g2_hi"])
        for i, k in enumerate(ks):
            wr.writerow([k] + [f"{stat[q][m][i]:.4f}"
                               for m in ("c", "g1", "g2") for q in ("med", "lo", "hi")])

    # ---- Plot ----
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10))

    # (a) c-Konvergenz
    a1.plot(ks, stat["med"]["c"], "-", color="steelblue", lw=2, label="$c$ (Bootstrap-Median)")
    a1.fill_between(ks, stat["lo"]["c"], stat["hi"]["c"], color="steelblue", alpha=0.2,
                    label="5–95 %-Band")
    a1.axhline(c_true, color="k", ls=":", lw=1.2, label=f"$c$(400) = {c_true:.3f}")
    a1.set_xlabel("Stichprobengröße $k$"); a1.set_ylabel("$c = \\beta\\,\\mathrm{std}(\\Delta E)$")
    a1.set_title("(a) $c$ konvergiert schnell — stabil ab $k\\approx15$")
    a1.legend(fontsize=9); a1.grid(alpha=0.3)

    # (b) relative SE gegen 1/sqrt(k)-Theorie, log-log
    rse = (stat["hi"]["c"] - stat["lo"]["c"]) / (2 * 1.645 * stat["med"]["c"])
    a2.loglog(ks, 100 * rse, "o", color="steelblue", ms=5, label="$c$: Bootstrap")
    a2.loglog(ks, 100 * np.sqrt((g2_true + 2) / (4 * ks)), "-", color="steelblue", lw=1.5,
              label="$c$: $\\sqrt{(\\gamma_2+2)/4k}$")
    seg1 = (stat["hi"]["g1"] - stat["lo"]["g1"]) / (2 * 1.645)
    a2.loglog(ks, 100 * seg1 / abs(g1_true), "s", color="darkorange", ms=5,
              label="$\\gamma_1$: Bootstrap (rel.)")
    a2.loglog(ks, 100 * np.sqrt(6 / ks) / abs(g1_true), "-", color="darkorange", lw=1.5,
              label="$\\gamma_1$: $\\sqrt{6/k}$")
    a2.set_xlabel("Stichprobengröße $k$"); a2.set_ylabel("relativer Fehler [%]")
    a2.set_title("(b) $c$ und $\\gamma_1$ fallen wie $1/\\sqrt{k}$ — aber $\\gamma_1$ höher")
    a2.legend(fontsize=8.5); a2.grid(alpha=0.3, which="both")

    # (c) gamma1-Konvergenz mit Bias
    a3.plot(ks, stat["med"]["g1"], "-", color="darkorange", lw=2, label="$\\gamma_1$ (Median)")
    a3.fill_between(ks, stat["lo"]["g1"], stat["hi"]["g1"], color="darkorange", alpha=0.2)
    a3.axhline(g1_true, color="k", ls=":", lw=1.2, label=f"$\\gamma_1$(400) = {g1_true:+.3f}")
    a3.set_xlabel("Stichprobengröße $k$"); a3.set_ylabel("$\\gamma_1$ (Schiefe)")
    a3.set_title("(c) $\\gamma_1$ langsamer & bei kleinem $k$ nach unten verzerrt")
    a3.legend(fontsize=9); a3.grid(alpha=0.3)

    # (d) Entscheidungskonsequenz: Falsch-FAIL-Zone
    cc = np.linspace(0.1, min(1.0, cs + 0.2), 300)
    ne_g = np.exp(-cc ** 2)
    ne_s = np.exp(-cc ** 2 + g1_true * cc ** 3 - (7.0 / 12.0) * g2_true * cc ** 4)
    a4.plot(cc, ne_g, color="steelblue", lw=2, label="Gauß  $e^{-c^2}$")
    a4.plot(cc, ne_s, color="darkorange", lw=2,
            label="mit $\\gamma_1,\\gamma_2$")
    a4.axhline(args.R, color="k", ls="--", lw=1.3, label=f"R = {args.R}")
    a4.axvspan(cg, cs, color="crimson", alpha=0.18, label="FALSCH-FAIL-Zone")
    a4.axvline(cg, color="steelblue", ls=":", lw=1.2)
    a4.axvline(cs, color="darkorange", ls=":", lw=1.2)
    a4.plot([c_true], [np.exp(-c_true**2 + g1_true*c_true**3 - 7/12*g2_true*c_true**4)],
            "k*", ms=15, label=f"dieses Modell (c={c_true:.2f})", zorder=5)
    a4.set_xlabel("$c$"); a4.set_ylabel("$N_\\mathrm{eff}/n$")
    a4.set_title("(d) zwischen den Schranken: Gauß sagt FAIL, wahr ist PASS")
    a4.legend(fontsize=8.5); a4.grid(alpha=0.3)

    fig.suptitle(
        f"Momenten-Konvergenz & warum die Prognose $\\gamma_1$ braucht — "
        f"{args.model}, R={args.R}, T={args.temperature:.0f} K", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = args.outdir / f"moment_convergence_{args.model}.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")
    print(f"[csv  ] {f1}")


if __name__ == "__main__":
    main()
