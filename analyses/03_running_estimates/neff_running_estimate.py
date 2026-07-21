"""
Iterative N_eff-Schaetzung ueber den CV-Zusammenhang
============================================================================

Fuer eine Verteilung von Gewichten w_i wird bei JEDEM neu hinzukommenden w_k
der effektive Stichprobenumfang aus der bisher beobachteten Streuung geschaetzt:

        CV_k    = std(w_1..w_k) / mean(w_1..w_k)
        N_eff_k = k / (1 + CV_k^2)

(Kernfunktion: uq_mace.reweighting.running_neff_cv - vektorisiert, O(n).)

Die Schaetzung ist algebraisch identisch zur Kish-Formel auf den ersten k
Gewichten; interessant ist die KONVERGENZ: ab welchem k stabilisiert sich
N_eff/k? Bei schweren Raendern springt die Kurve noch spaet -> Warnsignal.

Damit die Aussage nicht von einer einzelnen (zufaelligen) Reihenfolge abhaengt,
wird die laufende Schaetzung ueber mehrere Permutationen wiederholt und als
Median + 10/90-%-Band gezeigt.

Quelle der Gewichte:
  --source real       echte w_i aus E_DFT - E_MACE (braucht MACE-Cache / GPU-Lauf)
  --source lognormal|gamma|pareto|exponential|halfnormal|constant   synthetisch

Beispiele:
    python analyses/neff_running_estimate.py --source real
    python analyses/neff_running_estimate.py --source lognormal --sigma 1.0 -n 2000
    python analyses/neff_running_estimate.py --source pareto --alpha 2.0 --shuffles 200
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.reweighting import effective_sample_size, running_neff_cv

HERE = Path(__file__).resolve().parent   # Ausgaben landen neben dem Skript
K_B = 8.617333262e-5  # eV/K


# ---------------------------------------------------------------------------
# Gewichte beschaffen
# ---------------------------------------------------------------------------
def real_weights(ensemble: str, testset: str, temperature: float) -> np.ndarray:
    """Echte Reweighting-Gewichte w_i = exp(-beta*(E_DFT - E_MACE))."""
    from uq_mace.predictions import get_predictions
    from uq_mace.reweighting import reweighting_weights

    pred = get_predictions(ensemble, testset)
    e_mace = pred["energies"].mean(axis=0)
    beta = 1.0 / (K_B * temperature)
    return reweighting_weights(pred["e_dft"], e_mace, beta)


from uq_mace.predictions import load_weights as load_weights_file  # noqa: E402


def synthetic_weights(source: str, n: int, rng, *, sigma: float, alpha: float,
                      shape: float) -> np.ndarray:
    """Zieht n positive Gewichte aus einer benannten Verteilung."""
    if source == "lognormal":
        return rng.lognormal(0.0, sigma, n)
    if source == "gamma":
        return rng.gamma(shape, 1.0, n)
    if source == "pareto":
        return rng.pareto(alpha, n) + 1.0
    if source == "exponential":
        return rng.exponential(1.0, n)
    if source == "halfnormal":
        return np.abs(rng.normal(0.0, 1.0, n))
    if source == "constant":
        return np.ones(n)
    raise ValueError(f"unbekannte Verteilung: {source}")


# ---------------------------------------------------------------------------
# Iterative Schaetzung ueber mehrere Reihenfolgen
# ---------------------------------------------------------------------------
def running_over_shuffles(w: np.ndarray, shuffles: int, rng):
    """Laufende N_eff/k und CV ueber mehrere Permutationen der Gewichte.

    Rueckgabe: k, ratio_med/lo/hi, cv_med  (jeweils (n,))."""
    n = w.size
    ratios = np.empty((shuffles, n))
    cvs = np.empty((shuffles, n))
    for s in range(shuffles):
        perm = rng.permutation(w) if shuffles > 1 else w
        res = running_neff_cv(perm)
        ratios[s] = res["neff_ratio"]
        cvs[s] = res["cv"]
    k = np.arange(1, n + 1)
    return (k,
            np.median(ratios, axis=0),
            np.percentile(ratios, 10, axis=0),
            np.percentile(ratios, 90, axis=0),
            np.median(cvs, axis=0))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(k, r_med, r_lo, r_hi, cv_med, w, label, out_png):
    n = w.size
    neff_final = effective_sample_size(w)
    ratio_final = neff_final / n

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # oben: N_eff/k (laufende Schaetzung) mit Band
    if not np.allclose(r_lo, r_hi):
        ax1.fill_between(k, r_lo, r_hi, color="steelblue", alpha=0.20,
                         label="10-90 % ueber Reihenfolgen")
    ax1.plot(k, r_med, color="steelblue", lw=1.8, label="laufende Schaetzung  $N_{eff}/k$")
    ax1.axhline(ratio_final, color="firebrick", ls="--", lw=1.4,
                label=f"Endwert (Kish) = {ratio_final:.3f}")
    ax1.set_ylabel("$N_{eff}/k$  (aus CV)")
    ax1.set_ylim(0, 1.02)
    ax1.set_title(f"Iterative N_eff-Schaetzung ueber CV   |   {label}   |   "
                  f"n={n},  N_eff={neff_final:.1f}")
    ax1.legend(fontsize=9, loc="best")
    ax1.grid(alpha=0.3)

    # zweite y-Achse: absolute N_eff-Schaetzung
    ax1b = ax1.twinx()
    ax1b.plot(k, r_med * k, color="seagreen", lw=1.0, alpha=0.6)
    ax1b.set_ylabel("$N_{eff}$ (absolut)", color="seagreen")
    ax1b.tick_params(axis="y", labelcolor="seagreen")

    # unten: laufender CV
    ax2.plot(k, cv_med, color="darkorange", lw=1.8)
    ax2.set_ylabel("CV$_k$ = std/mean")
    ax2.set_xlabel("Stichprobenumfang  k  (Zahl der beruecksichtigten $w_i$)")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"[plot ] gespeichert -> {out_png}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=None,
                    help="Pfad zu vorhandenen Daten (.npy/.npz/.txt/.csv) - kein MACE-Neulauf. "
                         "Akzeptiert fertige Gewichte ODER einen Energie-Cache mit e_dft/e_mace "
                         "(z.B. cache/mace_energies_*.npz) -> w_i wird bei --temperature berechnet.")
    ap.add_argument("--source", default="lognormal",
                    choices=["real", "lognormal", "gamma", "pareto",
                             "exponential", "halfnormal", "constant"])
    ap.add_argument("-n", "--n", type=int, default=2000, help="Anzahl Gewichte (synthetisch)")
    ap.add_argument("--shuffles", type=int, default=100,
                    help="Permutationen fuer das Konvergenz-Band (1 = eine feste Reihenfolge)")
    ap.add_argument("--seed", type=int, default=0)
    # Verteilungs-Parameter
    ap.add_argument("--sigma", type=float, default=1.0, help="lognormal sigma")
    ap.add_argument("--alpha", type=float, default=2.0, help="pareto alpha")
    ap.add_argument("--shape", type=float, default=2.0, help="gamma shape k")
    # nur fuer --source real
    ap.add_argument("--ensemble", default="ensemble_L2c",
                    choices=["ensemble_L2c", "ensemble_L0c", "ensemble_L0"])
    ap.add_argument("--testset", default="big", choices=["big", "small"])
    ap.add_argument("--temperature", type=float, default=300.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.weights:
        w = load_weights_file(args.weights, args.temperature)
        label = f"Datei: {Path(args.weights).name}"
        tag = Path(args.weights).stem
    elif args.source == "real":
        w = real_weights(args.ensemble, args.testset, args.temperature)
        label = f"real: {args.ensemble}, test{args.testset}, {args.temperature:.0f} K"
        tag = f"real_{args.ensemble}_test{args.testset}"
    else:
        w = synthetic_weights(args.source, args.n, rng,
                              sigma=args.sigma, alpha=args.alpha, shape=args.shape)
        label = f"{args.source} (n={args.n})"
        tag = f"{args.source}_n{args.n}"

    k, r_med, r_lo, r_hi, cv_med = running_over_shuffles(w, args.shuffles, rng)

    neff_final = effective_sample_size(w)
    print(f"\nQuelle: {label}")
    print(f"  n = {w.size}")
    print(f"  finales CV       = {w.std() / w.mean():.4f}")
    print(f"  finales N_eff    = {neff_final:.1f}  ({neff_final / w.size * 100:.1f}% von n)")
    print(f"  N_eff/k Median bei k=n/2: {r_med[w.size // 2 - 1]:.3f},  bei k=n: {r_med[-1]:.3f}")

    root = Path(__file__).resolve().parents[2]
    out_png = HERE / f"neff_running_{tag}.png"
    plot(k, r_med, r_lo, r_hi, cv_med, w, label, out_png)


if __name__ == "__main__":
    main()
