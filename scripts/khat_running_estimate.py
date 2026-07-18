"""
Iterativer Pareto-Tail-Index k_hat ueber die Gewichte w_i  (ECHTE Daten)
============================================================================

Analog zu neff_running_estimate.py, aber fuer die ZUVERLAESSIGKEITS-Diagnostik:
bei wachsendem Stichprobenumfang k wird der Pareto-Tail-Index

        k_hat(k) = GPD-Formparameter des oberen Tails von w_1..w_k

geschaetzt (Vehtari et al., JMLR 25, 2024; Implementierung: uq_mace.reweighting
.psis_khat). Faustregel: k_hat < min(1 - 1/log10(S), 0.7) -> gewichtsbasierte
Schaetzungen (inkl. Kish-N_eff) sind verlaesslich; k_hat > 0.7 -> UNZUVERLAESSIG,
egal wie gut N_eff aussieht.

Warum iterativ: N_eff kann bei kleinem k harmlos aussehen, waehrend der Tail noch
gar nicht "gesehen" wurde. Die k_hat-Kurve zeigt, ab wann die Tail-Schaetzung
ueberhaupt stabil ist - und ob sie mit mehr Daten nach oben wandert (spaet
auftauchende schwere Raender).

Technische Randbedingung: psis_khat braucht >= 25 Gewichte und >= 5 Tail-Punkte,
darunter gibt es NaN. Die Kurve startet daher bei k = 25.

Quelle der Gewichte (Default: die reellen w_i aus dem L2c-Reweighting):
  --weights PATH   vorhandene Datei (fertige w ODER Energie-Cache e_dft/e_mace)
  --source real    ueber den predictions-Cache (kein MACE-Neulauf, wenn gecacht)
  --source lognormal|pareto|...   synthetisch, zum Vergleich

Beispiele:
    python scripts/khat_running_estimate.py \
        --weights results/mace_energies_ensemble_L2c_testbig.npz
    python scripts/khat_running_estimate.py --source pareto --alpha 1.25 -n 3000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.predictions import load_weights
from uq_mace.reweighting import (
    effective_sample_size,
    khat_threshold,
    psis_khat,
    running_neff_cv,
)

K_B = 8.617333262e-5  # eV/K
K_MIN = 25            # psis_khat liefert darunter NaN


# ---------------------------------------------------------------------------
# Gewichte beschaffen
# ---------------------------------------------------------------------------
def real_weights(ensemble: str, testset: str, temperature: float) -> np.ndarray:
    """Echte Gewichte ueber den predictions-Cache (rechnet MACE nur, wenn kein Cache)."""
    from uq_mace.predictions import get_predictions
    from uq_mace.reweighting import reweighting_weights

    pred = get_predictions(ensemble, testset)
    beta = 1.0 / (K_B * temperature)
    return reweighting_weights(pred["e_dft"], pred["energies"].mean(axis=0), beta)


def synthetic_weights(source: str, n: int, rng, *, sigma: float, alpha: float,
                      shape: float) -> np.ndarray:
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
    raise ValueError(f"unbekannte Verteilung: {source}")


# ---------------------------------------------------------------------------
# Laufendes k_hat
# ---------------------------------------------------------------------------
def running_khat(w: np.ndarray, ks: np.ndarray) -> np.ndarray:
    """k_hat auf den ersten k Gewichten, fuer jedes k in ks."""
    return np.array([psis_khat(w[:k]) for k in ks])


def running_khat_over_shuffles(w: np.ndarray, ks: np.ndarray, shuffles: int, rng):
    """Median + 10/90-%-Band von k_hat(k) ueber mehrere Reihenfolgen.

    Wie beim N_eff-Skript: die Streuung ueber Permutationen zeigt, wie stark die
    Tail-Schaetzung davon abhaengt, welche Gewichte man zufaellig zuerst sieht.
    Bei k = n sind alle Reihenfolgen identisch -> Band kollabiert.
    """
    out = np.empty((shuffles, ks.size))
    for s in range(shuffles):
        perm = rng.permutation(w) if shuffles > 1 else w
        out[s] = running_khat(perm, ks)
    with np.errstate(invalid="ignore"):
        return (np.nanmedian(out, axis=0),
                np.nanpercentile(out, 10, axis=0),
                np.nanpercentile(out, 90, axis=0))


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------
def plot(ks, med, lo, hi, w, label, out_png):
    n = w.size
    khat_final = psis_khat(w)
    thr_final = khat_threshold(n)
    neff = effective_sample_size(w)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 2]})

    # --- oben: k_hat(k) ---
    if not np.allclose(lo, hi, equal_nan=True):
        ax1.fill_between(ks, lo, hi, color="firebrick", alpha=0.18,
                         label="10-90 % ueber Reihenfolgen")
    ax1.plot(ks, med, color="firebrick", lw=1.9, label="laufendes $\\hat{k}$")

    # Zuverlaessigkeits-Schwellen
    ax1.axhline(0.7, color="k", ls="--", lw=1.3, label="$\\hat{k}=0.7$ (unzuverlaessig darueber)")
    ax1.plot(ks, [khat_threshold(k) for k in ks], color="steelblue", ls=":", lw=1.4,
             label="Schwelle $\\min(1-1/\\log_{10}S,\\,0.7)$")
    ax1.axhline(0.5, color="seagreen", ls=":", lw=1.1, alpha=0.8, label="$\\hat{k}=0.5$ (gut)")
    ax1.axhline(khat_final, color="darkorange", ls="-.", lw=1.3,
                label=f"Endwert $\\hat{{k}}$ = {khat_final:.2f}")

    ax1.set_ylabel("$\\hat{k}$  (Pareto-Tail-Index)")
    ax1.set_title(f"Iterativer Pareto-Tail-Index   |   {label}   |   "
                  f"n={n},  $N_{{eff}}$={neff:.1f} ({neff/n*100:.1f}%),  "
                  f"$\\hat{{k}}_{{final}}$={khat_final:.2f} (Schwelle {thr_final:.2f})")
    ax1.legend(fontsize=8.5, loc="best", ncol=2)
    ax1.grid(alpha=0.3)
    finite = med[np.isfinite(med)]
    if finite.size:
        ax1.set_ylim(min(-0.2, finite.min() - 0.1), max(0.9, np.nanmax(hi) + 0.1))

    # --- unten: Groesse des ausgewerteten Tails ---
    n_tail = np.minimum(0.2 * ks, 3.0 * np.sqrt(ks)).astype(int)
    ax2.plot(ks, n_tail, color="slategray", lw=1.8)
    ax2.axhline(5, color="crimson", ls="--", lw=1.1, label="Minimum 5 Tail-Punkte")
    ax2.set_ylabel("Tail-Punkte für den GPD-Fit")
    ax2.set_xlabel("Stichprobenumfang  k  (Zahl der beruecksichtigten $w_i$)")
    ax2.legend(fontsize=9)
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
                    help="vorhandene Datei (.npy/.npz/.txt/.csv): fertige w_i ODER "
                         "Energie-Cache mit e_dft/e_mace (z.B. results/mace_energies_*.npz)")
    ap.add_argument("--source", default="real",
                    choices=["real", "lognormal", "gamma", "pareto",
                             "exponential", "halfnormal"])
    ap.add_argument("-n", "--n", type=int, default=2000, help="Anzahl Gewichte (synthetisch)")
    ap.add_argument("--step", type=int, default=0,
                    help="Schrittweite des k-Gitters (0 = automatisch, ~200 Punkte)")
    ap.add_argument("--shuffles", type=int, default=40,
                    help="Permutationen fuer das Band (1 = feste Reihenfolge, kein Band)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--alpha", type=float, default=2.0)
    ap.add_argument("--shape", type=float, default=2.0)
    ap.add_argument("--ensemble", default="ensemble_L2c",
                    choices=["ensemble_L2c", "ensemble_L0c", "ensemble_L0"])
    ap.add_argument("--testset", default="big", choices=["big", "small"])
    ap.add_argument("--temperature", type=float, default=300.0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    if args.weights:
        w = load_weights(args.weights, args.temperature)
        label = f"Datei: {Path(args.weights).name}"
        tag = Path(args.weights).stem
    elif args.source == "real":
        w = real_weights(args.ensemble, args.testset, args.temperature)
        label = f"real: {args.ensemble}, test{args.testset}, {args.temperature:.0f} K"
        tag = f"real_{args.ensemble}_test{args.testset}"
    else:
        w = synthetic_weights(args.source, args.n, rng, sigma=args.sigma,
                              alpha=args.alpha, shape=args.shape)
        label = f"{args.source} (n={args.n})"
        tag = f"{args.source}_n{args.n}"

    n = w.size
    if n < K_MIN:
        raise SystemExit(f"Nur {n} Gewichte - psis_khat braucht mindestens {K_MIN}.")

    step = args.step or max(1, (n - K_MIN) // 200)
    ks = np.arange(K_MIN, n + 1, step)
    if ks[-1] != n:
        ks = np.append(ks, n)

    print(f"[calc ] k_hat auf {ks.size} Stuetzstellen x {args.shuffles} Reihenfolgen ...")
    med, lo, hi = running_khat_over_shuffles(w, ks, args.shuffles, rng)

    khat_final = psis_khat(w)
    neff = effective_sample_size(w)
    rn = running_neff_cv(w)
    print(f"\nQuelle: {label}")
    print(f"  n = {n},  CV = {w.std()/w.mean():.3f}")
    print(f"  N_eff  = {neff:.1f}  ({neff/n*100:.1f}% von n),  laufend N_eff/k final = {rn['neff_ratio'][-1]:.3f}")
    print(f"  k_hat  = {khat_final:.3f}   (Schwelle {khat_threshold(n):.3f}) -> "
          f"{'VERLAESSLICH' if khat_final < khat_threshold(n) else 'UNZUVERLAESSIG'}")
    print(f"  k_hat Median bei k={ks[0]}: {med[0]:.3f},  bei k=n/2: {med[ks.size//2]:.3f},  bei k=n: {med[-1]:.3f}")

    root = Path(__file__).resolve().parents[1]
    plot(ks, med, lo, hi, w, label, root / "results" / f"khat_running_{tag}.png")


if __name__ == "__main__":
    main()
