"""
N_eff- und k_hat-Verlauf ueber das gesamte N_eff-Spektrum (10 Verteilungen)
============================================================================

Fuer 10 Gewichtsverteilungen, die das N_eff-Spektrum von N_eff/n ~ 1 bis ~0
abdecken, wird bei JEDEM hinzukommenden w_k iterativ geschaetzt:

    CV_k     = std(w_1..w_k)/mean(w_1..w_k)
    N_eff_k  = k / (1 + CV_k^2)        (= Kish auf den ersten k, exakt)
    k_hat_k  = Pareto-Tail-Index des oberen Tails von w_1..w_k

Pro Verteilung wird EIN eigener Plot erzeugt, und pro Verteilung werden
--orders (Default 10) verschiedene Reihenfolgen des Sampelns gezeichnet.
n = 5000 Gewichte pro Verteilung (Default).

Warum beides zusammen: N_eff misst die STREUUNG der Gewichte, k_hat die
SCHWERE des Randes. Zwei Verteilungen koennen dasselbe N_eff haben und trotzdem
voellig verschieden verlaesslich sein (vgl. Gamma vs. Pareto weiter unten) -
genau das macht der Vergleich der beiden Kurven sichtbar.

Reihenfolgen: per Default werden dieselben 5000 Gewichte 10-mal PERMUTIERT
(gleiche Verteilung, andere Ankunftsreihenfolge) - deshalb laufen alle Kurven
bei k = n exakt zusammen. Mit --resample werden stattdessen 10 unabhaengige
Stichproben gezogen, dann streuen auch die Endwerte.

Laufzeit: k_hat wird auf einem log-Gitter (~--grid Stuetzstellen) statt bei
jedem k ausgewertet; 10 Verteilungen x 10 Reihenfolgen x ~200 Stuetzstellen
sind einige zehn Sekunden. N_eff laeuft vektorisiert in voller Aufloesung.

Ausfuehren:
    python scripts/neff_khat_spectrum.py
    python scripts/neff_khat_spectrum.py -n 5000 --orders 10 --grid 250
    python scripts/neff_khat_spectrum.py --resample --seed 1
    python scripts/neff_khat_spectrum.py --only lognormal_s2,pareto_a125

Ausgabe: results/neff_khat_spectrum/<name>.png  je Verteilung
         results/neff_khat_spectrum/summary.csv  Kennzahlen aller Verteilungen
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.reweighting import (
    effective_sample_size,
    khat_threshold,
    psis_khat,
    running_neff_cv,
)

K_MIN = 25  # psis_khat liefert darunter NaN


# ---------------------------------------------------------------------------
# 10 Verteilungen, geordnet nach fallendem theoretischem N_eff/n = 1/(1+CV^2).
# 'cv' und 'khat' sind die ASYMPTOTISCHEN Erwartungswerte (nur zur Orientierung
# im Plot; gemessen wird immer die tatsaechliche Stichprobe).
# ---------------------------------------------------------------------------
DISTRIBUTIONS: dict[str, dict] = {
    "konstant": dict(
        label="Konstant (w=1)",
        sampler=lambda rng, n: np.ones(n),
        cv=0.0, khat=0.0,
        note="Referenz: alle Gewichte gleich -> perfektes Reweighting.",
    ),
    "normal_s02": dict(
        label="Normal(1, 0.2), positiv beschnitten",
        sampler=lambda rng, n: np.clip(rng.normal(1.0, 0.2, n), 1e-12, None),
        cv=0.20, khat=0.0,
        note="Schmale Glocke: fast gleiche Gewichte, leichter Rand.",
    ),
    "gamma_k4": dict(
        label="Gamma(k=4)",
        sampler=lambda rng, n: rng.gamma(4.0, 1.0, n),
        cv=0.50, khat=0.0,
        note="Exponentiell abfallender Rand -> k_hat -> 0.",
    ),
    "uniform01": dict(
        label="Uniform(0, 1)",
        sampler=lambda rng, n: rng.uniform(0.0, 1.0, n),
        cv=1.0 / np.sqrt(3.0), khat=-1.0,
        note="Nach oben BESCHRAENKT -> k_hat negativ (leichtester Fall).",
    ),
    "halfnormal": dict(
        label="Half-Normal",
        sampler=lambda rng, n: np.abs(rng.normal(0.0, 1.0, n)),
        cv=np.sqrt(np.pi / 2.0 - 1.0), khat=0.0,
        note="Gauss-Rand, also sehr leicht.",
    ),
    "pareto_a25": dict(
        label="Pareto(alpha=2.5)",
        sampler=lambda rng, n: rng.pareto(2.5, n) + 1.0,
        cv=np.sqrt(1.0 / (2.5 * 0.5)), khat=1.0 / 2.5,
        note="Gegenstueck zu Exponential: AEHNLICHES N_eff, aber k_hat=0.4 "
             "-> gleiche Streuung, deutlich schwererer Rand.",
    ),
    "exponential": dict(
        label="Exponential(1)",
        sampler=lambda rng, n: rng.exponential(1.0, n),
        cv=1.0, khat=0.0,
        note="Grenzfall leichter Rand bei CV=1 -> N_eff/n = 1/2.",
    ),
    "lognormal_s1": dict(
        label="Lognormal(sigma=1)",
        sampler=lambda rng, n: rng.lognormal(0.0, 1.0, n),
        cv=np.sqrt(np.exp(1.0) - 1.0), khat=0.0,
        note="Formal k_hat->0, konvergiert aber sehr langsam.",
    ),
    "lognormal_s15": dict(
        label="Lognormal(sigma=1.5)",
        sampler=lambda rng, n: rng.lognormal(0.0, 1.5, n),
        cv=np.sqrt(np.exp(2.25) - 1.0), khat=0.0,
        note="Starke Konzentration: wenige Frames tragen fast das ganze Gewicht.",
    ),
    "pareto_a125": dict(
        label="Pareto(alpha=1.25)",
        sampler=lambda rng, n: (rng.pareto(1.25, n) + 1.0),
        cv=np.inf, khat=1.0 / 1.25,
        note="UNENDLICHE Varianz: N_eff-Schaetzung konvergiert nie, "
             "k_hat=0.8 > 0.7 flaggt das korrekt.",
    ),
}


# ---------------------------------------------------------------------------
def make_kgrid(n: int, n_points: int) -> np.ndarray:
    """Log-Gitter von K_MIN bis n (k_hat ist zu teuer fuer jedes einzelne k)."""
    grid = np.geomspace(K_MIN, n, n_points)
    grid = np.unique(np.round(grid).astype(int))
    grid = grid[(grid >= K_MIN) & (grid <= n)]
    if grid[-1] != n:
        grid = np.append(grid, n)
    return grid


def running_khat(w: np.ndarray, kgrid: np.ndarray) -> np.ndarray:
    """k_hat auf den ersten k Gewichten, fuer jedes k im Gitter."""
    return np.array([psis_khat(w[:k]) for k in kgrid])


def run_one(spec: dict, n: int, orders: int, kgrid: np.ndarray, rng,
            resample: bool):
    """Liefert Kurvenscharen fuer eine Verteilung.

    Rueckgabe:
        w0      : die (erste) Stichprobe, fuer das Histogramm
        ratios  : (orders, n)      laufendes N_eff/k
        khats   : (orders, kgrid)  laufendes k_hat
        finals  : Liste der Endwerte (cv, neff, ratio, khat) je Reihenfolge
    """
    w0 = spec["sampler"](rng, n)
    ratios = np.empty((orders, n))
    khats = np.empty((orders, kgrid.size))
    finals = []

    for o in range(orders):
        if resample:
            w = spec["sampler"](rng, n)          # unabhaengige Stichprobe
        elif o == 0:
            w = w0                                # erste Reihenfolge: Originallage
        else:
            w = rng.permutation(w0)               # gleiche Menge, andere Reihenfolge

        ratios[o] = running_neff_cv(w)["neff_ratio"]
        khats[o] = running_khat(w, kgrid)

        neff = effective_sample_size(w)
        finals.append(dict(cv=float(w.std() / w.mean()) if w.mean() > 0 else np.inf,
                           neff=float(neff), ratio=float(neff / n),
                           khat=float(psis_khat(w))))
    return w0, ratios, khats, finals


# ---------------------------------------------------------------------------
def plot_one(name: str, spec: dict, w0, ratios, khats, finals, kgrid, n,
             orders: int, resample: bool, out_png: Path):
    k_full = np.arange(1, n + 1)
    thr = khat_threshold(n)
    ratio_mean = float(np.mean([f["ratio"] for f in finals]))
    khat_mean = float(np.nanmean([f["khat"] for f in finals]))
    cv_mean = float(np.mean([f["cv"] for f in finals]))
    colors = plt.cm.viridis(np.linspace(0, 0.88, orders))
    order_lbl = "unabh. Stichproben" if resample else "Reihenfolgen"

    fig = plt.figure(figsize=(11, 10))
    gs = fig.add_gridspec(3, 1, height_ratios=[3, 3, 2], hspace=0.32)

    # --- (a) laufendes N_eff/k -------------------------------------------
    ax1 = fig.add_subplot(gs[0])
    for o in range(orders):
        ax1.plot(k_full, ratios[o], color=colors[o], lw=1.0, alpha=0.75,
                 label=f"{orders} {order_lbl}" if o == 0 else None)
    ax1.axhline(ratio_mean, color="firebrick", ls="--", lw=1.5,
                label=f"Endwert (Kish) = {ratio_mean:.3f}")
    if np.isfinite(spec["cv"]):
        ax1.axhline(1.0 / (1.0 + spec["cv"] ** 2), color="black", ls=":", lw=1.2,
                    label=f"theoretisch = {1.0/(1.0+spec['cv']**2):.3f}")
    ax1.set_xscale("log")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("$N_{eff}/k$   (aus CV)")
    ax1.set_title("(a) Iterative $N_{eff}$-Schaetzung")
    ax1.legend(fontsize=8.5, loc="best")
    ax1.grid(alpha=0.3, which="both")

    # --- (b) laufendes k_hat ---------------------------------------------
    ax2 = fig.add_subplot(gs[1], sharex=ax1)
    for o in range(orders):
        ax2.plot(kgrid, khats[o], color=colors[o], lw=1.0, alpha=0.75, marker="",
                 label=f"{orders} {order_lbl}" if o == 0 else None)
    ax2.axhline(0.7, color="k", ls="--", lw=1.3, label="$\\hat{k}=0.7$ (darueber unzuverlaessig)")
    ax2.axhline(0.5, color="seagreen", ls=":", lw=1.1, label="$\\hat{k}=0.5$ (gut)")
    ax2.plot(kgrid, [khat_threshold(int(k)) for k in kgrid], color="steelblue",
             ls=":", lw=1.3, label="Schwelle $\\min(1-1/\\log_{10}S,\\,0.7)$")
    if np.isfinite(spec["khat"]):
        ax2.axhline(spec["khat"], color="darkorange", ls="-.", lw=1.2,
                    label=f"theoretisch $\\hat{{k}}$ = {spec['khat']:.2f}")
    ax2.set_xscale("log")
    ax2.set_ylabel("$\\hat{k}$  (Pareto-Tail-Index)")
    ax2.set_title("(b) Iterativer Pareto-Tail-Index")
    ax2.legend(fontsize=8, loc="best", ncol=2)
    ax2.grid(alpha=0.3, which="both")

    # --- (c) Histogramm der Gewichte -------------------------------------
    ax3 = fig.add_subplot(gs[2])
    hi = np.quantile(w0, 0.995)
    shown = w0[w0 <= hi] if hi > 0 else w0
    if np.allclose(w0, w0.flat[0]):
        ax3.axvline(float(w0.flat[0]), color="C2", lw=2)
        ax3.set_xlim(float(w0.flat[0]) - 1, float(w0.flat[0]) + 1)
    else:
        ax3.hist(shown, bins=60, density=True, alpha=0.65, color="steelblue")
        ax3.set_yscale("log")
    ax3.set_xlabel("$w$   (Tail bei 99.5-%-Quantil abgeschnitten)")
    ax3.set_ylabel("Dichte (log)")
    ax3.set_title("(c) Verteilung der Gewichte")
    ax3.grid(alpha=0.3, which="both")

    ax2.set_xlabel("Stichprobenumfang  k")

    fig.suptitle(
        f"{spec['label']}   |   n = {n},  {orders} {order_lbl}\n"
        f"gemessen: CV = {cv_mean:.3f},  $N_{{eff}}/n$ = {ratio_mean:.3f},  "
        f"$\\hat{{k}}$ = {khat_mean:.3f} (Schwelle {thr:.2f}) -> "
        f"{'verlaesslich' if khat_mean < thr else 'UNZUVERLAESSIG'}\n"
        f"{spec['note']}",
        fontsize=10.5)
    fig.subplots_adjust(top=0.86, bottom=0.07, left=0.09, right=0.97)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
    print(f"  [plot ] {out_png.name}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--n", type=int, default=5000, help="Gewichte pro Verteilung")
    ap.add_argument("--orders", type=int, default=10,
                    help="Zahl der Sampling-Reihenfolgen pro Verteilung")
    ap.add_argument("--grid", type=int, default=200,
                    help="Stuetzstellen des log-k-Gitters fuer k_hat")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resample", action="store_true",
                    help="statt zu permutieren pro Reihenfolge neu ziehen "
                         "(dann streuen auch die Endwerte)")
    ap.add_argument("--only", default=None,
                    help="Komma-Liste von Verteilungsnamen (Default: alle 10)")
    ap.add_argument("--outdir", default="results/neff_khat_spectrum")
    args = ap.parse_args()

    names = list(DISTRIBUTIONS)
    if args.only:
        wanted = [s.strip() for s in args.only.split(",") if s.strip()]
        unknown = [s for s in wanted if s not in DISTRIBUTIONS]
        if unknown:
            raise SystemExit(f"Unbekannte Verteilung(en): {unknown}\nVerfuegbar: {names}")
        names = wanted

    root = Path(__file__).resolve().parents[1]
    outdir = root / args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    kgrid = make_kgrid(args.n, args.grid)
    print(f"n = {args.n},  {args.orders} Reihenfolgen,  "
          f"{kgrid.size} k_hat-Stuetzstellen (log, {kgrid[0]}..{kgrid[-1]})")
    print(f"Ausgabe -> {outdir}\n")

    rows = []
    for idx, name in enumerate(names, 1):
        spec = DISTRIBUTIONS[name]
        # eigener Seed je Verteilung -> reproduzierbar und unabhaengig
        rng = np.random.default_rng(args.seed + 1000 * idx)
        print(f"[{idx}/{len(names)}] {spec['label']}")

        w0, ratios, khats, finals = run_one(spec, args.n, args.orders, kgrid,
                                            rng, args.resample)
        plot_one(name, spec, w0, ratios, khats, finals, kgrid, args.n,
                 args.orders, args.resample, outdir / f"{name}.png")

        cv_m = float(np.mean([f["cv"] for f in finals]))
        ratio_m = float(np.mean([f["ratio"] for f in finals]))
        neff_m = float(np.mean([f["neff"] for f in finals]))
        khat_m = float(np.nanmean([f["khat"] for f in finals]))
        thr = khat_threshold(args.n)
        rows.append(dict(name=name, label=spec["label"], cv=round(cv_m, 4),
                         neff=round(neff_m, 1), neff_ratio=round(ratio_m, 4),
                         khat=round(khat_m, 4), khat_threshold=round(thr, 4),
                         reliable=bool(khat_m < thr)))
        print(f"  CV={cv_m:.3f}  N_eff={neff_m:.0f} ({ratio_m*100:.1f}%)  "
              f"khat={khat_m:.3f} -> {'ok' if khat_m < thr else 'UNZUVERLAESSIG'}")

    # ---- Zusammenfassung ------------------------------------------------
    csv_path = outdir / "summary.csv"
    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("\n" + "=" * 92)
    print(f"{'Verteilung':<34}{'CV':>8}{'N_eff':>10}{'N_eff/n':>10}{'k_hat':>9}{'Urteil':>14}")
    print("-" * 92)
    for r in sorted(rows, key=lambda r: -r["neff_ratio"]):
        print(f"{r['label']:<34}{r['cv']:>8.3f}{r['neff']:>10.0f}"
              f"{r['neff_ratio']:>10.3f}{r['khat']:>9.3f}"
              f"{'ok' if r['reliable'] else 'UNZUVERLAESSIG':>14}")
    print("=" * 92)
    print(f"CSV -> {csv_path}")
    print("\nLesart: N_eff/n = 1/(1+CV^2) haengt NUR von der Streuung ab, k_hat NUR")
    print("vom Rand. Vergleiche Exponential und Pareto(2.5): aehnliches N_eff,")
    print("aber deutlich verschiedenes k_hat -> N_eff allein sagt nichts ueber die")
    print("Verlaesslichkeit der Schaetzung aus.")


if __name__ == "__main__":
    main()
