"""
ΔE-Vergleich über alle Ensembles: L0, L0c, L2c
============================================================================

Laesst jedes Ensemble ueber denselben Testsatz laufen und vergleicht die
Energiedifferenzen dE = E_DFT - E_MACE. Damit werden drei Fragen auf einmal
beantwortet, die bisher nur an L2c haengen:

  1. SKALIERT c MIT DER MODELLQUALITAET?
     L0 ist schlechter als L2c, sollte also groesseres std(dE) und damit
     groesseres c = beta*std(dE) haben.

  2. GILT DER c-FORMALISMUS UEBER MODELLQUALITAETEN HINWEG?
     Der eigentliche Test: liegen alle Ensembles auf der Kurve
     N_eff/n = exp(-c^2)? Wenn ja, ist c tatsaechlich DER Parameter und nicht
     nur eine an L2c angepasste Groesse. Abweichungen nach oben sind laut
     Kumulantenentwicklung durch die Schiefe erklaerbar (+gamma_1 c^3).

  3. IST DER ENSEMBLE-GAP STABIL?
     `neff_leave_one_out` unterschaetzt die Fehlervarianz, weil der allen
     Membern gemeinsame Bias unsichtbar bleibt (bei L2c: Faktor 2.39).
     L0 und L0c haben je 3 Member, der LOO-Schaetzer ist dort deutlich besser
     konditioniert als bei M=2. Bleibt der Faktor stabil, ist er strukturell
     und damit als KALIBRIERUNGSKONSTANTE brauchbar - dann funktioniert der
     DFT-freie Pfad doch. Schrumpft er mit M, war er teils Rauschen.

ACHTUNG - Rechenzeit: fuer noch nicht gecachte Ensembles laeuft MACE ueber alle
Frames (L0/L0c: je 3 Member x 400 Frames). Auf CPU sind das pro Ensemble
typischerweise 20-60 min, mit --device cuda deutlich weniger. Danach liegt alles
in cache/ und der Vergleich ist in Sekunden wiederholbar.

Ausfuehren:
    python analyses/09_ensemble_evaluation/compare_delta_e.py
    python analyses/09_ensemble_evaluation/compare_delta_e.py --device cuda
    python analyses/09_ensemble_evaluation/compare_delta_e.py --only ensemble_L2c
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew

from uq_mace.evaluation import energy_rmse_per_atom, force_rmse
from uq_mace.predictions import get_predictions
from uq_mace.reweighting import (
    effective_sample_size,
    khat_threshold,
    neff_leave_one_out,
    psis_khat,
    reweighting_weights,
)

HERE = Path(__file__).resolve().parent
K_B = 8.617333262e-5  # eV/K


def discover_ensembles() -> list[str]:
    """Alle Ensembles unter models/ finden, die mindestens einen Checkpoint haben.

    Bewusst dynamisch statt hartkodiert: neu trainierte Member (z.B. die drei
    zusaetzlichen L2c-Seeds aus train_ensemble.py) werden so automatisch
    mitgenommen, ohne dass dieses Skript angefasst werden muss. Sortiert nach
    Membernzahl, damit die statistisch besser konditionierten Ensembles zuerst
    stehen.
    """
    from uq_mace.ensemble import MODELS_DIR

    found = [(len(list(d.glob("*.model"))), d.name)
             for d in sorted(MODELS_DIR.iterdir()) if d.is_dir()]
    return [name for k, name in sorted(found, reverse=True) if k > 0]


# ---------------------------------------------------------------------------
def analyse(ensemble: str, testset: str, temperature: float, device: str) -> dict | None:
    """Alle Kennzahlen fuer ein Ensemble. None, wenn Modelle fehlen."""
    from uq_mace.ensemble import MODELS_DIR

    if not (MODELS_DIR / ensemble).exists():
        print(f"  {ensemble}: uebersprungen (Modelle nicht gefunden)")
        return None

    pred = get_predictions(ensemble, testset, device=device)
    e_dft = pred["e_dft"]
    energies = pred["energies"]          # (M, F)
    e_mace = energies.mean(axis=0)
    dE = e_dft - e_mace
    n = dE.size
    beta = 1.0 / (K_B * temperature)

    w = reweighting_weights(e_dft, e_mace, beta)
    neff = effective_sample_size(w)
    c = beta * dE.std(ddof=1)
    g1, g2 = float(skew(dE)), float(kurtosis(dE))

    # DFT-freie Prognose aus der Inter-Member-Streuung
    neff_loo = neff_leave_one_out(energies, beta)
    # zugehoerige vorhergesagte Streuung: N_eff = n exp(-beta^2 var)
    var_pred = -np.log(max(neff_loo / n, 1e-300)) / beta ** 2
    std_pred = float(np.sqrt(max(var_pred, 0.0)))

    f_mean = [np.mean(f, axis=0) for f in pred["forces"]]
    return dict(
        ensemble=ensemble, members=energies.shape[0], n=n,
        std_dE_meV=dE.std(ddof=1) * 1000, c=c, gamma1=g1, gamma2=g2,
        neff=neff, neff_ratio=neff / n,
        neff_gauss_ratio=float(np.exp(-c * c)),
        khat=psis_khat(w), khat_thr=khat_threshold(n),
        neff_loo=neff_loo, neff_loo_ratio=neff_loo / n,
        std_pred_meV=std_pred * 1000,
        gap=float(dE.std(ddof=1) / std_pred) if std_pred > 0 else np.nan,
        e_rmse=energy_rmse_per_atom(e_mace, e_dft, pred["n_atoms"]),
        f_rmse=force_rmse(f_mean, pred["f_ref"]),
        dE=dE,
    )


# ---------------------------------------------------------------------------
def make_plot(rows: list[dict], temperature: float, out_png: Path):
    cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(rows)))
    fig, ax = plt.subplots(2, 2, figsize=(14, 10))
    (a1, a2), (a3, a4) = ax

    # (a) dE-Verteilungen, zentriert
    for r, col in zip(rows, cols):
        d = (r["dE"] - r["dE"].mean()) * 1000
        a1.hist(d, bins=40, density=True, alpha=0.45, color=col,
                label=f"{r['ensemble']}  std={r['std_dE_meV']:.1f} meV")
    a1.set_xlabel("$\\Delta E$ (zentriert) [meV/Frame]")
    a1.set_ylabel("Dichte")
    a1.set_title("(a) Energiedifferenzen je Ensemble")
    a1.legend(fontsize=8.5); a1.grid(alpha=0.3)

    # (b) DER TEST: liegen alle auf N_eff/n = exp(-c^2)?
    cs = np.linspace(0, max(0.9, max(r["c"] for r in rows) * 1.3), 200)
    a2.plot(cs, np.exp(-cs ** 2), "k-", lw=2, label="Gauß:  $e^{-c^2}$")
    for r, col in zip(rows, cols):
        a2.plot(r["c"], r["neff_ratio"], "o", ms=13, color=col,
                label=f"{r['ensemble']}  (c={r['c']:.3f})")
        a2.annotate(f"{r['neff_ratio']:.3f}", (r["c"], r["neff_ratio"]),
                    textcoords="offset points", xytext=(9, 7), fontsize=8.5)
    a2.set_xlabel("$c = \\beta\\cdot$std$(\\Delta E)$")
    a2.set_ylabel("$N_{eff}/n$ (Kish, gemessen)")
    a2.set_title("(b) Fallen alle Modellqualitäten auf dieselbe Kurve?")
    a2.set_ylim(0, 1.02); a2.legend(fontsize=8.5); a2.grid(alpha=0.3)

    # (c) k_hat je Ensemble gegen die Gates
    x = np.arange(len(rows))
    a3.bar(x, [r["khat"] for r in rows], color=cols, alpha=0.85, width=0.55)
    a3.axhline(0.5, color="crimson", ls="--", lw=1.3, label="0.5 — Existenz von E[w²]")
    a3.axhline(0.25, color="darkorange", ls=":", lw=1.3, label="0.25 — Fehlerbalken")
    a3.axhline(0.0, color="k", lw=0.8)
    a3.set_xticks(x); a3.set_xticklabels([r["ensemble"].replace("ensemble_", "") for r in rows])
    a3.set_ylabel("$\\hat{k}$")
    a3.set_title("(c) Tail-Index — bleiben die Gates offen?")
    a3.legend(fontsize=8.5); a3.grid(alpha=0.3, axis="y")

    # (d) Ensemble-Gap: gemessene vs. aus Member-Streuung prognostizierte std
    wdt = 0.36
    a4.bar(x - wdt / 2, [r["std_dE_meV"] for r in rows], wdt,
           label="gemessen (mit DFT)", color="steelblue")
    a4.bar(x + wdt / 2, [r["std_pred_meV"] for r in rows], wdt,
           label="LOO-Prognose (ohne DFT)", color="darkorange")
    for i, r in enumerate(rows):
        a4.text(i, max(r["std_dE_meV"], r["std_pred_meV"]) * 1.04,
                f"Gap {r['gap']:.2f}×\n(M={r['members']})", ha="center", fontsize=9,
                fontweight="bold")
    a4.set_xticks(x); a4.set_xticklabels([r["ensemble"].replace("ensemble_", "") for r in rows])
    a4.set_ylabel("std($\\Delta E$) [meV]")
    a4.set_title("(d) Ensemble-Blindfleck — ist der Gap stabil?")
    a4.legend(fontsize=8.5); a4.grid(alpha=0.3, axis="y")

    fig.suptitle(f"ΔE-Vergleich über Ensembles   |   T = {temperature:.0f} K   |   "
                 f"n = {rows[0]['n']} Frames", fontsize=13)
    fig.subplots_adjust(top=0.92, hspace=0.28, wspace=0.22,
                        left=0.07, right=0.97, bottom=0.07)
    fig.savefig(out_png, dpi=140)
    print(f"\n[plot ] {out_png}")


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--testset", default="big", choices=["big", "small"])
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--device", default="cpu",
                    help="cpu | cuda (NVIDIA, z.B. VSC) | mps (Apple Silicon)")
    ap.add_argument("--only", default=None,
                    help="Komma-Liste statt aller gefundenen Ensembles")
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    names = ([s.strip() for s in args.only.split(",")] if args.only
             else discover_ensembles())
    if not names:
        raise SystemExit("Keine Ensembles mit .model-Dateien unter models/ gefunden.")
    print(f"[setup] gefundene Ensembles: {', '.join(names)}   (device={args.device})")
    outdir = Path(args.outdir) if args.outdir else HERE

    rows = []
    for name in names:
        print(f"[eval ] {name} ...")
        r = analyse(name, args.testset, args.temperature, args.device)
        if r is not None:
            rows.append(r)
    if not rows:
        raise SystemExit("Kein Ensemble auswertbar.")

    # ---- Tabelle ----
    print("\n" + "=" * 118)
    print(f"{'Ensemble':<14}{'M':>3}{'std(dE)':>10}{'c':>8}{'gamma1':>8}"
          f"{'N_eff/n':>9}{'Gauss':>8}{'khat':>7}{'E-RMSE':>9}{'F-RMSE':>9}"
          f"{'LOO N/n':>9}{'Gap':>7}")
    print(f"{'':14}{'':>3}{'[meV]':>10}{'':>8}{'':>8}{'':>9}{'e^-c²':>8}{'':>7}"
          f"{'meV/At':>9}{'meV/Å':>9}{'ohne DFT':>9}{'':>7}")
    print("-" * 118)
    for r in rows:
        print(f"{r['ensemble']:<14}{r['members']:>3}{r['std_dE_meV']:>10.2f}{r['c']:>8.3f}"
              f"{r['gamma1']:>+8.3f}{r['neff_ratio']:>9.3f}{r['neff_gauss_ratio']:>8.3f}"
              f"{r['khat']:>7.3f}{r['e_rmse']:>9.3f}{r['f_rmse']:>9.2f}"
              f"{r['neff_loo_ratio']:>9.3f}{r['gap']:>7.2f}")
    print("=" * 118)

    print("\nLesehilfe:")
    print("  N_eff/n vs. Gauss  -> Differenz sollte ~ gamma1*c^3 sein (Schiefe-Korrektur).")
    print("  LOO N/n            -> DFT-freie Prognose; systematisch zu optimistisch.")
    print("  Gap                -> gemessene / prognostizierte std. Stabil ueber Ensembles")
    print("                        = strukturell = als Kalibrierungsfaktor brauchbar.")
    if len(rows) > 1:
        gaps = [r["gap"] for r in rows if np.isfinite(r["gap"])]
        print(f"\n  Gap ueber {len(gaps)} Ensembles: {min(gaps):.2f} .. {max(gaps):.2f}"
              f"  (Spannweite {max(gaps)-min(gaps):.2f})")
        for r in rows:
            dev = (r["neff_ratio"] / r["neff_gauss_ratio"] - 1) * 100
            pred = (r["gamma1"] * r["c"] ** 3 - 7 / 12 * r["gamma2"] * r["c"] ** 4) * 100
            print(f"  {r['ensemble']:<14} Gauss-Abweichung {dev:+6.2f} %   "
                  f"aus Kumulanten erwartet {pred:+6.2f} %")

    # ---- CSV ----
    csv_path = outdir / "compare_delta_e.csv"
    keys = [k for k in rows[0] if k != "dE"]
    with open(csv_path, "w", newline="") as fh:
        wtr = csv.DictWriter(fh, fieldnames=keys)
        wtr.writeheader()
        for r in rows:
            wtr.writerow({k: r[k] for k in keys})
    print(f"[csv  ] {csv_path}")

    make_plot(rows, args.temperature, outdir / "compare_delta_e.png")


if __name__ == "__main__":
    main()
