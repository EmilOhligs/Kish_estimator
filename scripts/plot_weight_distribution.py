"""
Empirische Verteilung der Reweighting-Gewichte w_i (ECHTE Daten, nicht synthetisch)
============================================================================

Bisher plotten die Skripte im Repo nur *synthetische* Gewichtsfamilien
(Lognormal, Gamma, ...). Dieses Skript berechnet die TATSAECHLICHEN Gewichte

        w_i = exp(-beta * (E_DFT(R_i) - E_MACE(R_i)))          beta = 1/(k_B T)

aus den DFT-Energien des Testsatzes und den MACE-Ensemble-Vorhersagen und
zeigt, welche zugrundeliegende Verteilung die w_i haben.

Ablauf:
  1. water_testset_big.xyz laden -> E_DFT pro Frame (400 Frames, 189 Atome).
  2. L2c-Ensemble (2 Member, wie Hilpert & Kresse 2026) ueber alle Frames laufen
     lassen -> E_MACE pro Frame (Ensemble-Mittel) + Energien pro Member.
     Das laeuft ueber uq_mace.predictions.get_predictions, das denselben
     results/predictions_<ensemble>_test<testset>.npz-Cache nutzt wie
     evaluate_ensembles.py -> MACE-Inferenz nur EINMAL fuer beide Analysen
     (--force erzwingt Neuberechnung).
  3. dE = E_DFT - E_MACE, w_i = exp(-beta * dE). Da eine additive Konstante in
     dE weder N_eff noch die Form der Verteilung aendert, wird dE fuer die
     Darstellung um seinen Mittelwert zentriert.
  4. Plot (results/weight_distribution_L2c.png) mit 4 Panels + Diagnostik:
       CV(w), Kish-N_eff, N_eff/N, Pareto-k_hat, alpha_max, Energy/Force-RMSE.

Ausfuehren (im Projekt-Root, venv aktiv):
        python scripts/plot_weight_distribution.py
        python scripts/plot_weight_distribution.py --temperature 298 --force
        python scripts/plot_weight_distribution.py --ensemble ensemble_L0c
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.evaluation import energy_rmse_per_atom, force_rmse
from uq_mace.predictions import get_predictions
from uq_mace.reweighting import (
    effective_sample_size,
    khat_threshold,
    psis_khat,
    reweighting_weights,
    sample_overlap,
)

# Boltzmann-Konstante in eV/K (CODATA)
K_B = 8.617333262e-5


# ---------------------------------------------------------------------------
# 1) + 2)  Energien beschaffen (gemeinsamer Cache via uq_mace.predictions)
# ---------------------------------------------------------------------------
def get_energies(ensemble: str, testset: str, force: bool):
    """Gibt (e_dft, e_mace_mean, energies_per_member, n_atoms, rmse_dict) zurueck.

    Nutzt denselben results/predictions_*-Cache wie evaluate_ensembles.py.
    e_mace_mean : Ensemble-Mittel der Gesamtenergie pro Frame (eV)
    energies    : (n_member, n_frame) Energien der einzelnen Member (eV)
    """
    pred = get_predictions(ensemble, testset, force=force)
    e_dft, energies, n_atoms = pred["e_dft"], pred["energies"], pred["n_atoms"]
    e_mace = energies.mean(axis=0)                # Ensemble-Mittel

    # Sanity-Metriken (gegen Paper-Benchmarks pruefbar), aus denselben Zahlen
    f_mean = [np.mean(f, axis=0) for f in pred["forces"]]
    rmse = {
        "energy_meV_atom": energy_rmse_per_atom(e_mace, e_dft, n_atoms),
        "force_meV_A": force_rmse(f_mean, pred["f_ref"]),
    }
    return e_dft, e_mace, energies, n_atoms, rmse


# ---------------------------------------------------------------------------
# 3) + 4)  Gewichte, Diagnostik, Plot
# ---------------------------------------------------------------------------
def analyse_and_plot(e_dft, e_mace, beta, temperature, rmse, ensemble, out_png):
    dE = e_dft - e_mace                 # Gesamtenergie-Differenz pro Frame (eV)
    dE_c = dE - dE.mean()               # additive Konstante ist fuer w_i irrelevant
    n = dE.size

    # Gewichte (reweighting_weights zentriert intern um min -> identische Form/N_eff)
    w = reweighting_weights(e_dft, e_mace, beta)
    p = w / w.sum()                     # normierte Gewichte

    cv = w.std() / w.mean()
    neff = effective_sample_size(w)
    khat = psis_khat(w)
    khat_thr = khat_threshold(n)
    alpha_max = p.max()
    ovl = sample_overlap(w)

    # Gauss-Vorhersage N_eff = n * exp(-beta^2 Var(dE)) zum Vergleich
    neff_gauss = n * np.exp(-(beta ** 2) * dE.var())

    diag = (
        f"Ensemble {ensemble}  |  T = {temperature:.0f} K  (beta = {beta:.2f} eV$^{{-1}}$)  |  N = {n}\n"
        f"E-RMSE = {rmse['energy_meV_atom']:.3f} meV/Atom,  F-RMSE = {rmse['force_meV_A']:.2f} meV/A   |   "
        f"std($\\Delta E$) = {dE.std()*1000:.1f} meV,  beta·std($\\Delta E$) = {beta*dE.std():.2f}\n"
        f"CV(w) = {cv:.3f}   N_eff = {neff:.1f}  ({neff/n*100:.1f}% von N)   "
        f"N_eff(Gauss) = {neff_gauss:.1f}   $\\alpha_{{max}}$ = {alpha_max:.3f}   "
        f"OVL = {ovl:.3f}   $\\hat k$ = {khat:.2f} (Schwelle {khat_thr:.2f})"
    )

    print("\n" + "=" * 78)
    print(diag.replace("$", "").replace("^{-1}", "^-1").replace("\\Delta", "d").replace("\\alpha", "alpha").replace("\\hat k", "khat").replace("{", "").replace("}", ""))
    print("=" * 78 + "\n")

    fig, ax = plt.subplots(2, 2, figsize=(13, 9))

    # (a) dE-Verteilung mit Gauss-Fit
    a = ax[0, 0]
    a.hist(dE_c * 1000, bins=40, density=True, alpha=0.6, color="steelblue",
           label="$\\Delta E$ (zentriert)")
    xs = np.linspace((dE_c * 1000).min(), (dE_c * 1000).max(), 200)
    a.plot(xs, np.exp(-0.5 * (xs / (dE.std() * 1000)) ** 2) / (dE.std() * 1000 * np.sqrt(2 * np.pi)),
           "r--", lw=1.5, label="Gauss-Fit")
    a.set_xlabel("$\\Delta E = E_{DFT}-E_{MACE}$  [meV/Frame]")
    a.set_ylabel("Dichte")
    a.set_title("(a) Energiedifferenz  $\\Delta E$")
    a.legend(fontsize=9)
    a.grid(alpha=0.3)

    # (b) Verteilung der normierten Gewichte w/mean(w)  (linear)
    b = ax[0, 1]
    wn = w / w.mean()
    b.hist(wn, bins=50, density=True, alpha=0.6, color="darkorange")
    b.axvline(1.0, color="k", ls=":", lw=1, label="Mittel = 1")
    b.set_xlabel("$w_i / \\langle w \\rangle$")
    b.set_ylabel("Dichte")
    b.set_title("(b) Gewichtsverteilung (linear)")
    b.legend(fontsize=9)
    b.grid(alpha=0.3)

    # (c) gleiche Verteilung, log-y  -> Tail sichtbar
    c = ax[1, 0]
    c.hist(wn, bins=50, density=True, alpha=0.6, color="darkorange")
    c.set_yscale("log")
    c.set_xlabel("$w_i / \\langle w \\rangle$")
    c.set_ylabel("Dichte (log)")
    c.set_title("(c) Gewichtsverteilung (log-y, Tail)")
    c.grid(alpha=0.3, which="both")

    # (d) sortierte kumulative Gewichte (Lorenz-Kurve) -> Konzentration
    d = ax[1, 1]
    ps = np.sort(p)[::-1]
    cum = np.cumsum(ps)
    frac = np.arange(1, n + 1) / n
    d.plot(frac * 100, cum * 100, color="seagreen", lw=2)
    d.plot([0, 100], [0, 100], "k:", lw=1, label="Gleichverteilung")
    d.set_xlabel("Anteil der Frames [%] (nach Gewicht sortiert)")
    d.set_ylabel("kumuliertes Gewicht [%]")
    d.set_title("(d) Gewichtskonzentration (Lorenz)")
    d.legend(fontsize=9)
    d.grid(alpha=0.3)

    fig.suptitle("Empirische Verteilung der Reweighting-Gewichte $w_i$\n" + diag,
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140)
    print(f"[plot ] gespeichert -> {out_png}")

    return dict(cv=cv, neff=neff, neff_ratio=neff / n, neff_gauss=neff_gauss,
                khat=khat, khat_thr=khat_thr, alpha_max=alpha_max, ovl=ovl,
                dE_std_meV=dE.std() * 1000)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ensemble", default="ensemble_L2c",
                    choices=["ensemble_L2c", "ensemble_L0c", "ensemble_L0"],
                    help="Modell-Ensemble fuer E_MACE (default: ensemble_L2c wie Paper)")
    ap.add_argument("--testset", default="big", choices=["big", "small"],
                    help="big = water_testset_big.xyz (400, 189 Atome); small = 125 gemischt")
    ap.add_argument("--temperature", type=float, default=300.0,
                    help="Temperatur in K fuer beta = 1/(k_B T) (default 300)")
    ap.add_argument("--force", action="store_true",
                    help="Energien neu berechnen statt Cache zu nutzen")
    args = ap.parse_args()

    beta = 1.0 / (K_B * args.temperature)
    root = Path(__file__).resolve().parents[1]
    out_png = root / "results" / f"weight_distribution_{args.ensemble}_test{args.testset}.png"

    e_dft, e_mace, energies, n_atoms, rmse = get_energies(
        args.ensemble, args.testset, args.force
    )
    analyse_and_plot(e_dft, e_mace, beta, args.temperature, rmse, args.ensemble, out_png)


if __name__ == "__main__":
    main()
