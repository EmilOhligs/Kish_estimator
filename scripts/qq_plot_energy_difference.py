"""
QQ-Plot der Energiedifferenzen dE = E_DFT - E_MACE
============================================================================
 
Prueft, ob die Energiedifferenzen des L2c-Modells auf dem 400-Frame-Testsatz
NORMALVERTEILT sind. Das ist keine kosmetische Frage, sondern die zentrale
Annahme hinter dem DFT-freien N_eff-Praediktor:
 
        N_eff = n * exp(-beta^2 * Var(dE))        (predicted_neff_gauss)
 
Diese Formel folgt aus <exp(-beta*dE)> fuer dE ~ Normal. Haelt die Normalitaet
nicht - insbesondere in den RAENDERN - dann ueberschaetzt sie N_eff, weil
w = exp(-beta*dE) Abweichungen im Tail exponentiell verstaerkt.
 
Panels:
  (a) Normal-QQ-Plot mit Referenzgerade (Mittel/Std) und 95-%-Monte-Carlo-
      Envelope: liegen die Punkte innerhalb des Bands, ist die Abweichung mit
      reiner Stichprobenstreuung vertraeglich.
  (b) Histogramm mit Normal-Fit.
  (c) QQ-Plot der Gewichte-relevanten Groesse beta*dE, zusaetzlich gegen eine
      Student-t-Referenz - zeigt, ob schwere Raender besser passen.
 
Kennzahlen: Mittel, Std, Schiefe, Exzess-Kurtosis, Shapiro-Wilk, Anderson-Darling.
 
Ausfuehren:
    python scripts/qq_plot_energy_difference.py
    python scripts/qq_plot_energy_difference.py \
        --energies results/mace_energies_ensemble_L2c_testbig.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

K_B = 8.617333262e-5  # eV/K


def get_delta_e(args) -> tuple[np.ndarray, str]:
    """dE = E_DFT - E_MACE (in eV) aus Cache-Datei oder ueber predictions."""
    from uq_mace.predictions import load_energies

    if args.energies:
        e_dft, e_mace = load_energies(args.energies)
        label = Path(args.energies).name
    else:
        from uq_mace.predictions import cache_path, get_predictions

        p = cache_path(args.ensemble, args.testset)
        if p.exists():
            e_dft, e_mace = load_energies(p)
            label = p.name
        else:  # kein Cache -> MACE laeuft (einmalig)
            pred = get_predictions(args.ensemble, args.testset)
            e_dft, e_mace = pred["e_dft"], pred["energies"].mean(axis=0)
            label = f"{args.ensemble}/test{args.testset}"
    return e_dft - e_mace, label


def normal_envelope(n: int, reps: int, rng, q_theo: np.ndarray):
    """95-%-Monte-Carlo-Envelope fuer einen Normal-QQ-Plot (standardisiert)."""
    sims = np.sort(rng.standard_normal((reps, n)), axis=1)
    return np.percentile(sims, 2.5, axis=0), np.percentile(sims, 97.5, axis=0)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--energies", default=None,
                    help="npz mit e_dft und e_mace/energies "
                         "(default: predictions-Cache des gewaehlten Ensembles)")
    ap.add_argument("--ensemble", default="ensemble_L2c",
                    choices=["ensemble_L2c", "ensemble_L0c", "ensemble_L0"])
    ap.add_argument("--testset", default="big", choices=["big", "small"])
    ap.add_argument("--temperature", type=float, default=300.0)
    ap.add_argument("--reps", type=int, default=2000, help="Monte-Carlo-Ziehungen fuer die Envelope")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    dE, label = get_delta_e(args)
    n = dE.size
    beta = 1.0 / (K_B * args.temperature)

    mu, sd = dE.mean(), dE.std(ddof=1)
    z = (dE - mu) / sd                     # standardisiert
    skew = stats.skew(dE)
    kurt = stats.kurtosis(dE)              # Exzess (0 = normal)
    sw_stat, sw_p = stats.shapiro(dE)
    ad = stats.anderson(dE, dist="norm")
    # scipy.stats.anderson may return an object with attributes or a tuple-like
    # scipy.stats.anderson may return an object with attributes or a tuple-like
    # Use hasattr/getattr to support both styles and avoid attribute access issues
    if hasattr(ad, "statistic"):
        ad_stat = getattr(ad, "statistic")
        crit_vals = getattr(ad, "critical_values")
        sig_levels = getattr(ad, "significance_level")
    else:
        # fallback for older scipy returning (statistic, critical_values, significance_level)
        ad_stat = ad[0]
        crit_vals = ad[1]
        sig_levels = ad[2]
    # critical_values and significance_level are sequence-like; pick 5% entry
    # Support different return types (lists, numpy arrays, object-like)
    sig_arr = np.asarray(sig_levels)
    # find index of the entry closest to 5.0 (in case of floats or different ordering)
    idx = int(np.argmin(np.abs(sig_arr - 5.0)))
    ad_crit5 = np.asarray(crit_vals)[idx]

    # Plotting-Positionen (Blom) und theoretische Quantile
    i = np.arange(1, n + 1)
    pp = (i - 0.375) / (n + 0.25)
    q_theo = stats.norm.ppf(pp)
    q_emp = np.sort(z)
    lo, hi = normal_envelope(n, args.reps, rng, q_theo)

    print(f"\ndE = E_DFT - E_MACE   |   {label}   |   n = {n}")
    print(f"  Mittel      = {mu*1000:+.2f} meV   (konstanter Offset, fuer w_i irrelevant)")
    print(f"  Std         = {sd*1000:.2f} meV   -> beta*std = {beta*sd:.3f}")
    print(f"  Schiefe     = {skew:+.3f}   (0 = symmetrisch)")
    print(f"  Exz.-Kurt.  = {kurt:+.3f}   (0 = normal, >0 = schwerere Raender)")
    print(f"  Shapiro-Wilk W = {sw_stat:.4f}, p = {sw_p:.4f} "
    f"-> {'Normalitaet NICHT verworfen' if sw_p > 0.05 else 'Normalitaet verworfen (p<0.05)'}")
    print(f"  Anderson-Darling A2 = {ad_stat:.3f} (krit. 5% = {ad_crit5:.3f}) "
    f"-> {'ok' if ad_stat < ad_crit5 else 'Abweichung'}")

    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(14, 5.6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.25, 1, 1], wspace=0.28)

    # (a) Normal-QQ mit Envelope
    ax = fig.add_subplot(gs[0, 0])
    ax.fill_between(q_theo, lo, hi, color="steelblue", alpha=0.18,
                    label="95 % Monte-Carlo-Envelope")
    lim = (min(q_theo.min(), q_emp.min()) - 0.3, max(q_theo.max(), q_emp.max()) + 0.3)
    ax.plot(lim, lim, "r--", lw=1.4, label="ideal normal")
    ax.plot(q_theo, q_emp, "o", ms=3.5, color="darkorange", alpha=0.85, label="$\\Delta E$ (standardisiert)")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("theoretische Quantile (Normal)")
    ax.set_ylabel("empirische Quantile von $\\Delta E$")
    ax.set_title("(a) Normal-QQ-Plot")
    ax.legend(fontsize=8.5, loc="upper left")
    ax.grid(alpha=0.3)

    # (b) Histogramm + Normal-Fit
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(dE * 1000, bins=35, density=True, alpha=0.6, color="steelblue", label="$\\Delta E$")
    xs = np.linspace((dE * 1000).min(), (dE * 1000).max(), 300)
    ax2.plot(xs, stats.norm.pdf(xs, mu * 1000, sd * 1000), "r-", lw=1.6, label="Normal-Fit")
    ax2.set_xlabel("$\\Delta E$  [meV/Frame]")
    ax2.set_ylabel("Dichte")
    ax2.set_title("(b) Verteilung mit Normal-Fit")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    # (c) QQ gegen Student-t (Tail-Vergleich)
    ax3 = fig.add_subplot(gs[0, 2])
    df_fit, _, _ = stats.t.fit(z, floc=0.0, fscale=1.0)
    q_t = stats.t.ppf(pp, df_fit)
    ax3.plot(q_theo, q_emp, "o", ms=3.2, color="darkorange", alpha=0.8, label="vs. Normal")
    ax3.plot(q_t, q_emp, "s", ms=3.2, color="seagreen", alpha=0.7,
             label=f"vs. Student-t (df={df_fit:.1f})")
    lim3 = (
        min(q_theo.min(), q_t.min(), q_emp.min()) - 0.3,
        max(q_theo.max(), q_t.max(), q_emp.max()) + 0.3,
    )
    ax3.plot(lim3, lim3, "r--", lw=1.3)
    ax3.set_xlim(lim3); ax3.set_ylim(lim3)
    ax3.set_xlabel("theoretische Quantile")
    ax3.set_ylabel("empirische Quantile")
    ax3.set_title("(c) Normal vs. schwerer Rand")
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.grid(alpha=0.3)

    fig.suptitle(
        f"QQ-Analyse der Energiedifferenzen  $\\Delta E = E_{{DFT}} - E_{{MACE}}$   |   {label}   |   n={n}\n"
        f"std = {sd*1000:.2f} meV,  $\\beta\\cdot$std = {beta*sd:.3f},  "
        f"Schiefe = {skew:+.2f},  Exz.-Kurtosis = {kurt:+.2f},  Shapiro-Wilk p = {sw_p:.3f}",
        fontsize=11)
    fig.subplots_adjust(top=0.80, bottom=0.12, left=0.06, right=0.98)

    root = Path(__file__).resolve().parents[1]
    out = root / "results" / f"qq_energy_difference_{args.ensemble}_test{args.testset}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] gespeichert -> {out}")


if __name__ == "__main__":
    main()
