"""
Unsicherheit von k_hat: Threshold-Stabilitaet, Bootstrap, Gate-Sicherheit
============================================================================

k_hat ist nicht irgendeine Nebendiagnostik, sondern die EXISTENZBEDINGUNG des
gesamten c-Formalismus:

    E[w^2] = E[exp(-2c z)] < unendlich   <=>   k_hat < 0.5
    E[w^4] < unendlich                   <=>   k_hat < 0.25   (Fehlerbalken)

Existiert E[w^2] nicht, ist K(-2c) undefiniert und die gesamte
Kumulantenherleitung (c_max, Gauss-Kriterium, ...) bedeutungslos - nicht ungenau,
sondern sinnlos. Deshalb muss die Unsicherheit von k_hat VOR allem anderen
geklaert werden.

Dieses Skript liefert drei Dinge, die die laufende k_hat-Kurve
(khat_running_estimate.py) NICHT liefert - dort sieht man nur die Konvergenz:

  (a) THRESHOLD-STABILITAET - haengt k_hat von der willkuerlichen Wahl der
      Tail-Groesse M ab? Ein flaches Plateau heisst: die GPD-Naeherung greift.
      Ein systematischer Drift heisst: k_hat misst die Schwellenwahl, nicht den
      Rand. Das ist die MODELLunsicherheit und meist die kritischere.
  (b) PARAMETRISCHER BOOTSTRAP - Konfidenzintervall des Endwerts. Bewusst
      parametrisch: der nichtparametrische Bootstrap ist fuer Tail-Groessen
      unzuverlaessig, weil Ziehen mit Zuruecklegen keine NEUEN Extremwerte
      erzeugt und die Randvariabilitaet dadurch unterschaetzt.
  (c) GATE-SICHERHEIT - nicht "wie gross ist k_hat", sondern "wie sicher liegt
      es unter 0.5 bzw. 0.25", gemessen in Standardfehlern.

Ausfuehren:
    python analyses/khat_uncertainty.py
    python analyses/khat_uncertainty.py --weights cache/mace_energies_ensemble_L2c_testbig.npz
    python analyses/khat_uncertainty.py --boot 4000 --m-points 30
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import genpareto

from uq_mace.predictions import load_weights
from uq_mace.reweighting import _gpd_fit_khat, khat_threshold, psis_khat

HERE = Path(__file__).resolve().parent   # Ausgaben landen neben dem Skript
PRIOR_K = 10.0  # wie in _gpd_fit_khat


# ---------------------------------------------------------------------------
def tail_exceedances(w_sorted: np.ndarray, m: int) -> np.ndarray:
    """Die m groessten Gewichte, um die Schwelle reduziert."""
    cut = w_sorted[-m - 1]
    ex = w_sorted[-m:] - cut
    return ex[ex > 0]


def khat_at_m(w_sorted: np.ndarray, m: int) -> float:
    ex = tail_exceedances(w_sorted, m)
    return _gpd_fit_khat(ex) if ex.size >= 5 else np.nan


def standard_m(n: int) -> int:
    """Tail-Groesse nach der Regel in psis_khat: min(0.2 S, 3 sqrt(S))."""
    return int(min(0.2 * n, 3.0 * np.sqrt(n)))


def raw_khat(reported: float, m: int) -> float:
    """Prior-freien Rohwert zurueckrechnen.

    _gpd_fit_khat gibt (m*k_roh + prior_k*0.5)/(m + prior_k) zurueck.
    Bei kleinem m dominiert der Prior - gut zu wissen, bevor man 0.065 als
    'leicht schwerer Rand' interpretiert.
    """
    return (reported * (m + PRIOR_K) - PRIOR_K * 0.5) / m


def parametric_bootstrap(ex: np.ndarray, reps: int, rng) -> np.ndarray:
    """GPD an den Tail fitten, daraus neue Tails simulieren, jeweils neu fitten.

    Erzeugt echte neue Extremwerte - im Gegensatz zum nichtparametrischen
    Bootstrap, der nur die beobachteten recycelt.
    """
    c_fit, _, s_fit = genpareto.fit(ex, floc=0.0)
    out = np.empty(reps)
    for b in range(reps):
        sim = genpareto.rvs(c_fit, loc=0.0, scale=s_fit, size=ex.size, random_state=rng)
        out[b] = _gpd_fit_khat(sim)
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="cache/mace_energies_ensemble_L2c_testbig.npz",
                    help="fertige w_i ODER Energie-Cache mit e_dft/e_mace")
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--boot", type=int, default=2000, help="Bootstrap-Wiederholungen (Endwert)")
    ap.add_argument("--boot-m", type=int, default=400, help="Bootstrap-Wdh. je M (Band)")
    ap.add_argument("--m-points", type=int, default=22, help="Stuetzstellen der M-Achse")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default=None, help="Ausgabeordner (Default: neben dem Skript)")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[2]
    path = args.weights if Path(args.weights).is_absolute() else root / args.weights
    w = load_weights(path, args.temperature)
    n = w.size
    ws = np.sort(w)
    rng = np.random.default_rng(args.seed)

    m_std = standard_m(n)
    khat = psis_khat(w)
    thr_v = khat_threshold(n)

    # ---- (a) Threshold-Stabilitaet mit Bootstrap-Band --------------------
    m_max = min(int(0.5 * n), n - 2)
    Ms = np.unique(np.linspace(20, m_max, args.m_points).astype(int))
    k_of_m, lo_of_m, hi_of_m = [], [], []
    print(f"[calc ] Threshold-Stabilitaet ueber {Ms.size} Tail-Groessen ...")
    for m in Ms:
        k_of_m.append(khat_at_m(ws, int(m)))
        ex = tail_exceedances(ws, int(m))
        if ex.size >= 8:
            b = parametric_bootstrap(ex, args.boot_m, rng)
            lo_of_m.append(np.nanpercentile(b, 10))
            hi_of_m.append(np.nanpercentile(b, 90))
        else:
            lo_of_m.append(np.nan); hi_of_m.append(np.nan)
    k_of_m = np.array(k_of_m); lo_of_m = np.array(lo_of_m); hi_of_m = np.array(hi_of_m)

    # ---- (b) Bootstrap am Standard-M ------------------------------------
    print(f"[calc ] parametrischer Bootstrap bei M = {m_std} ({args.boot} Wdh.) ...")
    ex_std = tail_exceedances(ws, m_std)
    boot = parametric_bootstrap(ex_std, args.boot, rng)
    se = float(np.nanstd(boot))
    ci_lo, ci_hi = np.nanpercentile(boot, [2.5, 97.5])

    # ---- Ausgabe ---------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"n = {n} Gewichte,  Standard-Tailgroesse M = {m_std}")
    print("=" * 74)
    print(f"  k_hat (psis_khat)        = {khat:.4f}")
    print(f"  davon Rohwert ohne Prior = {raw_khat(khat, ex_std.size):.4f}   "
          f"(Prior zieht gegen 0.5)")
    print(f"  Bootstrap 95%-KI         = [{ci_lo:.4f}, {ci_hi:.4f}]   SE = {se:.4f}")
    print(f"  Faustformel (1+k)/sqrt(M)= {(1+khat)/np.sqrt(m_std):.4f}")
    print(f"\n  Threshold-Stabilitaet: k_hat von {np.nanmin(k_of_m):+.3f} bis "
          f"{np.nanmax(k_of_m):+.3f} ueber M = {Ms[0]}..{Ms[-1]}")
    drift = np.nanmax(k_of_m) - np.nanmin(k_of_m)
    print(f"    Spannweite {drift:.3f} vs. typische Bootstrap-Breite "
          f"{np.nanmedian(hi_of_m - lo_of_m):.3f}  -> "
          f"{'im Rauschen, Plateau' if drift < 1.5*np.nanmedian(hi_of_m-lo_of_m) else 'SYSTEMATISCHER DRIFT'}")

    print("\n  GATE-SICHERHEIT (Abstand in Standardfehlern):")
    gates = [(0.5, "Existenz E[w^2] - Fundament"),
             (0.25, "Fehlerbalken E[w^4]"),
             (thr_v, "Vehtari-Schwelle")]
    for t, lbl in gates:
        d = (t - khat) / se
        print(f"    k_hat < {t:.3f}  ({lbl:<28}): {d:>5.1f} SE  "
              f"-> {'sicher' if d > 2 else 'NICHT gesichert'}")

    # ---- Plot ------------------------------------------------------------
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.2, 1], hspace=0.34, wspace=0.22)

    # (a) Threshold-Stabilitaet
    ax = fig.add_subplot(gs[0, :])
    ax.fill_between(Ms, lo_of_m, hi_of_m, color="steelblue", alpha=0.18,
                    label="10-90 % parametrischer Bootstrap")
    ax.plot(Ms, k_of_m, "o-", color="steelblue", lw=1.8, ms=4, label="$\\hat{k}(M)$")
    ax.axvline(m_std, color="purple", ls="-.", lw=1.4,
               label=f"Standardwahl M = {m_std}")
    ax.axhline(0.5, color="crimson", ls="--", lw=1.3, label="0.5 — Existenz von E[w²]")
    ax.axhline(0.25, color="darkorange", ls=":", lw=1.3, label="0.25 — Fehlerbalken")
    ax.axhline(0.0, color="k", lw=0.8, alpha=0.5)
    ax.set_xlabel("M — Zahl der Tail-Punkte im GPD-Fit")
    ax.set_ylabel("$\\hat{k}$")
    ax.set_title("(a) Threshold-Stabilität — misst $\\hat{k}$ den Rand oder die Schwellenwahl?",
                 pad=38)
    ax.legend(fontsize=8.5, ncol=2, loc="upper right")
    ax.grid(alpha=0.3)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x / n, lambda x: x * n))
    sec.set_xlabel("M / n", fontsize=9, labelpad=2)

    # (b) Bootstrap-Verteilung
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(boot, bins=45, density=True, color="steelblue", alpha=0.65)
    ax2.axvline(khat, color="purple", lw=2, label=f"$\\hat{{k}}$ = {khat:.3f}")
    ax2.axvspan(ci_lo, ci_hi, color="purple", alpha=0.12, label=f"95 % KI")
    ax2.axvline(0.5, color="crimson", ls="--", lw=1.3)
    ax2.axvline(0.25, color="darkorange", ls=":", lw=1.3)
    ax2.set_xlabel("$\\hat{k}$ (Bootstrap)")
    ax2.set_ylabel("Dichte")
    ax2.set_title(f"(b) Bootstrap-Verteilung bei M = {m_std}\nSE = {se:.3f}")
    ax2.legend(fontsize=8.5)
    ax2.grid(alpha=0.3)

    # (c) Gate-Sicherheit
    ax3 = fig.add_subplot(gs[1, 1])
    ypos = np.arange(len(gates))[::-1]
    for y, (t, lbl) in zip(ypos, gates):
        d = (t - khat) / se
        ax3.barh(y, t, height=0.55,
                 color=("#d4edda" if d > 2 else "#f8d7da"), edgecolor="gray")
        ax3.text(t + 0.01, y, f"{d:.1f} SE", va="center", fontsize=9,
                 color=("seagreen" if d > 2 else "firebrick"), fontweight="bold")
        ax3.text(0.01, y + 0.33, lbl, fontsize=8, color="#333")
    ax3.errorbar([khat], [len(gates)], xerr=[[khat - ci_lo], [ci_hi - khat]],
                 fmt="o", color="purple", capsize=5, ms=8, label="$\\hat{k}$ ± 95 % KI")
    ax3.set_yticks(list(ypos) + [len(gates)])
    ax3.set_yticklabels([f"< {t:.2f}" for t, _ in gates] + ["Messung"], fontsize=9)
    ax3.set_xlabel("$\\hat{k}$")
    ax3.set_title("(c) Wie sicher liegen die Gates?")
    ax3.legend(fontsize=8.5, loc="upper left")
    ax3.grid(alpha=0.3, axis="x")
    ax3.set_xlim(min(-0.3, ci_lo - 0.05), 0.80)
    ax3.set_ylim(-0.6, len(gates) + 0.7)

    fig.suptitle(
        f"Unsicherheit von $\\hat{{k}}$ — Existenzbedingung des c-Formalismus\n"
        f"{Path(args.weights).name},  n = {n},  T = {args.temperature:.0f} K   |   "
        f"$\\hat{{k}}$ = {khat:.3f}  95 % KI [{ci_lo:.3f}, {ci_hi:.3f}]",
        fontsize=12, y=0.98)
    fig.subplots_adjust(top=0.845, left=0.07, right=0.97, bottom=0.07)

    out = (Path(args.outdir) if args.outdir else HERE) / "khat_uncertainty.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] gespeichert -> {out}")


if __name__ == "__main__":
    main()
