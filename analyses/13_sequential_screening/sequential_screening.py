"""
Sequenzielles Screening — wie wenige DFT-Punkte entscheiden über ein Modell?
============================================================================

DIE FRAGE (Tobi: DFT-Sparen bei L0-Modellen ist interessant). Für ein SCHLECHTES
Modell muss man nicht 400 (oder 5000) DFT-Einzelpunkte rechnen, um zu sehen, dass
das Reweighting nicht trägt. c = beta*std(dE) konvergiert schnell, und sobald c
sicher über c_max = sqrt(-ln R) liegt, ist die Entscheidung "NO-GO" gefallen —
jeder weitere Punkt ist verschwendet.

Dieses Skript SIMULIERT den iterativen Workflow auf den realen Testsatz-Daten
per BOOTSTRAP: jede simulierte Sequenz wird durch Ziehen MIT Zuruecklegen aus den
400 realen dE erzeugt (iid aus der empirischen Verteilung). Nach jedem Punkt wird
neu entschieden. Ueber viele Bootstrap-Sequenzen entsteht ein Band, das zeigt, ab
welchem k die Entscheidung STABIL ist — also wie viele DFT-Punkte man braucht.

WARUM BOOTSTRAP UND NICHT PERMUTATION. Permutation zieht OHNE Zuruecklegen; das
Band kollabiert dann kuenstlich bei k = n (endliche Population, Faktor
sqrt((n-k)/(n-1))). Im Produktionslauf zieht man aber aus einer langen
Trajektorie, also faktisch aus einem unendlichen Pool. Bootstrap (mit
Zuruecklegen) modelliert genau das: die Streuung bei grossem k bleibt erhalten
und entspricht dem echten Stichprobenfehler SE(c)/c = sqrt((gamma2+2)/4k).

Verfolgte Größen, laufend:
    c(k)         = beta * std(dE_1..k)              Entscheidungsgröße
    N_eff/n (k)  aus der Kumulantenentwicklung       Prognose
    N_eff/n (k)  Kish, exakt auf den k Punkten       Wahrheit-auf-Teilmenge
    khat(k)      GPD-Tail                            Existenz-Gate

Entscheidungsregel (bewusst einfach, Gauss-Schranke):
    c(k) + z*SE(c) < c_max   -> PASS gesichert
    c(k) - z*SE(c) > c_max   -> FAIL gesichert
    sonst                     -> weiterrechnen
mit SE(c)/c = sqrt((gamma2+2)/(4k)).

WICHTIG: das ist ein Test des VERFAHRENS, nicht ein DFT-Sparen im
Produktionslauf (dort liegt n=5000 fest). Für das Modell-SCREENING — "lohnt sich
dieses Modell überhaupt?" — ist frühes Abbrechen dagegen echt: man bricht ab,
bevor der ganze Testsatz durchgerechnet ist.

Ausführen:
    python analyses/13_sequential_screening/sequential_screening.py
    python analyses/13_sequential_screening/sequential_screening.py --R 0.7
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from uq_mace.predictions import load_energies
from uq_mace.reweighting import effective_sample_size, psis_khat, running_moments

HERE = Path(__file__).resolve().parent
CACHE = Path(__file__).resolve().parents[2] / "cache"
K_B = 8.617333262e-5

MODELS = {
    "mace-L0-01": "single_mace-L0-01_testbig.npz",
    "mace-L0-c-01": "single_mace-L0-c-01_testbig.npz",
    "mace-L2-c-01": "single_mace-L2-c-01_testbig.npz",
    "ensemble_L2c": "mace_energies_ensemble_L2c_testbig.npz",
}


def running_c(dE: np.ndarray, beta: float) -> np.ndarray:
    """c(k) = beta*std(dE_1..k), ddof=1, vektorisiert."""
    k = np.arange(1, dE.size + 1)
    s1 = np.cumsum(dE)
    s2 = np.cumsum(dE ** 2)
    with np.errstate(invalid="ignore"):
        var = (s2 - s1 ** 2 / k) / (k - 1)
    return beta * np.sqrt(np.maximum(var, 0.0))


def ratio_gauss(c: np.ndarray) -> np.ndarray:
    return np.exp(-c ** 2)


def running_pred(dE: np.ndarray, beta: float) -> np.ndarray:
    """N_eff/n(k), aus c,gamma1,gamma2 der ersten k Punkte (Kumulantenformel)."""
    m = running_moments(dE)
    c = beta * m["std"]
    g1 = np.nan_to_num(m["skew"])
    g2 = np.nan_to_num(m["kurtosis"])
    return np.exp(-c ** 2 + g1 * c ** 3 - (7.0 / 12.0) * g2 * c ** 4)


def khat_band(dE, beta, k_grid, reps, rng):
    """Bootstrap-Verteilung von khat bei jeder Stichprobengroesse k (mit Zuruecklegen)."""
    n = dE.size
    med = np.empty(k_grid.size)
    lo = np.empty(k_grid.size)
    hi = np.empty(k_grid.size)
    for i, k in enumerate(k_grid):
        vals = np.empty(reps)
        for j in range(reps):
            d = dE[rng.integers(0, n, size=k)]
            vals[j] = psis_khat(np.exp(-beta * (d - d.mean())))
        med[i] = np.median(vals)
        lo[i], hi[i] = np.percentile(vals, [5, 95])
    return med, lo, hi


def decide(c, k, gamma2, cmax, z=1.64):
    """PASS / FAIL / WEITER anhand c gegen c_max mit einseitigem z-Band."""
    se = c * np.sqrt((gamma2 + 2.0) / (4.0 * np.maximum(k, 2)))
    out = np.full(c.shape, "WEITER", dtype=object)
    out[c + z * se < cmax] = "PASS"
    out[c - z * se > cmax] = "FAIL"
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--R", type=float, default=0.8)
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--perms", type=int, default=800,
                    help="Zahl der Bootstrap-Sequenzen")
    ap.add_argument("--khat-reps", type=int, default=250,
                    help="Bootstrap-Wiederholungen je k fuer das khat-Band")
    ap.add_argument("--kmax", type=int, default=400,
                    help="Laenge jeder Bootstrap-Sequenz (Pool wird mit Zuruecklegen gezogen)")
    ap.add_argument("--z", type=float, default=1.64, help="einseitiges Konfidenzniveau (1.64=95%)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    beta = 1.0 / (K_B * args.temperature)
    cmax = np.sqrt(-np.log(args.R))
    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    print(f"\nR = {args.R:.2f}  ->  c_max (Gauss) = {cmax:.3f}   "
          f"(T = {args.temperature:.0f} K, {args.perms} Bootstrap-Sequenzen)")
    print("=" * 92)
    print(f"{'Modell':<15}{'c(400)':>8}{'N_eff/n':>9}{'khat':>7}{'Urteil':>8}"
          f"{'k_med':>7}{'k95':>6}{'DFT gespart':>14}{'Fehl':>9}")
    print("-" * 92)

    curves = {}
    rows = []
    for name, fname in MODELS.items():
        p = CACHE / fname
        if not p.exists():
            continue
        e_dft, e_mace = load_energies(p)
        dE = e_dft - e_mace
        n = dE.size
        c_fin = beta * dE.std(ddof=1)
        neff_fin = effective_sample_size(np.exp(-beta * (dE - dE.mean()))) / n
        khat_fin = psis_khat(np.exp(-beta * (dE - dE.mean())))
        g2_fin = float(((dE - dE.mean()) ** 4).mean() / dE.std() ** 4 - 3.0)
        final = "PASS" if c_fin < cmax else "FAIL"

        kmax = args.kmax
        ks = np.arange(1, kmax + 1)
        c_stack = np.empty((args.perms, kmax))
        pred_stack = np.empty((args.perms, kmax))
        stop_k = np.empty(args.perms, dtype=int)
        stop_wrong = np.zeros(args.perms, dtype=bool)
        wrong_verdict = "PASS" if final == "FAIL" else "FAIL"
        for j in range(args.perms):
            d = dE[rng.integers(0, n, size=kmax)]     # Bootstrap: mit Zuruecklegen
            c = running_c(d, beta)
            c_stack[j] = c
            pred_stack[j] = running_pred(d, beta)
            dec = decide(c, ks, g2_fin, cmax, args.z)
            hit = np.where((dec == final) & (ks >= 5))[0]
            wrong = np.where((dec == wrong_verdict) & (ks >= 5))[0]
            k_correct = ks[hit[0]] if hit.size else kmax
            k_wrong = ks[wrong[0]] if wrong.size else kmax + 1
            stop_k[j] = min(k_correct, k_wrong)
            stop_wrong[j] = k_wrong < k_correct       # falsches Urteil zuerst

        k_med = int(np.median(stop_k))
        k95 = int(np.percentile(stop_k, 95))
        saved = 100 * (1 - stop_k.mean() / n)
        misfire = 100 * stop_wrong.mean()

        # --- c sagt N_eff voraus? Prognose (Kumulanten) vs. Wahrheit (Kish) ---
        pred_fin = float(np.exp(-c_fin ** 2
                                + float(((dE - dE.mean()) ** 3).mean() / dE.std() ** 3) * c_fin ** 3
                                - (7.0 / 12.0) * g2_fin * c_fin ** 4))
        pred_dev = 100 * (pred_fin / neff_fin - 1)

        # --- khat laufend (eigene Bootstrap-Baender auf einem Gitter) ---
        k_grid = np.unique(np.clip(
            np.r_[np.arange(15, 60, 5), np.arange(60, kmax + 1, 30)], 15, kmax))
        kh_med, kh_lo, kh_hi = khat_band(dE, beta, k_grid, args.khat_reps, rng)
        curves[name] = dict(ks=ks, c=c_stack, pred=pred_stack, final=final, c_fin=c_fin,
                            neff_fin=neff_fin, stop_k=stop_k, pred_fin=pred_fin,
                            k_grid=k_grid, kh_med=kh_med, kh_lo=kh_lo, kh_hi=kh_hi,
                            khat_fin=khat_fin)
        rows.append(dict(name=name, c_fin=c_fin, neff_fin=neff_fin, khat=khat_fin,
                         final=final, k_med=k_med, k95=k95, saved=saved,
                         misfire=misfire, pred_fin=pred_fin, pred_dev=pred_dev))
        print(f"{name:<15}{c_fin:>8.3f}{neff_fin:>9.3f}{khat_fin:>7.3f}{final:>8}"
              f"{k_med:>7d}{k95:>6d}{saved:>12.0f} %{misfire:>8.1f} %")
    print("=" * 92)
    print("  k_med / k95: DFT-Punkte bis zur GESICHERTEN Entscheidung (Median / 95%).")
    print("  'DFT gespart': Anteil der 400, den man im Mittel NICHT rechnen muss.")
    print("  'Fehl': Anteil der Sequenzen, in denen zuerst das FALSCHE Urteil feuert")
    print(f"          (Zielwert unter dem einseitigen Niveau 5% bei z={args.z}).")

    # ---- Validierung: sagt c das endgueltige N_eff voraus? ----
    print("\nSagt c das N_eff am Ende voraus?  Prognose (Kumulanten aus c,g1,g2) vs. Kish (exakt)")
    print("-" * 72)
    print(f"{'Modell':<15}{'c(400)':>8}{'N_eff/n Prognose':>18}{'N_eff/n Kish':>15}{'Abw.':>9}")
    for r in rows:
        print(f"{r['name']:<15}{r['c_fin']:>8.3f}{r['pred_fin']:>18.4f}"
              f"{r['neff_fin']:>15.4f}{r['pred_dev']:>+8.2f} %")
    print("  Kleine Abweichung = die c-basierte Formel trifft das exakte Kish-N_eff.")

    # ---- CSV ----
    f1 = args.outdir / f"sequential_screening_R{args.R:.2f}.csv"
    with open(f1, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["model", "c_final", "neff_ratio_kish", "neff_ratio_pred_c",
                     "pred_dev_pct", "khat", "verdict", "k_median", "k95",
                     "pct_dft_saved", "misfire_pct"])
        for r in rows:
            wr.writerow([r["name"], f"{r['c_fin']:.4f}", f"{r['neff_fin']:.4f}",
                         f"{r['pred_fin']:.4f}", f"{r['pred_dev']:.2f}",
                         f"{r['khat']:.4f}", r["final"], r["k_med"], r["k95"],
                         f"{r['saved']:.1f}", f"{r['misfire']:.2f}"])

    cols = {"mace-L0-01": "firebrick", "mace-L0-c-01": "darkorange",
            "mace-L2-c-01": "steelblue", "ensemble_L2c": "seagreen"}
    XL = min(120, args.kmax)

    # ================= FIGUR 1 — der ganze Workflow, 2x2 =================
    fig, ((a1, a2), (a3, a4)) = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Entscheidungs-Gate: laufendes c gegen c_max
    for name, cv in curves.items():
        col = cols.get(name, "gray")
        med = np.median(cv["c"], axis=0)
        lo, hi = np.percentile(cv["c"], [5, 95], axis=0)
        a1.plot(cv["ks"], med, color=col, lw=1.8, label=f"{name} ({cv['final']})")
        a1.fill_between(cv["ks"], lo, hi, color=col, alpha=0.15)
    a1.axhline(cmax, color="k", ls="--", lw=1.5, label=f"$c_{{max}}$={cmax:.3f} (R={args.R})")
    a1.set_xlabel("gerechnete DFT-Punkte $k$")
    a1.set_ylabel("$c(k)=\\beta\\,\\mathrm{std}(\\Delta E)$")
    a1.set_title("① Entscheidungs-Gate — laufendes $c$ gegen $c_{max}$")
    a1.set_xlim(0, XL); a1.legend(fontsize=8); a1.grid(alpha=0.3)

    # (b) Existenz-Gate: laufendes khat gegen 0.5 / 0.25
    for name, cv in curves.items():
        col = cols.get(name, "gray")
        a2.plot(cv["k_grid"], cv["kh_med"], "o-", color=col, ms=3, lw=1.6, label=name)
        a2.fill_between(cv["k_grid"], cv["kh_lo"], cv["kh_hi"], color=col, alpha=0.13)
    a2.axhline(0.5, color="crimson", ls="--", lw=1.4, label="0.5 — $N_{eff}$ existiert")
    a2.axhline(0.25, color="darkgoldenrod", ls=":", lw=1.4, label="0.25 — Fehlerbalken")
    a2.set_xlabel("gerechnete DFT-Punkte $k$")
    a2.set_ylabel("$\\hat k$ (PSIS-Tail)")
    a2.set_title("② Existenz-Gate — laufendes $\\hat k$ mit 5–95 %-Band")
    a2.set_xlim(0, XL); a2.legend(fontsize=8); a2.grid(alpha=0.3)

    # (c) Prognose-Validierung: c-basierte N_eff-Prognose gegen Kish-Wahrheit
    # Formel nur ab k0 sinnvoll (bei sehr kleinem k sprengt der c^4-Term alles).
    k0 = 10
    for name, cv in curves.items():
        col = cols.get(name, "gray")
        pr = np.clip(cv["pred"][:, k0 - 1:], 0.0, 1.05)
        xs = cv["ks"][k0 - 1:]
        med = np.median(pr, axis=0)
        lo, hi = np.percentile(pr, [5, 95], axis=0)
        a3.plot(xs, med, color=col, lw=1.8, label=f"{name}")
        a3.fill_between(xs, lo, hi, color=col, alpha=0.15)
        a3.axhline(cv["neff_fin"], color=col, ls=":", lw=1.3)
    a3.axhline(args.R, color="k", ls="--", lw=1.3, label=f"R={args.R}")
    a3.set_xlabel("gerechnete DFT-Punkte $k$")
    a3.set_ylabel("$N_{eff}/n$")
    a3.set_title("③ Prognose aus $c$ (Linie) → Kish-Wahrheit (gepunktet)")
    a3.set_xlim(0, XL); a3.set_ylim(0, 1.02)
    a3.legend(fontsize=8); a3.grid(alpha=0.3)

    # (d) wie frueh steht das Urteil?
    for name, cv in curves.items():
        col = cols.get(name, "gray")
        xs = np.sort(cv["stop_k"]); ys = np.arange(1, xs.size + 1) / xs.size
        a4.plot(xs, ys, color=col, lw=2, label=name)
    a4.axhline(0.95, color="gray", ls=":", lw=1)
    a4.set_xlabel("DFT-Punkte bis zur gesicherten Entscheidung")
    a4.set_ylabel("Anteil der Läufe entschieden")
    a4.set_title("④ wie früh steht das Urteil fest?")
    a4.set_xlim(0, XL); a4.legend(fontsize=8); a4.grid(alpha=0.3)

    fig.suptitle(
        f"Sequenzieller Screening-Workflow — DFT-Punkte $k$ akkumulieren, "
        f"nach jedem neu entscheiden   |   R={args.R}, T={args.temperature:.0f} K",
        fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out1 = args.outdir / f"sequential_workflow_R{args.R:.2f}.png"
    fig.savefig(out1, dpi=140)

    # ================= FIGUR 2 — khat im Detail =================
    fig2, (b1, b2) = plt.subplots(1, 2, figsize=(14, 5.4))

    # (a) Konvergenz von khat mit k, plus n^-1/4-Schrumpfung des Bandes
    for name, cv in curves.items():
        col = cols.get(name, "gray")
        b1.plot(cv["k_grid"], cv["kh_med"], "o-", color=col, ms=4, lw=1.7,
                label=f"{name}  ($\\hat k_{{400}}$={cv['khat_fin']:+.2f})")
        b1.fill_between(cv["k_grid"], cv["kh_lo"], cv["kh_hi"], color=col, alpha=0.13)
    b1.axhline(0.5, color="crimson", ls="--", lw=1.4, label="0.5 — Existenz-Gate")
    b1.axhline(0.25, color="darkgoldenrod", ls=":", lw=1.4, label="0.25 — 4. Moment")
    b1.axhline(0.0, color="k", lw=0.6, alpha=0.4)
    b1.set_xlabel("Stichprobengröße $k$")
    b1.set_ylabel("$\\hat k$")
    b1.set_title("(a) $\\hat k$-Konvergenz — Band schrumpft wie $k^{-1/4}$")
    b1.legend(fontsize=8.5); b1.grid(alpha=0.3)

    # (b) Threshold-Stabilitaet bei k=400: khat gegen die Tail-Groesse M
    fracs = np.linspace(0.05, 0.35, 13)
    for name, fname in MODELS.items():
        if name not in curves:
            continue
        col = cols.get(name, "gray")
        e_dft, e_mace = load_energies(CACHE / fname)
        w = np.exp(-beta * ((e_dft - e_mace) - (e_dft - e_mace).mean()))
        sw = np.sort(w)
        kk = []
        for fr in fracs:
            M = max(int(fr * w.size), 5)
            from scipy.stats import genpareto
            u = sw[-M - 1]
            k_, _, _ = genpareto.fit(sw[-M:] - u, floc=0)
            kk.append(k_)
        b2.plot(100 * fracs, kk, "o-", color=col, ms=4, lw=1.6, label=name)
    b2.axhline(0.5, color="crimson", ls="--", lw=1.4)
    b2.axhline(0.25, color="darkgoldenrod", ls=":", lw=1.4)
    b2.set_xlabel("Tail-Anteil $M/n$ [%] (Schwellenwahl)")
    b2.set_ylabel("$\\hat k$ (roher GPD-Fit)")
    b2.set_title("(b) Threshold-Stabilität bei $k$=400 — Plateau = GPD trägt")
    b2.legend(fontsize=8.5); b2.grid(alpha=0.3)

    fig2.suptitle(
        f"$\\hat k$-Untersuchung — das Existenz-Gate des Verfahrens   |   "
        f"T={args.temperature:.0f} K", fontsize=12)
    fig2.tight_layout(rect=(0, 0, 1, 0.93))
    out2 = args.outdir / f"sequential_khat_R{args.R:.2f}.png"
    fig2.savefig(out2, dpi=140)

    print(f"\n[plot ] {out1}")
    print(f"[plot ] {out2}")
    print(f"[csv  ] {f1}")


if __name__ == "__main__":
    main()
