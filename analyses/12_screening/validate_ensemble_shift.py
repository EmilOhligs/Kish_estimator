"""
Beweis, dass die Ensemble-Korrektur funktioniert — analytisch bekannter Testfall
============================================================================

DAS PROBLEM MIT ECHTEN DATEN. Auf water_testset_big laesst sich die A1-Korrektur
nicht validieren: dazu braeuchte man DFT-Referenzen auf Frames der MACE-MD, und
genau die existieren nicht (sonst braeuchte man das Screening nicht). Jede
Pruefung an echten Daten vergleicht Schaetzer gegen Schaetzer.

DIE LOESUNG. Man konstruiert einen Fall, in dem BEIDE Ensembles analytisch
bekannt sind, und prueft, ob die Umgewichtung aus p_DFT-Stichproben den wahren
p_MACE-Wert trifft.

Die Gamma-Familie ist unter exponentiellem Tilten geschlossen. Ist dE unter
p_MACE gammaverteilt, X ~ Gamma(k, theta), dann folgt aus

    p_D(d) ∝ p_M(d) exp(-beta d)      (die Reweighting-Identitaet)

sofort  X ~ Gamma(k, theta/(1+beta theta))  unter p_DFT. Beide Ensembles sind
damit geschlossen bekannt:

    c_M     = beta theta sqrt(k)                   c_D = c_M/(1+beta theta)
    gamma_1 = 2/sqrt(k)                            (in BEIDEN, skaleninvariant)
    N_eff/n |_M = (1+2 beta theta)^k / (1+beta theta)^(2k)

WAS GEPRUEFT WIRD. Aus n Stichproben von p_DFT werden geschaetzt:

    c_MACE     ueber die exakte Rueckwaerts-Umgewichtung mit 1/w
    N_eff/n    ueber  n^2 / (sum w * sum 1/w)          [direkt, annahmefrei]
    c 1.Ordn.  ueber  c_D (1 + gamma_1 c_D / 2)        [die Naeherungsformel]

und gegen die analytische Wahrheit gehalten. Zusaetzlich laeuft der naive Wert
mit (Testsatz-N_eff/n unkorrigiert als Prognose) — er ist der Strohmann, den die
Korrektur schlagen muss.

WICHTIG: die Gamma-Annahme dient NUR dazu, eine bekannte Wahrheit zu haben. Der
geprueft Schaetzer selbst ist verteilungsfrei — er sieht nur die Stichprobe.

PARAMETERWAHL. Default k=16, beta*theta=0.089 reproduziert die realen L2c-Werte
(gamma_1 = 0.50, c_test = 0.327, N_eff/n = 0.9125). Das ist kein Zufall bei
gamma_1 und c (danach wurde kalibriert), wohl aber bei N_eff/n.

TEIL 3 — DIE GRENZE DER METHODE (Abdeckung).

Alles oben setzt voraus, dass p_test dort Masse hat, wo p_MACE Masse hat. Ist das
verletzt, versagt das Verfahren — und zwar UNBEMERKT. Modelliert wird der
klassische MLIP-Pathologiefall: ein LOCH in der Potentialflaeche, also eine
Struktur, die MACE fuer stark guenstig haelt und DFT nicht. Die MD faellt hinein.

    p_MACE = (1-f) * gutes Gebiet  +  f * Loch bei dE = +D

Unter p_DFT ist das Loch um exp(-beta D) unterdrueckt, taucht im Testsatz also
kaum auf. Noetiger Testsatzumfang, um es EINMAL zu treffen:

    n >= exp(beta D) / f          EXPONENTIELL in der Lochtiefe

Zwei Befunde, beide unangenehm:
  * Das Warnlicht N_eff(1/w) ist NICHT MONOTON. Es schlaegt bei mittlerer
    Lochtiefe an (einzelne Loch-Frames mit riesigem 1/w dominieren die Summe),
    geht bei tiefen Loechern aber wieder auf "alles gut" — was gar nicht in der
    Stichprobe ist, hinterlaesst keine Spur.
  * Der noetige Umfang uebersteigt schnell den Produktionslauf selbst. Bei
    D = 8 k_BT waeren ~10000 DFT-Rechnungen noetig, um 5000 abzusichern. Das
    Screening kostete dann mehr als die Sache, die es screent.

SCHLUSSFOLGERUNG: die Abdeckungsfrage ist strukturell KEINE Testsatz-Frage. Sie
gehoert an die Trajektorie (Komitee-sigma(R) je Frame, oder Deskriptorabstand),
wo sie ein billiges DETEKTIONS-Problem ist statt eines teuren Schaetzproblems.
Die Kernaussage des Screenings ist damit konditional:

    "Unter der Bedingung, dass die MD im abgedeckten Bereich bleibt, sind die
     Gewichte statistisch brauchbar."

Ausfuehren:
    python analyses/12_screening/validate_ensemble_shift.py
    python analyses/12_screening/validate_ensemble_shift.py --k 9 --repeats 1000
    python analyses/12_screening/validate_ensemble_shift.py --skip-coverage
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


# --- analytische Wahrheit -------------------------------------------------
def truth(k: int, u: float) -> dict:
    """u = beta*theta. beta = 1 o.B.d.A. — nur das Produkt geht ein."""
    uD = u / (1.0 + u)                      # beta*theta im DFT-Ensemble
    return dict(
        u=u, k=k,
        gamma1=2.0 / np.sqrt(k),
        c_D=uD * np.sqrt(k),                # was man auf dem Testsatz misst
        c_M=u * np.sqrt(k),                 # was im Produktionslauf gilt
        ratio_M=(1 + 2 * u) ** k / (1 + u) ** (2 * k),          # ZIEL
        ratio_D=(1 + 2 * uD) ** k / (1 + uD) ** (2 * k),        # naiv
        scale_D=uD,
    )


def _lse(v: np.ndarray) -> float:
    m = v.max()
    return float(m + np.log(np.exp(v - m).sum()))


def estimate(sample: np.ndarray) -> dict:
    """Alles aus einer p_DFT-Stichprobe. beta = 1 (in der Skala absorbiert)."""
    d = sample
    n = d.size
    log_w = -d                                    # w = exp(-beta dE)

    # exakte Rueckwaerts-Umgewichtung p_DFT -> p_MACE mit 1/w
    r = np.exp(d - d.max())                       # ∝ 1/w, stabilisiert
    mean_m = (r * d).sum() / r.sum()
    var_m = (r * (d - mean_m) ** 2).sum() / r.sum()

    # direkter Schaetzer: n^2 / (sum w * sum 1/w)
    ratio = float(np.exp(2 * np.log(n) - _lse(log_w) - _lse(-log_w)))

    # Naeherung 1. Ordnung aus den p_DFT-Momenten
    s = d.std(ddof=1)
    g1 = float(((d - d.mean()) ** 3).mean() / s ** 3)
    c_D = s

    # Selbstdiagnose: wie gut ist die Rueckwaerts-Umgewichtung selbst?
    neff_back = float(r.sum() ** 2 / (r ** 2).sum())

    return dict(c_exact=float(np.sqrt(var_m)),
                c_first=float(c_D * (1 + 0.5 * g1 * c_D)),
                ratio=ratio, neff_back=neff_back,
                ratio_naive=float(np.exp(2 * np.log(n)
                                         - 2 * _lse(log_w) + _lse(2 * log_w))))


def run(k, u, n, repeats, rng) -> dict:
    t = truth(k, u)
    out = {key: [] for key in ("c_exact", "c_first", "ratio", "neff_back")}
    for _ in range(repeats):
        d = rng.gamma(k, t["scale_D"], n)
        e = estimate(d)
        for key in out:
            out[key].append(e[key])
    res = {key: (float(np.mean(v)), float(np.std(v))) for key, v in out.items()}
    res["truth"] = t
    res["n"] = n
    return res


def run_hole(D: float, f: float, spread: float, n: int, repeats: int, rng) -> dict:
    """Loch-Modell: p_MACE = (1-f)*N(0,spread) + f*N(D,spread), beta = 1.

    Die Wahrheit wird per Grosstichprobe aus p_MACE bestimmt, die Schaetzung aus
    n Frames, die per Wichtungs-Resampling aus p_MACE nach p_DFT gezogen werden
    (p_DFT ∝ p_MACE * exp(-dE)).
    """
    def sample_M(m):
        hole = rng.random(m) < f
        return rng.normal(np.where(hole, D, 0.0), spread)

    big = sample_M(2_000_000)
    lw = -big
    truth = float(np.exp(2 * np.log(big.size) - _lse(lw) - _lse(-lw)))

    est, back, seen = [], [], []
    for _ in range(repeats):
        pool = sample_M(200_000)
        pw = np.exp(-pool - (-pool).max())          # ∝ w, fuer das Resampling
        x = pool[rng.choice(pool.size, n, p=pw / pw.sum())]
        lwx = -x
        est.append(np.exp(2 * np.log(n) - _lse(lwx) - _lse(-lwx)))
        r = np.exp(x - x.max())
        back.append(r.sum() ** 2 / (r ** 2).sum())
        seen.append(int((x > D / 2).sum()))

    return dict(D=D, f=f, n=n, truth=truth,
                est=float(np.mean(est)), est_sd=float(np.std(est)),
                neff_back=float(np.mean(back)), hole_frames=float(np.mean(seen)),
                n_required=float(np.exp(D) / f))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=16, help="Gamma-Form; gamma_1 = 2/sqrt(k)")
    ap.add_argument("--u", type=float, default=0.089, help="beta*theta (kalibriert auf L2c)")
    ap.add_argument("--repeats", type=int, default=400)
    ap.add_argument("--hole-fraction", type=float, default=0.30,
                    help="Zeitanteil f, den die MD im Loch verbringt")
    ap.add_argument("--skip-coverage", action="store_true",
                    help="Teil 3 ueberspringen (teuer: Resampling aus grossen Pools)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- Teil 1: Konsistenz — konvergiert der Schaetzer gegen die Wahrheit? ----
    t = truth(args.k, args.u)
    print(f"\nANALYTISCHE WAHRHEIT   (k = {args.k}, beta*theta = {args.u})")
    print("=" * 78)
    print(f"  gamma_1                                        {t['gamma1']:.4f}")
    print(f"  c unter p_DFT   (Messung auf dem Testsatz)     {t['c_D']:.4f}")
    print(f"  c unter p_MACE  (gilt im Produktionslauf)      {t['c_M']:.4f}")
    print(f"  N_eff/n unter p_DFT   (naive Prognose)         {t['ratio_D']:.4f}")
    print(f"  N_eff/n unter p_MACE  <- ZIEL                  {t['ratio_M']:.4f}")
    print(f"\n  Formel 1. Ordnung  c_D(1+g1 c_D/2) = {t['c_D']*(1+0.5*t['gamma1']*t['c_D']):.4f}"
          f"   ({100*(t['c_D']*(1+0.5*t['gamma1']*t['c_D'])/t['c_M']-1):+.2f} % gegen exakt)")
    print(f"\n  Der naive Wert ist um {100*(t['ratio_D']/t['ratio_M']-1):+.2f} % falsch —")
    print( "  SYSTEMATISCH: der Fehler bleibt bei n -> unendlich bestehen.")

    ns = [100, 400, 2000, 10000, 50000]
    rows_n = [run(args.k, args.u, n, args.repeats, rng) for n in ns]

    print(f"\nSCHAETZUNG AUS p_DFT-STICHPROBEN   ({args.repeats} Wiederholungen)")
    print("=" * 78)
    print(f"{'n':>8}{'c_MACE':>20}{'N_eff/n':>22}{'Bias':>10}")
    print("-" * 78)
    for r in rows_n:
        cm, cs = r["c_exact"]
        rm, rs = r["ratio"]
        print(f"{r['n']:>8}   {cm:.4f} +- {cs:.4f}    {rm:.4f} +- {rs:.4f}"
              f"   {100*(rm/t['ratio_M']-1):>+7.2f} %")
    print("=" * 78)
    print("  Bias faellt gegen 0, Streuung wie 1/sqrt(n) -> der Schaetzer ist KONSISTENT.")

    # ---- Teil 2: Wo bricht er? ----
    us = [0.03, 0.089, 0.2, 0.35, 0.5, 0.8]
    rows_u = [run(args.k, u, 400, args.repeats, rng) for u in us]

    print(f"\nGUELTIGKEITSGRENZE   (n = 400, k = {args.k})")
    print("=" * 78)
    print(f"{'c_test':>8}{'wahr':>10}{'naiv':>10}{'geschaetzt':>22}{'Bias':>9}{'N_eff(1/w)':>12}")
    print("-" * 78)
    for r in rows_u:
        tt = r["truth"]
        rm, rs = r["ratio"]
        nb, _ = r["neff_back"]
        print(f"{tt['c_D']:>8.3f}{tt['ratio_M']:>10.4f}{tt['ratio_D']:>10.4f}"
              f"   {rm:.4f} +- {rs:.4f}{100*(rm/tt['ratio_M']-1):>+8.2f} %"
              f"{nb:>9.0f}/400")
    print("=" * 78)
    print("  N_eff(1/w) ist die SELBSTDIAGNOSE: faellt sie, wird der Schaetzer unzuverlaessig.")
    print("  Der Zusammenbruch liegt dort, wo die Antwort ohnehin 'durchgefallen' lautet.")

    # ---- Teil 3: Abdeckung — die Grenze der Methode ----
    rows_h = []
    if not args.skip_coverage:
        f = args.hole_fraction
        Ds = [2.0, 4.0, 6.0, 8.0, 12.0]
        rows_h = [run_hole(D, f, 0.33, 400, max(args.repeats // 3, 50), rng) for D in Ds]

        print(f"\nABDECKUNG — die Grenze der Methode   (n = 400, Lochanteil f = {f:.2f})")
        print("=" * 78)
        print(f"{'D [k_BT]':>9}{'Loch-Frames':>13}{'wahr':>9}{'geschaetzt':>12}"
              f"{'Bias':>13}{'N_eff(1/w)':>13}{'Warnlicht':>12}")
        print("-" * 78)
        for r in rows_h:
            bias = 100 * (r["est"] / r["truth"] - 1) if r["truth"] > 0 else float("inf")
            warn = "JA" if r["neff_back"] < 300 else "NEIN - BLIND"
            bs = f"{bias:+.0f} %" if abs(bias) < 1e6 else f"{bias:+.1e} %"
            nb_txt = f"{int(round(r['neff_back']))}/400"
            print(f"{r['D']:>9.0f}{r['hole_frames']:>13.2f}{r['truth']:>9.4f}"
                  f"{r['est']:>12.4f}{bs:>13}{nb_txt:>13}{warn:>14}")
        print("=" * 78)
        print("  Das Warnlicht ist NICHT MONOTON: es schlaegt bei mittlerer Lochtiefe an")
        print("  und geht bei tiefen Loechern wieder auf 'alles gut'. Was nicht in der")
        print("  Stichprobe ist, hinterlaesst keine Spur.")

        print(f"\n  Noetiger Testsatzumfang n >= exp(beta*D)/f, um ein Loch EINMAL zu treffen:")
        print(f"  {'D [k_BT]':>9}{'D [meV, 292K]':>15}"
              + "".join(f"{'f=' + format(x, '.2f'):>12}" for x in (0.30, 0.10, 0.03)))
        for D in (2, 4, 6, 8, 10, 12):
            cells = "".join(
                f"{(format(np.exp(D)/x, '.0f') if np.exp(D)/x < 1e7 else format(np.exp(D)/x, '.0e')):>12}"
                for x in (0.30, 0.10, 0.03))
            print(f"  {D:>9d}{D*25.16:>15.0f}{cells}")
        print(f"\n  Euer Testsatz: n = 400  ->  deckt bei f = {f:.2f} Loecher bis "
              f"D ~ {np.log(400*f):.1f} k_BT ab.")
        print("  Fuer D = 8 waeren ~10000 DFT-Rechnungen noetig — mehr als die 5000 des")
        print("  Produktionslaufs. Das Screening kostete dann mehr als die abgesicherte Sache.")
        print("\n  -> Die Abdeckungsfrage gehoert NICHT an den Testsatz, sondern an die")
        print("     Trajektorie: Komitee-sigma(R) je Frame. Dort ist sie ein billiges")
        print("     DETEKTIONS-Problem statt eines exponentiell teuren Schaetzproblems.")

    # ---- CSV ----
    f1 = args.outdir / "validate_ensemble_shift.csv"
    with open(f1, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["part", "k", "beta_theta", "n", "c_D_true", "c_M_true",
                     "ratio_M_true", "ratio_D_naive", "ratio_est", "ratio_std",
                     "bias_pct", "neff_backward"])
        for part, rows in (("convergence", rows_n), ("breakdown", rows_u)):
            for r in rows:
                tt = r["truth"]
                rm, rs = r["ratio"]
                wr.writerow([part, tt["k"], f"{tt['u']:.4f}", r["n"],
                             f"{tt['c_D']:.5f}", f"{tt['c_M']:.5f}",
                             f"{tt['ratio_M']:.5f}", f"{tt['ratio_D']:.5f}",
                             f"{rm:.5f}", f"{rs:.5f}",
                             f"{100*(rm/tt['ratio_M']-1):.3f}",
                             f"{r['neff_back'][0]:.1f}"])

    if rows_h:
        f2 = args.outdir / "validate_coverage_limit.csv"
        with open(f2, "w", newline="") as fh:
            wr = csv.writer(fh, lineterminator="\n")
            wr.writerow(["D_kT", "D_meV_292K", "hole_fraction", "n", "hole_frames_seen",
                         "ratio_true", "ratio_est", "ratio_std", "bias_pct",
                         "neff_backward", "n_required_to_detect"])
            for r in rows_h:
                bias = 100 * (r["est"] / r["truth"] - 1) if r["truth"] > 0 else ""
                wr.writerow([r["D"], f"{r['D']*25.16:.0f}", r["f"], r["n"],
                             f"{r['hole_frames']:.2f}", f"{r['truth']:.6f}",
                             f"{r['est']:.6f}", f"{r['est_sd']:.6f}",
                             f"{bias:.1f}" if bias != "" else "",
                             f"{r['neff_back']:.1f}", f"{r['n_required']:.0f}"])
        print(f"[csv  ] {f2}")

    # ---- Plot ----
    ncols = 3 if rows_h else 2
    fig, axes = plt.subplots(1, ncols, figsize=(6.9 * ncols, 5.4))
    a1, a2 = axes[0], axes[1]
    a3 = axes[2] if rows_h else None

    nn = np.array([r["n"] for r in rows_n])
    est = np.array([r["ratio"][0] for r in rows_n])
    sd = np.array([r["ratio"][1] for r in rows_n])
    a1.axhline(t["ratio_M"], color="seagreen", lw=2,
               label=f"Wahrheit unter $p_\\mathrm{{MACE}}$ = {t['ratio_M']:.4f}")
    a1.axhline(t["ratio_D"], color="crimson", ls="--", lw=1.6,
               label=f"naiv (unkorrigiert) = {t['ratio_D']:.4f}")
    a1.errorbar(nn, est, yerr=sd, fmt="o-", color="steelblue", ms=6, lw=1.6,
                capsize=4, label="Schätzer aus $p_\\mathrm{DFT}$-Stichproben")
    a1.set_xscale("log")
    a1.set_xlabel("Stichprobenumfang $n$")
    a1.set_ylabel("$N_\\mathrm{eff}/n$")
    a1.set_title("(a) Konsistenz — trifft der Schätzer die Wahrheit?")
    a1.legend(fontsize=9)
    a1.grid(alpha=0.3, which="both")

    cs = np.array([r["truth"]["c_D"] for r in rows_u])
    a2.plot(cs, [r["truth"]["ratio_M"] for r in rows_u], "-", color="seagreen",
            lw=2.2, label="Wahrheit $p_\\mathrm{MACE}$")
    a2.plot(cs, [r["truth"]["ratio_D"] for r in rows_u], "--", color="crimson",
            lw=1.6, label="naiv (Testsatz-Wert)")
    a2.errorbar(cs, [r["ratio"][0] for r in rows_u],
                yerr=[r["ratio"][1] for r in rows_u], fmt="o", color="steelblue",
                ms=7, capsize=4, label="Schätzer, $n$ = 400")
    for r in rows_u:
        a2.annotate(f"{r['neff_back'][0]:.0f}", (r["truth"]["c_D"], r["ratio"][0]),
                    textcoords="offset points", xytext=(0, -16), ha="center",
                    fontsize=7.5, color="gray")
    a2.axvline(0.327, color="k", ls=":", lw=1.2, label="L2c ($c$ = 0.327)")
    a2.set_xlabel("$c$ auf dem Testsatz")
    a2.set_ylabel("$N_\\mathrm{eff}/n$")
    a2.set_title("(b) Gültigkeitsgrenze — grau: $N_\\mathrm{eff}$ der Rückwärts-Umgewichtung")
    a2.legend(fontsize=9)
    a2.grid(alpha=0.3)

    if a3 is not None:
        Ds = np.array([r["D"] for r in rows_h])
        a3.plot(Ds, [r["truth"] for r in rows_h], "o-", color="seagreen", lw=2.2,
                ms=7, label="Wahrheit $p_\\mathrm{MACE}$")
        a3.plot(Ds, [r["est"] for r in rows_h], "s--", color="steelblue", lw=1.8,
                ms=7, label="Schätzer aus dem Testsatz")
        a3.set_yscale("log")
        a3.set_xlabel("Lochtiefe $D$ [$k_BT$]")
        a3.set_ylabel("$N_\\mathrm{eff}/n$  (log)")
        a3.set_title("(c) Abdeckungsgrenze — wo die Methode blind wird")
        a3.grid(alpha=0.3, which="both")

        ax4 = a3.twinx()
        ax4.plot(Ds, [r["neff_back"] for r in rows_h], "^:", color="crimson",
                 lw=1.5, ms=8, label="$N_\\mathrm{eff}(1/w)$ — Warnlicht")
        ax4.axhline(300, color="crimson", ls=":", lw=1, alpha=0.5)
        ax4.set_ylabel("$N_\\mathrm{eff}(1/w)$ von 400", color="crimson")
        ax4.tick_params(axis="y", labelcolor="crimson")
        ax4.set_ylim(0, 420)

        h1, l1 = a3.get_legend_handles_labels()
        h2, l2 = ax4.get_legend_handles_labels()
        a3.legend(h1 + h2, l1 + l2, fontsize=8.5, loc="lower left")
        a3.annotate("Warnlicht geht wieder\nauf »alles gut« —\nMethode blind",
                    xy=(9.5, 0.4), fontsize=8.5, color="crimson", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="crimson", lw=0.8))

    fig.suptitle(
        f"Validierung der Ensemble-Korrektur an analytisch bekannter Wahrheit\n"
        f"Gamma-Modell (unter exponentiellem Tilten geschlossen), "
        f"$k$ = {args.k}, $\\gamma_1$ = {t['gamma1']:.2f}"
        + (f"   |   (c) Loch-Modell, Anteil $f$ = {args.hole_fraction:.2f}" if rows_h else ""),
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = args.outdir / "validate_ensemble_shift.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")
    print(f"[csv  ] {f1}")


if __name__ == "__main__":
    main()
