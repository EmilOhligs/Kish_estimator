"""
Screening: aus dem Testsatz auf den Produktionslauf schliessen — vor der MD
============================================================================

DIE IDEE. Die DFT-Referenz des Testsatzes existiert bereits (sie fiel beim
Erzeugen der Trainingsdaten an). Man bekommt also

    c = beta * std(dE),    dE = E_DFT - E_MACE

fuer NULL neue DFT-Rechnungen, sobald ein Modell trainiert ist. Ist c zu gross,
bricht man ab, bevor MD und 5000 Einzelpunkte gerechnet werden. Ohne diesen
Schritt erfaehrt man erst am Ende, dass das Modell nicht traegt.

Der Schritt liefert allerdings c fuer den TESTSATZ. Gebraucht wird c fuer den
Produktionslauf, und der unterscheidet sich in zwei Punkten:

  (1) ENSEMBLE. Der Testsatz ist (Annahme) DFT-gesampelt, die Produktions-MD
      laeuft auf MACE. Korrektur: uq_mace.reweighting.ensemble_shift, exakt per
      Rueckwaerts-Umgewichtung mit 1/w. Fuer L2c sind das rund +9 % auf c.
      Richtung: die Testsatz-Schaetzung ist OPTIMISTISCH.

  (2) SYSTEMGROESSE. Der Testsatz hat 63 Molekuele, das Paper rechnet bis 128.
      Skalierung c ∝ sqrt(N) unter der Blockunabhaengigkeits-Annahme
      (siehe 11_error_correlation).

VALIDIERUNG. Beides zusammen auf 128 Molekuele angewandt ergibt eine Vorhersage
fuer N_eff/n, die sich gegen den im Paper GEMESSENEN Wert halten laesst
(Hilpert & Kresse 2026: 0.814 bei 128 Molekuelen, aus der echten MACE-MD mit
5000 Frames). Das ist der einzige unabhaengige Test, ob die Extrapolation vom
Testsatz auf den Produktionslauf ueberhaupt traegt.

WAS DAS SKRIPT NICHT LEISTET. Es sagt nichts ueber tau_corr der Trajektorie -
die 5000 Produktionsframes sind nicht unabhaengig, und der Kish-Schaetzer weiss
davon nichts. Und es sagt nichts darueber, wie gross N_eff sein MUSS; das haengt
an der geforderten Genauigkeit der thermophysikalischen Groessen.

Ausfuehren:
    python analyses/12_screening/screening_extrapolation.py
    python analyses/12_screening/screening_extrapolation.py --temperature 298
    python analyses/12_screening/screening_extrapolation.py --n-molecules 63 \
        --paper-n 128 --paper-ratio 0.814
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
from uq_mace.reweighting import (
    effective_sample_size,
    ensemble_shift,
    neff_ratio_cumulant,
    psis_khat,
    reweighting_weights,
    scale_to_system_size,
)

HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "cache"
K_B = 8.617333262e-5  # eV/K


def tidy(name: str) -> str:
    for a, b in (("single_", ""), ("mace_energies_", ""), ("predictions_", ""),
                 ("_testbig", ""), ("_testsmall", "")):
        name = name.replace(a, b)
    return name


def analyse(path: Path, beta: float) -> dict:
    e_dft, e_mace = load_energies(path)
    dE = e_dft - e_mace
    r = ensemble_shift(dE, beta)

    w = reweighting_weights(e_dft, e_mace, beta)
    r["neff_ratio_measured"] = effective_sample_size(w) / dE.size
    r["khat"] = psis_khat(w)
    r["name"] = tidy(path.stem)
    r["rel_error"] = (r["c_first_order"] - r["c_exact"]) / r["c_exact"] * 100
    r["shift_pct"] = (r["c_exact"] / r["c_test"] - 1.0) * 100
    return r


def collect(beta: float, explicit: list[str] | None) -> list[dict]:
    if explicit:
        paths = [Path(p) for p in explicit]
    else:
        paths, seen = [], set()
        for p in sorted(CACHE.glob("*.npz")):
            try:
                if "e_dft" not in np.load(p, allow_pickle=True).files:
                    continue
            except Exception:
                continue
            # predictions_* und mace_energies_* sind fuer dasselbe Ensemble
            # identisch -> nur einmal auswerten
            key = tidy(p.stem)
            if key in seen:
                continue
            seen.add(key)
            paths.append(p)
    if not paths:
        raise SystemExit(f"Keine auswertbaren Caches in {CACHE}.")
    return sorted((analyse(p, beta) for p in paths), key=lambda r: r["c_test"])


def size_series(row: dict, n_from: int, targets) -> list[dict]:
    """N_eff/n als Funktion der Systemgroesse, ausgehend von c_exact."""
    out = []
    for n_to in targets:
        s = scale_to_system_size(row["c_exact"], row["gamma1"], row["gamma2"],
                                 n_from, n_to)
        s["ratio_gauss"] = neff_ratio_cumulant(s["c"])
        s["ratio_skew"] = neff_ratio_cumulant(s["c"], s["gamma1"], s["gamma2"])
        s["rest"] = (2 * s["c"]) ** 5 / 120.0      # Restabschaetzung, kappa_5 ~ 1
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--caches", nargs="+", default=None)
    ap.add_argument("--temperature", type=float, default=292.0)
    ap.add_argument("--n-molecules", type=int, default=63,
                    help="Molekuele im Testsatz (water_testset_big: 63)")
    ap.add_argument("--reference", default="ensemble_L2c",
                    help="Modell fuer die Systemgroessen-Extrapolation")
    ap.add_argument("--paper-n", type=int, default=128)
    ap.add_argument("--paper-ratio", type=float, default=0.814,
                    help="im Paper gemessenes N_eff/n bei --paper-n Molekuelen")
    ap.add_argument("--outdir", type=Path, default=HERE)
    args = ap.parse_args()

    beta = 1.0 / (K_B * args.temperature)
    rows = collect(beta, args.caches)
    args.outdir.mkdir(parents=True, exist_ok=True)

    # ---- Teil 1: Ensemble-Korrektur ----
    print(f"\nT = {args.temperature:.0f} K,  beta = {beta:.2f} eV^-1,  "
          f"Testsatz = {args.n_molecules} Molekuele")
    print("=" * 100)
    print("TEIL 1 — Ensemble-Korrektur:  Testsatz (p_DFT)  ->  Produktion (p_MACE)")
    print("-" * 100)
    print(f"{'Modell':<20}{'c_test':>8}{'gamma1':>9}{'c exakt':>10}{'c 1.Ord':>10}"
          f"{'Verschieb.':>12}{'Fehler':>9}{'N_eff rueckw.':>15}")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<20}{r['c_test']:>8.3f}{r['gamma1']:>+9.3f}"
              f"{r['c_exact']:>10.3f}{r['c_first_order']:>10.3f}"
              f"{r['shift_pct']:>+11.1f}%{r['rel_error']:>+8.1f}%"
              f"{r['neff_backward']:>10.0f}/{r['n']:<4d}")
    print("=" * 100)
    print("  Verschieb. = (c_exakt/c_test - 1), also wie stark das MACE-Ensemble")
    print("               die dE-Verteilung verbreitert.  Positiv = Testsatz zu optimistisch.")
    print("  Fehler     = Abweichung der Entwicklung 1. Ordnung c(1+g1*c/2) vom exakten Wert.")
    print("  N_eff rueckw. = Guete der Rueckwaerts-Umgewichtung selbst. Klein -> c_exakt verrauscht.")

    # ---- Teil 2: Systemgroessen-Extrapolation ----
    ref = next((r for r in rows if r["name"] == args.reference), None)
    if ref is None:
        ref = rows[0]
        print(f"\n  ({args.reference} nicht gefunden, nehme {ref['name']})")

    targets = sorted({args.n_molecules, 96, args.paper_n, 160, 192, 256})
    series = size_series(ref, args.n_molecules, targets)

    print(f"\nTEIL 2 — Systemgroessen-Extrapolation, ausgehend von {ref['name']} "
          f"(c_exakt = {ref['c_exact']:.3f})")
    print("=" * 100)
    print(f"{'N [Molek.]':>11}{'c':>8}{'gamma1':>9}{'N_eff/n Gauss':>15}"
          f"{'mit Schiefe':>13}{'Rest (2c)^5/5!':>16}")
    print("-" * 100)
    for s in series:
        mark = "   <- Paper" if s["n_molecules"] == args.paper_n else ""
        print(f"{s['n_molecules']:>11d}{s['c']:>8.3f}{s['gamma1']:>9.3f}"
              f"{s['ratio_gauss']:>15.3f}{s['ratio_skew']:>13.3f}"
              f"{s['rest']:>16.3f}{mark}")
    print("=" * 100)

    hit = next((s for s in series if s["n_molecules"] == args.paper_n), None)
    if hit:
        lo, hi = sorted((hit["ratio_gauss"], hit["ratio_skew"]))
        dev = 100 * (hit["ratio_skew"] - args.paper_ratio) / args.paper_ratio
        print(f"\nVALIDIERUNG bei {args.paper_n} Molekuelen:")
        print(f"  Vorhersage aus dem Testsatz : {lo:.3f} (Gauss) ... {hi:.3f} (mit Schiefe)")
        print(f"  im Paper gemessen           : {args.paper_ratio:.3f}")
        print(f"  Abweichung zum besseren (Schiefe-)Wert: {dev:+.1f} %")
        print("\n  Einordnung: die Spanne Gauss...Schiefe ist KEIN Konfidenzintervall,")
        print("  sondern zeigt nur, wie stark die Formkorrektur eingreift. Massgeblich")
        print("  ist der Schiefe-Wert; die Gauss-Zahl steht als konservative Schranke daneben.")
        if abs(dev) < 5:
            print(f"  {abs(dev):.1f} % Abweichung bei einer Extrapolation ueber Faktor "
                  f"{args.paper_n/args.n_molecules:.1f} in der Systemgroesse.")
        else:
            print(f"  {abs(dev):.1f} % Abweichung — gross genug, um die sqrt(N)-Annahme")
            print("  oder die Ensemble-Korrektur in Frage zu stellen. Nachpruefen.")
        print("\n  Ein einzelner Datenpunkt. Er zeigt, dass die Groessenordnung stimmt,")
        print("  nicht, dass die Extrapolation allgemein traegt.")

    # ---- CSV ----
    f1 = args.outdir / "screening_ensemble_shift.csv"
    with open(f1, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["model", "n", "c_test", "gamma1", "gamma2", "c_exact",
                     "c_first_order", "shift_pct", "rel_error_pct",
                     "neff_backward", "neff_ratio_measured", "khat"])
        for r in rows:
            wr.writerow([r["name"], r["n"], f"{r['c_test']:.5f}", f"{r['gamma1']:.5f}",
                         f"{r['gamma2']:.5f}", f"{r['c_exact']:.5f}",
                         f"{r['c_first_order']:.5f}", f"{r['shift_pct']:.3f}",
                         f"{r['rel_error']:.3f}", f"{r['neff_backward']:.1f}",
                         f"{r['neff_ratio_measured']:.5f}", f"{r['khat']:.4f}"])

    f2 = args.outdir / "screening_size_scaling.csv"
    with open(f2, "w", newline="") as fh:
        wr = csv.writer(fh, lineterminator="\n")
        wr.writerow(["n_molecules", "c", "gamma1", "gamma2",
                     "neff_ratio_gauss", "neff_ratio_skew", "series_rest"])
        for s in series:
            wr.writerow([s["n_molecules"], f"{s['c']:.5f}", f"{s['gamma1']:.5f}",
                         f"{s['gamma2']:.5f}", f"{s['ratio_gauss']:.5f}",
                         f"{s['ratio_skew']:.5f}", f"{s['rest']:.5f}"])

    # ---- Plot ----
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.6))

    x = np.arange(len(rows))
    wdt = 0.27
    a1.bar(x - wdt, [r["c_test"] for r in rows], wdt, label="$c$ Testsatz ($p_\\mathrm{DFT}$)",
           color="steelblue")
    a1.bar(x, [r["c_exact"] for r in rows], wdt, label="$c$ exakt umgewichtet ($p_\\mathrm{MACE}$)",
           color="darkorange")
    a1.bar(x + wdt, [r["c_first_order"] for r in rows], wdt,
           label="$c\\,(1+\\frac{1}{2}\\gamma_1 c)$  1. Ordnung", color="seagreen")
    for i, r in enumerate(rows):
        a1.text(i, max(r["c_exact"], r["c_first_order"]) * 1.02,
                f"{r['shift_pct']:+.0f} %", ha="center", fontsize=8.5, fontweight="bold")
    a1.set_xticks(x)
    a1.set_xticklabels([r["name"] for r in rows], rotation=20, ha="right", fontsize=8.5)
    a1.set_ylabel("$c = \\beta\\,\\mathrm{std}(\\Delta E)$")
    a1.set_title("(a) Ensemble-Korrektur — und wie gut die 1. Ordnung sie trifft")
    a1.legend(fontsize=8.5)
    a1.grid(alpha=0.3, axis="y")

    nn = np.array([s["n_molecules"] for s in series])
    a2.plot(nn, [s["ratio_gauss"] for s in series], "o--", color="steelblue",
            lw=1.5, ms=5, label="Gauss  $e^{-c^2}$")
    a2.plot(nn, [s["ratio_skew"] for s in series], "s-", color="darkorange",
            lw=1.8, ms=6, label="mit $\\gamma_1 c^3 - \\frac{7}{12}\\gamma_2 c^4$")
    a2.fill_between(nn, [s["ratio_gauss"] for s in series],
                    [s["ratio_skew"] for s in series], color="orange", alpha=0.18,
                    label="Vorhersagebereich")
    a2.plot([args.paper_n], [args.paper_ratio], "*", color="crimson", ms=18,
            label=f"Paper, gemessen ({args.paper_ratio:.3f})", zorder=5)
    a2.axhline(0.8, color="k", ls=":", lw=1.2, label="Effizienzschwelle $R=0.8$")
    a2.axvline(args.n_molecules, color="gray", ls="--", lw=1,
               label=f"Testsatz ({args.n_molecules} Mol.)")
    a2.set_xlabel("Systemgröße $N$ [Moleküle]")
    a2.set_ylabel("$N_\\mathrm{eff}/n$")
    a2.set_title("(b) Extrapolation $c \\propto \\sqrt{N}$ — validiert am Paper-Wert")
    a2.legend(fontsize=8.5, loc="lower left")
    a2.grid(alpha=0.3)

    fig.suptitle(
        f"Screening vor der MD: Testsatz-$c$ → Produktionsprognose   |   "
        f"T = {args.temperature:.0f} K, Referenz {ref['name']}\n"
        f"Ensemble-Korrektur exakt (Rückwärts-Umgewichtung mit $1/w$); "
        f"$\\sqrt{{N}}$-Skalierung ist eine Annahme, keine Messung",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    out = args.outdir / "screening_extrapolation.png"
    fig.savefig(out, dpi=140)
    print(f"\n[plot ] {out}")
    print(f"[csv  ] {f1}")
    print(f"[csv  ] {f2}")


if __name__ == "__main__":
    main()
